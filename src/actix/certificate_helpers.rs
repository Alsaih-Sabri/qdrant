use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rustls::server::{AllowAnyAuthenticatedClient, ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use rustls::{Certificate, RootCertStore, ServerConfig};
use rustls_pemfile::Item;

use super::with_buf_read;
use crate::settings::{Settings, TlsConfig};

/// A TTL based rotating server certificate resolver
struct RotatingCertificateResolver {
    /// TLS configuration used for loading/refreshing certified key
    tls_config: TlsConfig,

    /// TTL for each rotation
    ttl: Option<Duration>,

    /// Current certified key
    key: RwLock<CertifiedKeyWithAge>,
}

impl RotatingCertificateResolver {
    pub fn new(tls_config: TlsConfig, ttl: Option<Duration>) -> io::Result<Self> {
        let certified_key = load_certified_key(&tls_config)?;

        Ok(Self {
            tls_config,
            ttl,
            key: RwLock::new(CertifiedKeyWithAge::from(certified_key)),
        })
    }

    /// Get certificate key or refresh
    ///
    /// The key is automatically refreshed when the TTL is reached.
    /// If refreshing fails, an error is logged and the old key is persisted.
    fn get_key_or_refresh(&self) -> Arc<CertifiedKey> {
        // Get read-only lock to the key. If TTL is not configured or is not expired, return key.
        let ttl = {
            let key = self.key.read();

             match self.ttl {
                Some(ttl) if key.is_expired(ttl) => ttl,
                _ => return key.key.clone(),
            }
        };

        // If TTL is expired:
        // - get read-write lock to the key
        // - *re-check that TTL is expired* (to avoid refreshing the key multiple times from concurrent threads)
        // - refresh and return the key
        let mut key = self.key.write();

        if key.is_expired(ttl) {
            if let Err(err) = key.refresh(&self.tls_config) {
                log::error!("Failed to refresh TLS certificate, keeping current: {err}");
            }
        }

        key.key.clone()
    }
}

impl ResolvesServerCert for RotatingCertificateResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.get_key_or_refresh())
    }
}

struct CertifiedKeyWithAge {
    /// Last time the certificate was updated/replaced
    last_update: Instant,

    /// Current certified key
    key: Arc<CertifiedKey>,
}

impl CertifiedKeyWithAge {
    pub fn from(key: Arc<CertifiedKey>) -> Self {
        Self {
            last_update: Instant::now(),
            key,
        }
    }

    pub fn refresh(&mut self, tls_config: &TlsConfig) -> io::Result<()> {
        *self = Self::from(load_certified_key(tls_config)?);
        Ok(())
    }

    pub fn age(&self) -> Duration {
        self.last_update.elapsed()
    }

    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.age() >= ttl
    }
}

/// Load TLS configuration and construct certified key.
fn load_certified_key(tls_config: &TlsConfig) -> io::Result<Arc<CertifiedKey>> {
    // Load certificates
    let certs: Vec<Certificate> = with_buf_read(&tls_config.cert, rustls_pemfile::read_all)?
        .into_iter()
        .filter_map(|item| match item {
            Item::X509Certificate(data) => Some(Certificate(data)),
            _ => None,
        })
        .collect();
    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "No server certificate found",
        ));
    }

    // Load private key
    let private_key_item = with_buf_read(&tls_config.key, rustls_pemfile::read_one)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No private key found"))?;
    let (Item::RSAKey(pkey) | Item::PKCS8Key(pkey) | Item::ECKey(pkey)) = private_key_item else {
        return Err(io::Error::new(io::ErrorKind::Other, "No private key found"))
    };
    let private_key = rustls::PrivateKey(pkey);
    let signing_key = rustls::sign::any_supported_type(&private_key)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    // Construct certified key
    let certified_key = CertifiedKey::new(certs, signing_key);
    Ok(Arc::new(certified_key))
}

/// Generate an actix server configuration with TLS
///
/// Uses TLS settings as configured in configuration by user.
pub fn actix_tls_server_config(settings: &Settings) -> io::Result<ServerConfig> {
    let config = ServerConfig::builder().with_safe_defaults();
    let tls_config = settings
        .tls
        .clone()
        .ok_or_else(Settings::tls_config_is_undefined_error)?;

    // Verify client CA or not
    let config = if settings.service.verify_https_client_certificate {
        let mut root_cert_store = RootCertStore::empty();
        let ca_certs: Vec<Vec<u8>> = with_buf_read(&tls_config.ca_cert, rustls_pemfile::certs)?;
        root_cert_store.add_parsable_certificates(&ca_certs[..]);
        config.with_client_cert_verifier(AllowAnyAuthenticatedClient::new(root_cert_store))
    } else {
        config.with_no_client_auth()
    };

    // Configure rotating certificate resolver
    let ttl = match tls_config.cert_ttl {
        None | Some(0) => None,
        Some(seconds) => Some(Duration::from_secs(seconds)),
    };
    let cert_resolver = RotatingCertificateResolver::new(tls_config, ttl)?;
    let config = config.with_cert_resolver(Arc::new(cert_resolver));

    Ok(config)
}
