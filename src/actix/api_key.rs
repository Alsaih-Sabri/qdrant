use std::future::{ready, Ready};

use actix_web::body::{BoxBody, EitherBody};
use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::http::header::Header;
use actix_web::http::Method;
use actix_web::{Error, HttpResponse};
use actix_web_httpauth::headers::authorization::{Authorization, Bearer};
use futures_util::future::LocalBoxFuture;
use regex::Regex;

use crate::common::auth::AuthKeys;

/// List of read-only POST request paths. List MUST be sorted.
const READ_ONLY_POST_PATTERNS: [&str; 11] = [
    "/collections/{name}/points",
    "/collections/{name}/points/count",
    "/collections/{name}/points/discover",
    "/collections/{name}/points/discover/batch",
    "/collections/{name}/points/recommend",
    "/collections/{name}/points/recommend/batch",
    "/collections/{name}/points/recommend/groups",
    "/collections/{name}/points/scroll",
    "/collections/{name}/points/search",
    "/collections/{name}/points/search/batch",
    "/collections/{name}/points/search/groups",
];

pub struct ApiKey {
    auth_keys: Option<AuthKeys>,
    whitelist: Vec<WhitelistItem>,
}

impl ApiKey {
    pub fn new(auth_keys: Option<AuthKeys>, whitelist: Vec<WhitelistItem>) -> Self {
        Self {
            auth_keys,
            whitelist,
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for ApiKey
where
    S: Service<ServiceRequest, Response = ServiceResponse<EitherBody<B, BoxBody>>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B, BoxBody>>;
    type Error = Error;
    type InitError = ();
    type Transform = ApiKeyMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyMiddleware {
            auth_keys: self.auth_keys.clone(),
            whitelist: self.whitelist.clone(),
            service,
        }))
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct WhitelistItem(pub String, pub PathMode);

impl WhitelistItem {
    pub fn exact<S: Into<String>>(path: S) -> Self {
        Self(path.into(), PathMode::Exact)
    }

    pub fn prefix<S: Into<String>>(path: S) -> Self {
        Self(path.into(), PathMode::Prefix)
    }

    pub fn matches(&self, other: &str) -> bool {
        self.1.check(&self.0, other)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum PathMode {
    /// Path must match exactly
    Exact,
    /// Path must have given prefix
    Prefix,
}

impl PathMode {
    fn check(&self, key: &str, other: &str) -> bool {
        match self {
            Self::Exact => key == other,
            Self::Prefix => other.starts_with(key),
        }
    }
}

pub struct ApiKeyMiddleware<S> {
    auth_keys: Option<AuthKeys>,
    /// List of items whitelisted from authentication.
    whitelist: Vec<WhitelistItem>,
    service: S,
}

impl<S> ApiKeyMiddleware<S> {
    pub fn is_path_whitelisted(&self, path: &str) -> bool {
        self.whitelist.iter().any(|item| item.matches(path))
    }
}

impl<S, B> Service<ServiceRequest> for ApiKeyMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<EitherBody<B, BoxBody>>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B, BoxBody>>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let path = req.path();

        if self.is_path_whitelisted(path) {
            return Box::pin(self.service.call(req));
        }

        // Grab API key from request
        let key =
            // Request header
            req.headers().get("api-key").and_then(|key| key.to_str().ok()).map(|key| key.to_string())
                // Fall back to authentication header with bearer token
                .or_else(|| {
                    Authorization::<Bearer>::parse(&req).ok().map(|auth| auth.as_ref().token().into())
                });

        if let Some(key) = key {
            let is_allowed = if let Some(ref auth_keys) = self.auth_keys {
                auth_keys.can_write(&key) || (is_read_only(&req) && auth_keys.can_read(&key))
            } else {
                // This code path should not be reached
                log::warn!("Auth for REST API is set up incorrectly. Denying access by default.");
                false
            };
            if is_allowed {
                return Box::pin(self.service.call(req));
            }
        }

        Box::pin(async {
            Ok(req
                .into_response(HttpResponse::Forbidden().body("Invalid api-key"))
                .map_into_right_body())
        })
    }
}

/// Matches a request path against a specified pattern.
///
/// This function transforms placeholders (`{}`) in the given pattern to a regex that matches any non-slash sequence.
/// It returns `true` if the pattern matches the request path, otherwise `false` if the pattern is invalid or does not match.
/// ```
/// let is_matched = match_pattern_to_request_path("/user/123/profile", "/user/{id}/profile");
/// assert!(is_matched, "The path should match the pattern.");
/// ```
/// This example checks if the path `/user/123/profile` matches the pattern `/user/{id}/profile`.
fn match_pattern_to_request_path(req_path: &str, pattern: &str) -> bool {
    let regex_result = Regex::new(&pattern.replace('{', "(?P<").replace('}', ">[^/]+)"));
    match regex_result {
        Ok(regex_pattern) => regex_pattern.is_match(req_path),
        Err(_) => false,
    }
}

fn is_read_only(req: &ServiceRequest) -> bool {
    match *req.method() {
        Method::GET => true,
        Method::POST => READ_ONLY_POST_PATTERNS
            .iter()
            .any(|&pattern| match_pattern_to_request_path(req.path(), pattern)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_post_patterns_sorted() {
        let mut sorted = READ_ONLY_POST_PATTERNS;
        sorted.sort_unstable();
        assert_eq!(
            READ_ONLY_POST_PATTERNS, sorted,
            "The READ_ONLY_POST_PATTERNS list must be sorted"
        );
    }

    #[test]
    fn test_match_pattern_to_request_path() {
        let test_cases = vec![
            (
                "/collections/test_collection/points",
                "/collections/{any_place_holder}/points",
                true,
            ),
            (
                "/collections/test_collection/points/count",
                "/collections/{collection_name}/points/count",
                true,
            ),
            (
                "/collections/test_collection/points/discover",
                "/collections/{name}/points/discover",
                true,
            ),
            (
                "/collections/test_collection/points/discover/batch",
                "/collections/{name}/points/discover/batch",
                true,
            ),
            (
                "/collections/test_collection/points/recommend",
                "/collections/{name}/points/recommend",
                true,
            ),
            (
                "/collections/test_collection/points/recommend/batch",
                "/collections/{name}/points/recommend/batch",
                true,
            ),
            (
                "/collections/test_collection/points/recommend/groups",
                "/collections/{name}/points/recommend/groups",
                true,
            ),
            (
                "/collections/test_collection/points/scroll",
                "/collections/{name}/points/scroll",
                true,
            ),
            (
                "/collections/test_collection/points/search",
                "/collections/{name}/points/search",
                true,
            ),
            (
                "/collections/test_collection/points/search/batch",
                "/collections/{name}/points/search/batch",
                true,
            ),
            (
                "/collections/test_collection/points/search/groups",
                "/collections/{name}/points/search/groups",
                true,
            ),
            (
                "/collections/test_collection/points/123",
                "/collections/{any_name}/points/{id}",
                true,
            ),
            (
                "/collections/some_collection/not_match",
                "/collections/{test_collection}/points",
                false,
            ),
            (
                "/collections/some_collection/points/not_match",
                "/collections/{test_collection}/points/scroll",
                false,
            ),
        ];

        for (req_path, pattern, expected) in test_cases {
            assert_eq!(
                match_pattern_to_request_path(req_path, pattern),
                expected,
                "Failed matching req_path: '{}' with pattern: '{}'",
                req_path,
                pattern
            );
        }
    }
}
