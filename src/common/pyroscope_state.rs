#[cfg(target_os = "linux")]
pub mod pyro {

    use pyroscope::pyroscope::PyroscopeAgentRunning;
    use pyroscope::{PyroscopeAgent, PyroscopeError};
    use pyroscope_pprofrs::{pprof_backend, PprofConfig};

    use crate::common::debug::PyroscopeConfig;

    pub struct PyroscopeState {
        pub config: PyroscopeConfig,
        pub agent: Option<PyroscopeAgent<PyroscopeAgentRunning>>,
    }

    impl PyroscopeState {
        fn build_agent(
            config: &PyroscopeConfig,
        ) -> Result<PyroscopeAgent<PyroscopeAgentRunning>, PyroscopeError> {
            let pprof_config = PprofConfig::new().sample_rate(config.sampling_rate.unwrap_or(100));
            let backend_impl = pprof_backend(pprof_config);

            log::info!(
                "Starting pyroscope agent with identifier {}",
                &config.identifier
            );
            // TODO: Add more tags like peerId and peerUrl
            let agent = PyroscopeAgent::builder(config.url.to_string(), "qdrant".to_string())
                .backend(backend_impl)
                .tags(vec![("app", "Qdrant"), ("identifier", &config.identifier)])
                .build()?;
            let running_agent = agent.start()?;

            Ok(running_agent)
        }

        pub fn from_config(config: Option<PyroscopeConfig>) -> Option<Self> {
            match config {
                Some(pyro_config) => {
                    let agent = PyroscopeState::build_agent(&pyro_config);
                    match agent {
                        Ok(agent) => Some(PyroscopeState {
                            config: pyro_config.clone(),
                            agent: Some(agent),
                        }),
                        Err(err) => {
                            log::warn!("Pyroscope agent failed to start {}", err);
                            None
                        }
                    }
                }
                None => {
                    log::info!("Pyroscope agent not started");
                    None
                }
            }
        }

        pub fn stop_agent(&mut self) -> bool {
            log::info!("STOP: Stopping pyroscope agent");
            if let Some(agent) = self.agent.take() {
                match agent.stop() {
                    Ok(stopped_agent) => {
                        stopped_agent.shutdown();
                        log::info!("STOP: Pyroscope agent stopped");
                        return true;
                    }
                    Err(err) => {
                        log::warn!("Pyroscope agent failed to stop {}", err);
                        return false;
                    }
                }
            }
            true
        }
    }

    impl Drop for PyroscopeState {
        fn drop(&mut self) {
            log::info!("DROP: Stopping pyroscope agent");
            if let Some(agent) = self.agent.take() {
                let stopped_agent = agent.stop().unwrap(); // Now you can call stop() on the agent directly
                stopped_agent.shutdown();
            }
            log::info!("DROP: Pyroscope agent stopped");
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub mod pyro {
    pub struct PyroscopeState {}

    impl PyroscopeState {
        pub fn from_config(config: Option<PyroscopeConfig>) -> Option<Self> {
            None
        }
    }
}
