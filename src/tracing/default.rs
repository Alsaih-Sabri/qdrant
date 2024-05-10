use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{filter, fmt, registry};

use super::*;
use crate::tracing::config::SpanEvent;

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub log_level: Option<String>,
    pub span_events: Option<HashSet<SpanEvent>>,
    pub color: Option<config::Color>,
}

#[rustfmt::skip] // `rustfmt` formats this into unreadable single line
pub type Logger<S> = filter::Filtered<
    Option<fmt::Layer<S>>,
    filter::EnvFilter,
    S,
>;

pub fn new_layer<S>(config: &Config) -> fmt::Layer<S>
where
    S: tracing::Subscriber + for<'span> registry::LookupSpan<'span>,
{
    let span_events = config
        .span_events
        .as_ref()
        .unwrap_or(&HashSet::new())
        .iter()
        .copied()
        .fold(fmt::format::FmtSpan::NONE, |events, event| {
            events | event.to_fmt_span()
        });

    fmt::Layer::default()
        .with_ansi(config.color.unwrap_or_default().to_bool())
        .with_span_events(span_events)
}

pub fn new_filter(config: &Config) -> filter::EnvFilter {
    filter(config.log_level.as_deref().unwrap_or(""))
}

pub fn new_logger<S>(config: &Config) -> Logger<S>
where
    S: tracing::Subscriber + for<'span> registry::LookupSpan<'span>,
{
    let layer = new_layer(config);
    let filter = new_filter(config);
    Some(layer).with_filter(filter)
}
