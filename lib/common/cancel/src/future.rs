use std::future::Future;

use super::*;

pub async fn spawn_cancel_on_drop<Task, Fut>(task: Task) -> Result<Fut::Output, Error>
where
    Task: FnOnce(CancellationToken) -> Fut,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    let cancel = CancellationToken::new();

    let future = task(cancel.child_token());

    let guard = cancel.drop_guard();
    let output = tokio::task::spawn(future).await?;
    guard.disarm();

    Ok(output)
}

/// # Safety
///
/// Future has to be cancel-safe!
pub async fn cancel_on_token<Fut>(cancel: CancellationToken, future: Fut) -> Result<Fut::Output, Error>
where
    Fut: Future,
{
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(Error::Cancelled),
        output = future => Ok(output),
    }
}
