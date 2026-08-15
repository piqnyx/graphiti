# piqnyx reliable episode queue

The production MCP entrypoint replaces the upstream per-group queue class with a
compatible reliability overlay.

Behavior:

- the active episode is retried in place before any later episode for that group;
- retries use exponential backoff, bounded by configurable limits;
- after the maximum number of attempts, only that `group_id` is blocked;
- already-pending episodes stay queued and are not allowed to leapfrog the failed
  predecessor;
- new enqueue attempts for a blocked group fail fast with the predecessor error;
- other groups continue independently;
- an operator can retry the blocked predecessor, and success resumes the preserved
  FIFO queue behind it.

Defaults:

```text
EPISODE_PROCESS_MAX_ATTEMPTS=5
EPISODE_PROCESS_RETRY_BASE_SECONDS=2
EPISODE_PROCESS_RETRY_MAX_SECONDS=30
```
