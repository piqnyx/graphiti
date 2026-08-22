# piqnyx: moving between gateways

What it takes to point this fork at a different OpenAI-compatible gateway, and
what was measured rather than assumed. Written 2026-08-22 while moving from
opencode.ai (mimo-v2.5) to Google (gemini-3.5-flash-lite) behind a local
key-rotating proxy.

Every claim below is a request that was actually sent. Where a number appears, it
came back from the gateway; where a code appears, that is what it answered.

## The schemas travel unchanged

The largest risk in the move was `response_format`. This fork builds it from raw
`model_json_schema()`, which routinely contains `$ref`, `$defs`, `anyOf` and
`default` — none of which every gateway accepts.

All six response models used in extraction were sent to Google with a real
prompt:

| Model | Result |
|---|---|
| `extract_nodes.ExtractedEntities` | 200, parsed, `finish_reason=stop` |
| `extract_edges.ExtractedEdges` | 200, parsed, `finish_reason=stop` |
| `dedupe_nodes.NodeResolutions` | 200, parsed, `finish_reason=stop` |
| `dedupe_edges.EdgeDuplicate` | 200, parsed, `finish_reason=stop` |
| `extract_edges.BatchEdgeTimestamps` | 200, parsed, `finish_reason=stop` |
| `extract_nodes.SummarizedEntities` | 200, parsed, `finish_reason=stop` |

So `structured_output_mode` stays on `json_schema`. The `json_object` fallback
exists for gateways that refuse the schema; this one does not.

## The two gateways hear different reasoning switches

This is the only incompatibility the move actually has.

| Sent | opencode.ai | Google |
|---|---|---|
| `extra_body: {"thinking": {"type": "enabled"}}` | honoured | `400 Unknown name "thinking"` |
| `reasoning_effort` | ignored at every value | honoured, alongside `response_format` |

Measured on Google with one trivial prompt, so the numbers compare only to each
other: `medium` spent 736 tokens, `low` 120, `minimal` 20, and sending nothing at
all also 20. `minimal` is therefore how a stage says "do not think" — `off` is
not a valid level and a zero thinking budget is rejected outright.

Both switches are resolvable per stage, and both fall back to a global value:

- `GRAPHITI_THINKING` / `GRAPHITI_THINKING_BY_PROMPT`
- `GRAPHITI_REASONING_EFFORT` / `GRAPHITI_REASONING_EFFORT_BY_PROMPT`

A deployment sets whichever its gateway hears. The unset one sends nothing, which
is why the two can be swapped without touching code.

The per-stage split matters regardless of gateway: measured 2026-08-20 by
replaying recorded requests, the contradiction judge (`dedupe_edges.resolve_edge`)
scored 14 of 15 with reasoning against 9 of 15 without, while entity
deduplication (`dedupe_nodes`) with reasoning began returning merges where
`duplicate_candidate_id == id` — the model copying the row number instead of
choosing.

## Reasoning is billed against the answer

On Google the thinking tokens come out of the same `max_tokens` as the response.
A stage with a tight budget and reasoning enabled can therefore exhaust it before
producing content.

That failure is not silent: `generate_response` raises `OutputLimitError`,
distinguishing three shapes of it — an explicit `finish_reason == 'length'`, an
empty body with the budget spent, and malformed or incomplete JSON with the
budget spent. Any of them is a signal to lower the reasoning level for that stage
rather than to raise the budget blindly.

## What each failure does

Retries are bounded at 4 attempts with randomized exponential backoff (5s to
120s), and only for exceptions `is_server_or_retry_error` accepts.

| Gateway answer | Behaviour |
|---|---|
| 429 | retried — arrives as `RateLimitError` |
| 5xx | retried, but only once the status is read off the exception rather than its class: the OpenAI SDK raises `openai.InternalServerError`, which is not an `httpx` exception |
| 401 / 403 | not retried; the stage fails and the queue replays the episode |
| empty body, malformed JSON, failed validation | retried |

In front of a key-rotating proxy this splits cleanly: the proxy handles the key
level (step to another key, quarantine the bad one), and this client handles the
pool level. A 429 that reaches here means every key was exhausted, so waiting is
the right answer. A 401 or 403 that reaches here means several keys failed in a
row, which is not a blink and should not be retried blindly.

## Verifying a move

`GRAPHITI_LLM_TRACE_FILE` writes the real request and the real response as JSONL.
The request recorded there is the dict expanded into `create()`, not a
reconstruction, so two runs of the same episode under two gateways can be
compared call for call: same entities, same names in base form, same relation
types, same merges, same contradictions.

Credentials are never written to it. Conversation text and raw model output are,
deliberately — the trace exists to reproduce malformed extraction.
