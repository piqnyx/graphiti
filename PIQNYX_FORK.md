# piqnyx Graphiti fork

This repository is a narrowly maintained fork of upstream Graphiti for the piqnyx OpenClaw memory stack.

It is **not** a general rewrite of Graphiti. The upstream project remains the source for Graphiti concepts, APIs, and general installation documentation. This file documents only the behavior and compatibility guarantees added by this fork.

## Baseline

The fork is based on Graphiti / `graphiti-core` **v0.29.3** (`021d3a57d511f21b10adaf7fa923bd5c1fce5e9d`).

The first piqnyx compatibility tag is:

```text
v0.29.3-piqnyx.2
```

That tag includes the isolation and structured-output fixes described below. Newer `main` commits additionally harden the MCP episode queue.

## Why this fork exists

The target deployment uses:

- OpenClaw as the agent runtime
- the Graphiti MCP server over loopback HTTP
- FalkorDB as the graph database
- one physical FalkorDB graph per OpenClaw agent (`main`, `igor`, and future agent IDs)
- OpenAI-compatible LLM and embedding endpoints
- strict cross-agent isolation
- asynchronous episode ingestion

The deployment requires predictable behavior under concurrent multi-agent traffic. The fork therefore carries only fixes needed to make that configuration safe and observable.

## Isolation model

One FalkorDB server may host multiple physical graphs. The OpenClaw integration maps the exact OpenClaw `agentId` to the Graphiti `group_id`, and this fork resolves that group to a request-scoped FalkorDB driver.

The critical invariant is:

```text
OpenClaw agent main -> Graphiti group main -> FalkorDB graph main
OpenClaw agent igor -> Graphiti group igor -> FalkorDB graph igor
```

A request for one group must never mutate a shared driver in a way that can redirect another concurrent request.

## Fork-specific fixes

### Request-scoped FalkorDB driver isolation

Commit:

```text
f364f009ee4e29f7006b196f331e42237a1557dd
```

Concurrent `add_episode` calls for different `group_id` values no longer mutate the shared `self.driver` / `self.clients.driver`. Each request receives a scoped driver/client bundle instead.

This prevents episodes from leaking into another physical FalkorDB graph when concurrent coroutines interleave across LLM/database awaits.

### MCP physical graph scoping and OpenAI-compatible structured output retries

Commit:

```text
066aeb550d8f573d9f4ab1d29ab35dce65a6152a
```

This hardens two production paths:

1. MCP episode reads use the requested FalkorDB graph rather than silently querying the default graph.
2. OpenAI-compatible JSON responses are validated against the requested Pydantic response model inside the retry boundary, and `pydantic.ValidationError` is retryable.

The second change is important for OpenAI-compatible providers that occasionally return syntactically valid JSON with the wrong schema. A wrong-shaped JSON object is retried instead of immediately failing episode ingestion.

### Bounded per-group MCP episode queues

Commits:

```text
dabc65fba0066ee7e1085c222018c19a2843a16d
05b1ef71632f8159f95724f11d23eb877deac680
44ab6f16a324b88cb527e7bb2195bde22d09713a
```

The MCP queue is bounded independently per `group_id`.

Default:

```text
100 pending episodes per group
```

When a group queue is full, enqueue fails immediately with `EpisodeQueueFullError` instead of growing without limit. Worker startup is also claimed before scheduling the worker task, avoiding duplicate workers for the same group.

Queue status exposes per-group pending/running state for diagnostics.

### Edge extraction output budget

`graphiti_core/utils/maintenance/edge_operations.py` is the only extraction path in core that pins its own `max_tokens` instead of using the configured one. Upstream pins 16384.

The target deployment runs a reasoning backend, where `max_tokens` covers reasoning tokens as well as the answer: a probe of the deployed model returned an empty body with `finish_reason=length` once the budget was consumed before any JSON was emitted. An empty body raises `EmptyResponseError`, which is retried four times and then fails the episode, blocking that group's queue.

The fork therefore raises this single constant to 65536 so the edge call has the same headroom as the configured budget for every other call. No other behavior changes; the value is a cap, not an allocation.

### Fork-only read-only tools

Four MCP tools exist only here. Each lives in its own `piqnyx_*` module and is
registered from `mcp_server/main.py` after upstream has created its FastMCP
instance, so no upstream tool definition is modified. All are read-only, take no
LLM, and are scoped to the physical graph selected by `group_id`.

```text
get_saga(saga_name, group_id)              persisted saga state and episode count
get_queue_status(group_id)                 in-memory queue health, including blocked state
get_graph_stats(group_id, top_entities)    size, most connected entities, memory age,
                                           integrity: duplicate episode names, episodes
                                           with no saga or no entities, sagas whose
                                           NEXT_EPISODE chain restarted, facts with no
                                           source episode, isolated entities
get_episodes_by_ref(uuids, names, group_id) specific episodes with their full text
```

`get_graph_stats` runs each query independently and reports failures in
`query_errors` rather than aborting: a diagnostic is needed precisely when the
graph is in an unusual state, so one unsupported query must not cost the whole
report.

`get_episodes_by_ref` exists because upstream can only return "the most recent N"
episodes. Facts record the uuids of the episodes that produced them, and episode
names carry a batch number, so a lookup by uuid finds a fact's source and a
lookup by name reaches its neighbours in the chain. The plugin's memory-context
tool is built entirely on this.

## Tested behavior

The target deployment has exercised:

- sequential `main` / `igor` ingestion
- concurrent ingestion into both graphs
- direct FalkorDB verification of physical separation
- restart persistence
- semantic node search
- memory-fact search
- no cross-agent episode leakage in tested traffic
- recovery from malformed structured LLM output through validation retries

These tests validate the target deployment, not every possible Graphiti backend/provider combination.

## Deployment dependencies used by piqnyx

The Docker deployment pins the Python clients used with FalkorDB:

```text
FalkorDB==1.6.2
redis==8.0.1
```

The target Graphiti MCP endpoint is loopback-only, normally:

```text
http://127.0.0.1:8000/mcp/
```

FalkorDB is also bound to loopback in the target deployment.

Provider credentials and models are deployment configuration and are not stored in this repository.

## Relationship to the OpenClaw plugin

This repository is the **Graphiti server/core fork**. It does not implement OpenClaw lifecycle hooks.

The OpenClaw adapter lives separately in:

```text
piqnyx/graphiti-openclaw-plugin
```

That plugin is responsible for:

- exact `ctx.agentId` validation
- automatic recall through `before_prompt_build`
- automatic capture through `agent_end`
- clean write-path filtering of recalled-memory envelopes
- bounded prompt injection
- per-agent failure cooldowns
- a second foreign-`group_id` filter at the OpenClaw boundary

The two repositories intentionally keep concerns separate: this fork makes Graphiti/FalkorDB safe for the target multi-agent workload; the plugin makes OpenClaw route to it safely.

## Upstream compatibility

Do not casually merge or rebase this fork onto a newer Graphiti release. Before upgrading:

1. check whether each fork patch has landed upstream or changed shape;
2. re-run physical FalkorDB isolation tests with at least two groups;
3. re-run concurrent ingestion stress;
4. verify structured-output behavior with the configured LLM provider;
5. verify MCP episode scoping and queue behavior;
6. re-register the four fork-only tools and confirm the plugin still sees them;
7. only then update the deployment image/tag.

The target stack values a boring, pinned memory backend more than novelty. Memory infrastructure is a poor place for surprise archaeology.
