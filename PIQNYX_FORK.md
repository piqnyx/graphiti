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

`graphiti_core/utils/maintenance/edge_operations.py` is the only extraction path in core that pins its own `max_tokens` instead of using the configured one. Upstream pins 16384. The fork passes nothing, so `generate_response` falls back to the configured budget and the deployment governs this call the way it governs every other.

The pin mattered because the target deployment runs a reasoning backend, where `max_tokens` covers reasoning tokens as well as the answer: a probe returned an empty body with `finish_reason=length` once the budget was consumed before any JSON was emitted, and an empty body raises `EmptyResponseError`, which is retried four times and then fails the episode, blocking that group's queue. A later batch stopped at exactly 65536 output tokens — a truncated reply rather than a long one, whose JSON never closes.

The fork answered both by raising the constant, first to 65536 and then to 131072. The second raise changed nothing: `OpenAIGenericClient` clamps an explicit request to the deployment ceiling — `min(requested, self.max_tokens)`, covered by `test_explicit_prompt_budget_cannot_exceed_deployment_ceiling` — so a pin above that ceiling asks for nothing extra and only misdescribes the call. Passing no budget at all leaves one knob that means what it says: set the deployment's budget and this call follows it.

### Reasoning ladder on an output limit

On 2026-11-12 an episode of `main` failed three times running on gemini-3.1-flash-lite: 62 912 tokens of reasoning and 2 609 of answer against 65 536, finish `length`. The durable queue sends a failed episode again every 300 s at the same reasoning level, so the graph stood behind it.

With `GRAPHITI_REASONING_FALLBACK=on`, `OpenAIGenericClient` asks a call that raised `OutputLimitError` again at once, one level lower: `high`, `medium`, `low`, `none`. Only that call steps down; the next starts again at the level `GRAPHITI_REASONING_EFFORT` / `..._BY_PROMPT` give it. A stage already at `minimal` or `none` does not step, an unset or unknown level steps to `medium`, and the last limit goes to the queue as before. Every other error goes up unchanged. The repeat is of the inner call, so the schema and language instruction are not written into the messages twice. Unset, the switch leaves every call as it was. Worst case per call: four full-budget requests. Covered by `tests/llm_client/test_openai_generic_reasoning_ladder.py`; details in `docs/piqnyx-moving-between-gateways.md`.

### Entity deduplication with each entity's own candidates

Upstream searches the fifteen nearest names for every extracted entity apart, then hands the model one shared list of all of them (`EXISTING ENTITIES`), numbered from zero just as the entities are. On long lists — up to 33 entities against 343 candidates in the deployment's trace — the model lost track of whose candidate was whose and wrote an entity's own number as the candidate's, and the code took it: "Пентагон" merged into "Чернобыльская АЭС", "Почта России" into "SpaceX". The recorded trace held 60 such merges in 25 of 529 calls since August, every one wrong; a replay of the 20 largest calls gave 30 more at `high` and about half of all merges on gemini-3.1-flash-lite at either level.

The fork's `dedupe_nodes.nodes` shows each entity with its own candidates under it (`candidates`), numbered after the entities — entity count plus place in the shared list — so a candidate number never coincides with an entity number, and the answer repeats the chosen candidate's name (`duplicate_candidate_name`). `_resolve_with_llm` takes a merge only with a candidate from that entity's own list whose name matches — case, spacing, ё/е and the invisible characters the client strips from the prompt aside. Entities whose answer fails that, or that the answer leaves out, are asked again, alone (`dedupe_nodes.nodes_again`), up to `GRAPHITI_DEDUPE_REASKS` times — once when unset, 10 on the deployment — each time only those still failing; what fails every question keeps no duplicate, with a warning naming the entity and the answer. Asking again is insurance: if the model's side of it fails — a refusal, a limit, a network fault, an answer that does not read — those entities stay new and the episode goes on with what was settled before; building the question is our own code and a fault there surfaces. A merge the model makes lands on the shared list's copy of the node, so two entities the model merges into one node share one copy, as before (merges by exact or fuzzy name land on the entity's own search copy, also as before). A duplicate entity costs a later merge; a wrong merge hangs one thing's facts on another.

Measured on 2026-11-12 with this prompt word for word, the same 20 calls 3× at `high` (58 answers: 41 from gemini-3.5-flash-lite, 11 from 3.1-flash-lite, 6 from senior models): the check caught nothing, no index-copy merges, 8 real duplicates the shared list missed and 4 it found missed; requests 21% larger, the same in number. Covered by `tests/utils/maintenance/test_dedupe_own_candidates.py`.

### FalkorDB fact search without a per-hit graph scan

`edge_fulltext_search` in `graphiti_core/search/search_utils.py` is the path FalkorDB runs (`driver.search_interface` is never assigned). Upstream re-matches every fulltext hit by uuid, `MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)`, which FalkorDB plans as a scan of every Entity node per hit, so a broad query costs hits × entities. On the deployment's 1373-fact graph, digest queries took 20–70 s and hit the 30 s query timeout.

For FalkorDB only, the fork takes the hit's ends with `startNode`/`endNode` — the rewrite upstream PR #1500 proposes — and keeps the old pattern's filter as `n:Entity AND m:Entity`. Compared on the live graph against the original function: identical results (every field, same order) for 87 queries, including digest post gists, fact texts, a date filter and the production candidate pool of 80, in under a second instead of 20–70 s.

The two forms are equal while no two `RELATES_TO` edges share a uuid: the old re-MATCH returned every edge with the hit's uuid, so a duplicate would have been repeated and taken extra `LIMIT` slots. The live `main` graph had none (1373 facts, 1373 distinct uuids). Unused copies of the old per-hit re-MATCH remain in `driver/falkordb/operations/search_ops.py` (fulltext and BFS), which nothing calls.

### FalkorDB indices built once per graph

Upstream schedules a full index build in every `FalkorDriver.__init__`, and every `clone()` creates a new driver, so each request re-sent all thirteen index-creation commands (nine range, four fulltext). They are writes: behind a long read they waited, and every query arriving after them waited too — a 0.1 s read took 27 s.

The fork builds once per connection and graph per process, and marks a graph only after its build succeeded, so a failed or cancelled build is retried by the next driver. Drivers created while a build runs do not start another. A build that never returns keeps its graph marked as under way, so no other driver builds it until a restart; only a graph that has no indices yet, such as a new agent's, is affected. An explicit `build_indices_and_constraints()` call still always runs. A graph deleted while the server runs gets no indices until the server restarts; the deployment's reset script restarts it.

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
