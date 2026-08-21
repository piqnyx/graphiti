# piqnyx graph health checklist

Specification for a rewritten `graphiti_status`, and the manual procedure until it
exists.

Two kinds of entry appear below. Some were derived from a defect actually seen on
the live graph — those carry a measurement and the date it was taken. The rest are
invariants that have not been violated yet and are listed so that the first
violation is noticed on the day it happens rather than a month later. Both belong
here: a health report written only from current pain goes stale the moment the pain
is fixed.

The tool may be slow. It is a diagnostic, not a hot path.

Each section is marked with what can decide it:

- **code** — a query and arithmetic decide it; no judgement, no model;
- **model** — the verdict is a judgement about meaning and needs an LLM pass;
- **eyes** — too noisy to automate; print it and let a person look.

---

## 1. Structure — saga and episodes · code

Absolute invariants. A failure is a bug in the capture pipeline, not a matter of degree.

| check | healthy |
|---|---|
| chain heads: episodes with no incoming `NEXT_EPISODE` | exactly one per saga |
| episodes with no saga | none |
| branching: more than one incoming or outgoing `NEXT_EPISODE` | none |
| `first_episode_uuid` / `last_episode_uuid` against the real ends | match |
| chain length | equals the episode count |
| chain order | equals the order by `created_at` |
| one session key | one saga |
| an episode belonging to two sagas | none |
| batch numbers within a saga | start at 1, increase by one, no gaps |
| `previous_episode_uuids` of an episode | is its actual chain predecessor |
| `reference_time` along a chain | never decreases |
| episode uuid | matches the deterministic derivation from saga and batch number |

Measured 2026-08-21: 2 sagas, 9 episodes, 7 `NEXT_EPISODE` — correct for two chains.
This held even after the gateway was killed mid-flush: the replay adopted the
retained uuid instead of writing a second episode.

The last four rows have never failed. They are listed because each one silently
breaks chronology rather than announcing itself, and chronology is what the whole
store is for.

## 1a. Episode bodies · code

The body is what extraction reads. Anything that leaked into it was read as if the
user had said it, and nothing downstream can tell the difference.

| check | healthy |
|---|---|
| body parses as JSON with `participants` and `messages` | always |
| speaker names in `participants` | match the configured actors for that agent |
| `<thinking>`, `<toolCall>` or any tool markup in the body | none |
| `<graphiti-context>`, `<openviking-context>`, `<relevant-memories>` in the body | none — injected memory being recaptured as conversation |
| the gateway's internal-context block | none |
| empty message text | none |
| `source_description` | the expected one for every episode |
| the same message text twice in one body | none — see the rewind signature in section 8 |

The injected-memory row is the one that compounds: memory recaptured as conversation
becomes a fact about itself, and the next recall injects that. Nothing about the
resulting fact looks synthetic.

## 2. Provenance and referential integrity · code

Nothing here has been violated yet. All of it would be invisible until a query
returned nonsense.

| check | healthy |
|---|---|
| edge `episodes` uuids that no episode has | none |
| `MENTIONS` pointing at a missing episode or entity | none |
| edge whose `source_node_uuid` or `target_node_uuid` has no node | none |
| entity reachable from no episode at all | none |
| `group_id` agreement across saga, episode, entity, edge | one value, equal to the graph name |
| episodes present in the graph but absent from the plugin's durable journal | none, once the queue is drained |

The last row is the only cross-system check in this document, and it is the one that
catches a whole class of silent loss: an episode the plugin believes it delivered and
the graph never received, or the reverse.

## 3. Retrieval readiness · code

A fact that cannot be retrieved is the same as a fact that was never stored, and
nothing about it looks wrong in the graph.

| check | healthy | measured 2026-08-21 |
|---|---|---|
| edges with no `fact_embedding` | none — invisible to recall | 0 |
| entities with no name embedding | none | not yet measured |
| embedding dimension across all vectors | one value, equal to the configured one | 1024 |
| fulltext index present and populated | yes | see note |
| reranker endpoint reachable | yes | — |
| embedder endpoint reachable | yes | — |

Mixed embedding dimensions are the dangerous case: a graph re-embedded with a
different model keeps working for old facts and silently ranks new ones nowhere near
the query. Check the set of dimensions, not just the configured value.

The fulltext branch is currently dead in this fork's backend and this is understood:
FalkorDB's query builder trips its own length guard on real recall queries, and there
is no Russian stemmer. The check should report the index as unusable rather than
report zero hits as health.

## 4. Entities · code, with one model pass

| check | healthy | measured 2026-08-21 |
|---|---|---|
| entities with no `MENTIONS` | zero, always | 0 of 174 |
| entities with no `RELATES_TO` | report the share | 40 of 174 (23%) |
| noise: exactly one mention and no edges | report the share | 36 of 174 (21%) |
| namesakes differing only by case | none | none |
| namesakes differing only by `ё`/`е` or by whitespace | none | not yet measured |
| names shorter than 4 characters | eyes | 12, all legitimate (`C++`, `PHP`, `git`) |
| names that are sentences, not names | none | not yet measured |
| names containing markup, emoji or a URL | none | not yet measured |
| entity harvest per episode | scales with body size | see below |
| hub degree | report the top few | not yet measured |
| components disconnected from the main graph | report the count and size | not yet measured |

**Unlinked entities are enumeration residue, not corruption.** They are things named
in a list or in passing about which no statement was made — dishes, cities, games,
file names, tool names. Observed: `плов`, `шашлык`, `Metro 2033`, `Арракис`, `Минск`,
`AGENTS.md`, `mcp__firecrawl_*`. They keep their provenance and may gain facts later,
but they dilute entity search. Report the share; do not call it an error.

A telling pair from the same run: `оджахури` and `аджахури` — one dish, the second
spelled wrong, both promoted to nodes. Whether those are one entity is a judgement,
so it belongs to the model pass, not to a query.

**Harvest must be judged against body size, not message count.** An episode is a
batch of up to 20 messages, so a spike means a long body, not a shredded enumeration:

```
24593 chars → 59 entities        6384 chars → 7 entities
12188 chars → 48                 4633 chars → 17
 8975 chars → 38
```

Flag an episode only when its entities-per-kilobyte departs from the run's own median.

**Hubs and components are listed before they hurt.** One person is the subject of
most facts, so a hub is expected and its degree is a scale signal, not a defect. A
second component is different: it means a set of facts that no path connects to the
rest, which is what happens when an entity is created under a variant name and never
merged. Neither has been measured yet.

### Summaries

Entities and sagas both carry a summary that no query validates.

| check | healthy | decided by |
|---|---|---|
| entities with an empty summary | report the share | code |
| saga with no summary, or `last_summarized_at` older than its last episode | report | code |
| a summary asserting something its own facts no longer support | none | model |

A summary is written once and then drifts: the facts under it get invalidated and it
keeps saying what it said. It is also what a person reads first, so it drifts in the
most visible place.

## 5. Facts · code, with one model pass

| check | healthy | measured 2026-08-21 |
|---|---|---|
| self-loops | none | none |
| fact text names neither endpoint | none | 6 of 191 |
| duplicate edges (same pair, same type) | none | 2 |
| distinct relation types per edge | low | 108 types / 191 edges (57%) |
| single `group_id` | yes | yes |
| facts with no `valid_at` | report the share | 29 of 191 (15%) |
| invalidated facts | only where a real contradiction exists | 3, all genuine |
| facts with no source episode | none | none |
| language of fact text | as configured | all Russian |
| relation type names | ASCII, `SCREAMING_SNAKE_CASE` | all ASCII |
| fact text length distribution | report the tail | not yet measured |
| facts that are questions, not statements | none | not yet measured |
| facts carrying markup, emoji or tool noise | none | not yet measured |
| facts attributed to the assistant's guess rather than the user's statement | none | seen once |

**A fact with no `valid_at` is not immortal.** Tested 2026-08-21 on the live graph:
an edge with an empty `valid_at` received `invalid_at` from a later contradicting
episode. The temporal guard does not block it. Report the share as a retrieval and
provenance signal, not as a supersession risk.

**The assistant's own mistakes become facts.** Measured 2026-08-21: recall returned
nothing for a direct question, the assistant answered from stale context and was
wrong, and its answer was captured and extracted into an edge whose text openly says
"по словам Эвы". It was invalidated in the same pass, so no harm — but the class is
real and needs its own count. Deciding whether a fact came from the user or from the
assistant's speculation is a judgement, so it belongs to the model pass.

### Fact text naming neither endpoint has three distinct causes

Separate them in the report — they need different fixes.

1. **Spelling variance between node name and fact text.** `Кишинёв` as a node,
   "Кишинев" in the fact. A `ё`/`е` difference is enough. Normalize before comparing,
   then report what survives.
2. **Script drift between node name and fact text.** `Graphiti` as a node, "Графити"
   in the fact; `OpenViking` → "Викинг". The fact text is what recall embeds and what
   the cross-encoder scores, so a question naming `Graphiti` scores low against its
   own fact. A real defect.
3. **An epithet in the node name.** `Дед Антон` as a node, "Антон" in the fact.
   Harmless; report as a hint.

### The subject can drift onto the wrong endpoints

Measured 2026-08-21. "Вит переехал из Григолети в Кобулети" became an edge between
the two villages, with the person absent from both ends:

```
Григолети MOVED_FROM_TO Кобулети   «Вит переехал из Григолети в Кобулети»
```

Two consequences, and the second is the one that matters. A traversal from the person
never reaches the new home. And supersession only ever compares edges that share
endpoints, so a statement landing on a different pair each time can never contradict
the previous one — the graph then accumulates rival facts that no judge will ever
resolve. Report facts whose text names an entity that is not one of the endpoints.

### Type proliferation is not diversity

Half the types occurring once is a symptom of the model packing the whole fact into
the type name: `CONVERTING_CASTLE_TO_STUDENT_HOUSING`, `HAD_MEDICAL_CHECK_FOR`,
`LOVES_UNIVERSE`. The cost is concrete, not cosmetic — two edges between the same
pair with different type names are never seen as duplicates by dedupe:

```
USER.md CONTAINS_INFO_ABOUT Вит
USER.md IS_PASPORT_FOR      Вит      (same relation, and a typo in the type name)
```

Report: type count, edge count, share of singleton types, and the pairs carrying more
than one type. Whether two type names mean the same thing is a judgement — model pass.
Trend: 95/195 on 2026-08-20, 108/191 on 2026-08-21 — getting worse.

### Refinement is not contradiction

An edge restating an earlier one more precisely does not invalidate it, and dedupe
does not merge it, so both survive:

```
Эва LIVES_IN Дюссельдорф   «Эва живёт в Дюссельдорфе»                     00:30
Эва LIVES_IN Дюссельдорф   «...в аскетичной квартирке на Рейне»           01:20
```

Report same-pair-same-type edges with their `valid_at`, so refinement pairs are
visible rather than being counted as plain duplicates.

## 6. Time · code

None of this has been violated yet. All of it corrupts chronology silently.

| check | healthy |
|---|---|
| `valid_at` in the future | none |
| `invalid_at` earlier than `valid_at` | none |
| more than one live edge on the same pair and type | none, or reported as a refinement pair |
| an edge invalidated with nothing that replaced it | report — an erased fact with no successor |
| `created_at` earlier than the `reference_time` of its own episode | none |

## 7. Recall health · code, from the plugin log

The graph can be perfect and recall still useless. This section reads the log, not
the graph, and it is the section that would have caught the day recall ran with every
improvement switched off.

| check | healthy | measured 2026-08-21 |
|---|---|---|
| turns with zero facts returned | low, and never on a question the graph can answer | 16 of 80 |
| turns hitting `recallLimit` exactly | not the majority | 58 of 80 (72%) before floors, far lower after |
| distinct facts ever injected, against the graph total | high | 78 of 191 (41%) |
| the most-injected fact, as a share of turns | low | 33 of 80 turns |
| facts filtered as superseded | non-zero once anything has been invalidated | works |
| facts filtered by the floor | non-zero once a floor is set | 0 — the floors were unset |
| recall latency | report median and worst | median 952 ms, worst 7.3 s |
| score distribution against the floor | report, once scores are logged | not yet logged |

Two effects to name explicitly, because each looks like the other in a summary:

- **the ceiling** — every turn returning exactly `recallLimit` means nothing is being
  filtered and the number is doing the choosing;
- **crowding** — a handful of facts appearing in most turns regardless of topic. On
  2026-08-21 the top fact was "Вит изучал JavaScript", injected in 33 of 80 turns,
  including a question about deleting a cloned folder.

A false negative is worth more attention than a false positive here, and it needs a
question whose answer is known to be in the graph. Measured: the floor returned zero
facts for a direct question about the user's home while the answer was present.

## 8. Capture health · code, from the plugin log

| check | healthy | measured 2026-08-21 |
|---|---|---|
| `queueSequence` gaps | none | none |
| payloads without a matching commit | none, unless a replay adopted the identity and said so | 10 payloads, 9 episodes, 1 explained replay |
| rows dropped by sanitisation | expected; report the share | 43 of 199 |
| empty messages reaching an episode | none | 0 |
| flush reasons | report the split between limit and timeout | both seen |
| batch size distribution | report | 20, 20, 20, 20, 20, 3, 20, 20, 7 |
| duplicate user messages with different `event_id` | none | 2 |
| lease conflicts and identity adoptions | reported, never silent | both seen and logged |
| cursor advanced without the batch being made durable | none | none |

**The duplicate has a signature.** Identical text, different `event_id`, sequence
numbers a few apart, and a `{"type":"leaf", targetId}` event between them. That is a
rewind: the gateway repoints the session key at a fresh session whose opening rows
are copies carrying their original ids — which the cursor already recognises — and
the re-sent message is genuinely new. Six `leaf` events existed across the whole
store, all at remembered rewinds, so the marker is trustworthy.

## 9. Cost · code, from the LLM trace

Requires `GRAPHITI_LLM_TRACE_FILE`. Skip the section, saying so, when the trace is off.

- calls per episode, broken down by prompt name;
- tokens in and out per episode, and per thousand captured messages;
- share of merges per dedupe call, and whether it tracks the candidate-list length;
- calls where the number of answers differs from the number of entities;
- runs of `candidate index == entity index` — direct evidence of the model degenerating
  into echoing positions rather than judging;
- wall-clock per episode, so a slow model is distinguishable from a slow graph.

## 10. Growth · code

Nothing here is a defect. It is the section that says whether the thing is still
behaving the way it did last week.

- nodes, edges and episodes against the previous run;
- entities and edges per kilobyte of episode body, as a trend;
- share of unlinked entities, as a trend;
- share of singleton relation types, as a trend;
- database size on disk;
- background queue depth at the moment of the run.

## 11. Deliberately noisy — eyes only

Name containment (`Нодар` inside `Краснодар`, `Лера` inside `Валера`, `java` inside
`JavaScript`) produces almost only false positives. Print it, never count it as a
finding.

---

## What the model pass decides

Everything above is arithmetic except these, and each one is a judgement about
meaning that a query cannot make:

- whether two entities with different names are the same thing;
- whether two relation type names mean the same relation;
- whether an invalidation was justified, or erased something still true;
- whether a fact states what the user said or what the assistant guessed;
- whether a fact is a refinement of another or a genuine second fact;
- whether a fact is worth keeping at all, or is a list item that became a node.

Give the model the candidates the queries produced, never the whole graph, and have
it return a verdict per candidate with a reason. That keeps the pass cheap and its
output auditable against the query that proposed the candidate.

## Output

Group by section, and lead each with a verdict rather than a number:

- ✅ nothing to look at
- ⚠️ a share worth watching, with the trend against the previous run
- ❌ an absolute invariant is broken

Every proportion carries its denominator. Every trend carries the date it is compared
against. A check that could not run — trace disabled, endpoint unreachable, query
unsupported by the backend — says so instead of silently reporting zero. A check that
has never been measured says that too, rather than printing a first value as if it
were a trend.

Store each run's counters so the next run can print the delta.

FalkorDB does not support `=~`. Checks needing a regular expression run their filter
outside the query.

## The scripts this replaces

Written during the manual rounds and kept until `graphiti_status` covers them:

| script | what it does |
|---|---|
| `graph_audit.sh` | sections 1, 4, 5 and part of 6 as raw queries |
| `trace_check.py` | entity numbering in dedupe answers, against the trace |
| `trace_edges.py` | what the model called a duplicate and a contradiction |
| `trace_dump.py` | one dedupe call in full, lists and raw answer |
| `dedupe_probe.py` | numbering alignment in batched dedupe, with and without thinking |
| `judge_probe.py` | the contradiction judge on five labelled cases |
| `recall_grid.py` | five ways of asking memory the same thing |
| `recall_probe.py` | recall on a long query against the last message alone |
| `four_questions.py` | four questions, three recall modes |
| `dump_messages.py` | the user's own lines, for replaying a corpus |
