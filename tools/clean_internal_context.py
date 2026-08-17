#!/usr/bin/env python3
"""Remove OpenClaw's runtime context block from episodes already stored in a graph.

The gateway prepends a block to the user's message carrying the chat id, the
sender's identity, session ids and the recent traffic of other sessions. The
plugin strips it at capture time now, but episodes recorded before that fix still
hold it, and since episodes became a searchable result type in their own right,
that text is no longer inert: a query can match the junk and lead the agent to
read it.

Only the `content` property is rewritten, one episode at a time. Nothing else is
touched — not the uuid, name or batch number, not NEXT_EPISODE, not MENTIONS, not
a single fact or its provenance. Integrity diagnostics must report exactly the
same before and after.

Dry by default: it prints what would change and writes nothing until --apply.

    docker exec -e FALKORDB_PASSWORD graphiti-server \\
        python tools/clean_internal_context.py main
    docker exec -e FALKORDB_PASSWORD graphiti-server \\
        python tools/clean_internal_context.py main --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# The same rule the plugin applies at capture time: everything up to and
# including the closing marker, preamble and all; an unclosed block was
# truncated mid-context and goes to the end of the text.
CLOSED_BLOCK_RE = re.compile(r'^[\s\S]*?<<<END_OPENCLAW_INTERNAL_CONTEXT>>>')
UNCLOSED_BLOCK_RE = re.compile(r'<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>[\s\S]*$')
WHOLE_TEXT_DUPLICATE_RE = re.compile(r'^([\s\S]+)\n\1$')

MARKERS = ('BEGIN_OPENCLAW_INTERNAL_CONTEXT', 'openclaw:ctx', '"chat_id"')


def clean_text(text: str) -> str:
    cleaned = UNCLOSED_BLOCK_RE.sub('', CLOSED_BLOCK_RE.sub('', text))
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return WHOLE_TEXT_DUPLICATE_RE.sub(r'\1', cleaned)


def clean_episode_body(raw: str) -> str | None:
    """Return the rewritten body, or None when nothing would change.

    The body is the JSON the plugin writes: participants plus a list of
    messages. Only message texts are rewritten, so the structure the reader
    depends on survives untouched. A body that is not that JSON is left alone —
    guessing at an unknown format is how data gets destroyed.
    """
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return None

    changed = False
    messages = []
    for message in body['messages']:
        if not isinstance(message, dict) or not isinstance(message.get('text'), str):
            messages.append(message)
            continue
        cleaned = clean_text(message['text'])
        if cleaned != message['text']:
            changed = True
        # A message emptied by the cleaning was nothing but the block; dropping
        # it is more honest than storing a speaker who said nothing.
        if cleaned:
            messages.append({**message, 'text': cleaned})
        else:
            changed = True

    if not changed:
        return None
    return json.dumps({**body, 'messages': messages}, ensure_ascii=False)


def preview(text: str, width: int = 160) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= width else f'{flat[:width]}…'


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('graphs', nargs='+', help='Graph names (one per agent), e.g. main igor')
    parser.add_argument('--apply', action='store_true', help='Write the changes. Without it, nothing is modified.')
    parser.add_argument('--host', default=os.environ.get('FALKORDB_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('FALKORDB_PORT', '6379')))
    args = parser.parse_args(argv[1:])

    password = os.environ.get('FALKORDB_PASSWORD')
    if not password:
        print('FALKORDB_PASSWORD is not set', file=sys.stderr)
        return 2

    try:
        from falkordb import FalkorDB
    except ImportError:
        print('the falkordb package is missing: run this inside the graphiti-server container', file=sys.stderr)
        return 2

    db = FalkorDB(host=args.host, port=args.port, password=password)
    total_found = 0
    total_changed = 0

    for name in args.graphs:
        graph = db.select_graph(name)
        # Fetched by explicit markers rather than by scanning every episode: a
        # large graph should not be pulled through the network to find a handful.
        found = graph.ro_query(
            """
            MATCH (e:Episodic)
            WHERE e.content CONTAINS $marker
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content
            ORDER BY e.created_at
            """,
            {'marker': 'OPENCLAW_INTERNAL_CONTEXT'},
        ).result_set

        print(f'=== {name}: {len(found)} episode(s) carrying the block')
        for uuid, episode_name, content in found:
            total_found += 1
            rewritten = clean_episode_body(content)
            if rewritten is None:
                print(f'  {episode_name}: body is not the expected JSON, or nothing to remove — left alone')
                continue

            total_changed += 1
            print(f'  {episode_name}: {len(content)} -> {len(rewritten)} chars')
            print(f'    after: {preview(rewritten)}')
            if args.apply:
                graph.query(
                    'MATCH (e:Episodic {uuid: $uuid}) SET e.content = $content',
                    {'uuid': uuid, 'content': rewritten},
                )

    print()
    if not args.apply:
        print(f'{total_changed} of {total_found} episode(s) would be rewritten. Nothing was written: pass --apply.')
    else:
        print(f'{total_changed} of {total_found} episode(s) rewritten.')
    print('Only the content property is touched; run graphiti_status to confirm the graph reports the same as before.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
