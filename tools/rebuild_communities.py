#!/usr/bin/env python3
"""Rebuild community summaries for one or more graphs, on a schedule.

Communities group densely connected entities and give each group an LLM-written
summary. That makes this the most expensive operation the server offers, which is
why it lives here and not in any tool an agent can call: a diagnostic must answer
immediately, and this does not. The summaries land in the graph as Community
nodes, so nothing is written to disk and no report can drift from reality — the
status tool reads what this run produced.

Written for cron. Every line is timestamped and the exit code is meaningful, so a
run that fails at three in the morning is visible in the log rather than showing
up a week later as a stale summary.

    python3 tools/rebuild_communities.py main igor

Environment:
    GRAPHITI_MCP_URL   MCP endpoint, default http://127.0.0.1:8000/mcp/
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

DEFAULT_URL = 'http://127.0.0.1:8000/mcp/'
# Community building runs over every entity in a graph and calls an LLM per
# cluster; on a large graph tens of minutes is normal, not a hang.
TIMEOUT_SECONDS = 3600
MAX_REDIRECTS = 5


def log(message: str) -> None:
    print(f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}', flush=True)


def call_tool(url: str, name: str, arguments: dict) -> dict:
    """Invoke one MCP tool over Streamable HTTP and return its parsed result."""
    session_id = None
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
    }

    # urllib refuses to repeat a POST across a 307, which is exactly what the MCP
    # endpoint answers when the path is missing its trailing slash — so the call
    # died on the redirect rather than on anything to do with communities. The
    # redirect is followed by hand, keeping the method, body and session header,
    # which is what 307 means in the first place.
    target = url

    def post(body: dict) -> tuple[str, dict]:
        nonlocal target
        for _ in range(MAX_REDIRECTS):
            request = urllib.request.Request(
                target,
                data=json.dumps(body).encode(),
                headers={**headers, **({'Mcp-Session-Id': session_id} if session_id else {})},
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    return response.read().decode('utf-8', 'replace'), dict(response.headers)
            except urllib.error.HTTPError as error:
                if error.code not in (307, 308) or not error.headers.get('Location'):
                    raise
                target = urllib.parse.urljoin(target, error.headers['Location'])
        raise RuntimeError(f'too many redirects, last was {target}')

    # Handshake first: the server assigns a session id that every later call must
    # carry, and rejects tool calls made without one.
    payload, response_headers = post(
        {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-06-18',
                'capabilities': {},
                'clientInfo': {'name': 'rebuild-communities', 'version': '1'},
            },
        }
    )
    session_id = response_headers.get('Mcp-Session-Id') or response_headers.get('mcp-session-id')
    post({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})

    payload, _ = post(
        {
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        }
    )

    # Streamable HTTP may answer as an SSE stream; the JSON is on the data lines.
    for line in payload.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            line = line[5:].strip()
        if not line.startswith('{'):
            continue
        message = json.loads(line)
        if message.get('id') == 2:
            return message
    raise RuntimeError(f'no result in response: {payload[:300]}')


def main(argv: list[str]) -> int:
    groups = argv[1:]
    if not groups:
        print(__doc__, file=sys.stderr)
        print('give at least one group id (one per agent)', file=sys.stderr)
        return 2

    url = os.environ.get('GRAPHITI_MCP_URL', DEFAULT_URL)
    failures = 0

    # One call per group, and the fork's group-scoped tool rather than the
    # upstream one: upstream builds against the default database, which on this
    # deployment is not where any agent's data lives. One call per group also
    # keeps a single failing graph from taking the others down with it.
    for group in groups:
        log(f'building communities for {group}')
        try:
            message = call_tool(url, 'build_communities_for_group', {'group_id': group})
        except Exception as error:  # noqa: BLE001 - a failed group must not stop the rest
            failures += 1
            log(f'{group}: FAILED {type(error).__name__}: {error}')
            continue

        if 'error' in message:
            failures += 1
            log(f'{group}: FAILED {message["error"]}')
            continue

        content = message.get('result', {}).get('content', [])
        text = ' '.join(part.get('text', '') for part in content if isinstance(part, dict))
        log(f'{group}: {text[:500] or "done"}')

    log(f'finished: {len(groups) - failures} of {len(groups)} group(s) rebuilt')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
