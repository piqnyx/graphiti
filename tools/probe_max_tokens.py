#!/usr/bin/env python3
"""Find the largest max_tokens the configured LLM endpoint will accept.

`llm.max_tokens` is the ceiling on one response, sent with every request. Too low
and long extractions are truncated, which is how edge extraction once returned
empty and blocked the queue; too high and the provider rejects the request
outright, which breaks not one heavy call but every call. The accepted maximum is
a property of the provider and model, is not reliably published anywhere, and
changes when the model changes — so it is measured rather than guessed.

The probe sends the smallest possible completion at each candidate ceiling and
watches only whether the request is accepted. Nothing is generated: a request
rejected for its max_tokens is rejected before any tokens are produced, so this
costs approximately nothing.

Reads GRAPHITI_LLM_API_URL, GRAPHITI_LLM_API_KEY and GRAPHITI_LLM_MODEL from the
environment. No credential is printed, including in error messages.

    set -a; . ~/memory/secrets/graphiti/graphiti.env; set +a
    python3 tools/probe_max_tokens.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# The search runs over powers of two rather than a plain binary search between 0
# and some invented upper bound: it finds the bracket in a handful of requests
# without assuming anything about the scale, which is the part that differs
# wildly between providers.
LOWEST = 1024
CEILING = 1 << 21  # 2097152; far above any current model, so it bounds the search
TIMEOUT_SECONDS = 30
# urllib announces itself as Python-urllib, which Cloudflare's browser integrity
# check rejects outright with 403/1010 — before the request reaches the model, so
# the answer would be about the CDN rather than the ceiling. The real client does
# not hit this because httpx sends its own agent. Override with PROBE_USER_AGENT
# if a provider wants something else.
USER_AGENT = os.environ.get('PROBE_USER_AGENT', '').strip() or 'graphiti-max-tokens-probe/1 (httpx-compatible)'


def accepts(url: str, key: str, model: str, max_tokens: int) -> tuple[bool, str]:
    """True when the endpoint accepts this ceiling, plus the reason when it does not."""
    payload = json.dumps(
        {
            'model': model,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip('/') + '/chat/completions',
        data=payload,
        headers={
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            response.read()
        return True, ''
    except urllib.error.HTTPError as error:
        body = error.read().decode('utf-8', 'replace')
        # Trimmed and stripped of the request: provider errors sometimes echo
        # headers back, and this output is meant to be safe to paste.
        return False, f'HTTP {error.code}: {body[:300]}'
    except Exception as error:  # noqa: BLE001 - network failure is a result, not a crash
        return False, f'{type(error).__name__}: {error}'


def main() -> int:
    url = os.environ.get('GRAPHITI_LLM_API_URL', '').strip()
    key = os.environ.get('GRAPHITI_LLM_API_KEY', '').strip()
    model = os.environ.get('GRAPHITI_LLM_MODEL', '').strip()
    missing = [
        name
        for name, value in (
            ('GRAPHITI_LLM_API_URL', url),
            ('GRAPHITI_LLM_API_KEY', key),
            ('GRAPHITI_LLM_MODEL', model),
        )
        if not value
    ]
    if missing:
        print(f'missing environment: {", ".join(missing)}', file=sys.stderr)
        return 2

    print(f'model: {model}')
    ok, reason = accepts(url, key, model, LOWEST)
    if not ok:
        print(f'the endpoint rejected even max_tokens={LOWEST}, so nothing here is about the ceiling:')
        print(f'  {reason}')
        return 1

    good = LOWEST
    bad = 0
    candidate = LOWEST * 2
    while candidate <= CEILING:
        ok, reason = accepts(url, key, model, candidate)
        print(f'  max_tokens={candidate}: {"accepted" if ok else "rejected"}')
        if not ok:
            bad = candidate
            break
        good = candidate
        candidate *= 2

    if bad == 0:
        print(f'accepted every value up to {good}; the endpoint appears not to cap max_tokens')
        return 0

    # Narrow the bracket. Stopping at 1024 apart is deliberate: the exact token is
    # of no practical use, and every step is a request.
    while bad - good > 1024:
        middle = (good + bad) // 2
        ok, _ = accepts(url, key, model, middle)
        print(f'  max_tokens={middle}: {"accepted" if ok else "rejected"}')
        if ok:
            good = middle
        else:
            bad = middle

    print()
    print(f'largest accepted max_tokens: {good}')
    print(f'first rejected: {bad}')
    print()

    # Providers typically check prompt + max_tokens against the context window
    # rather than capping output on its own. This probe sends a two-token prompt,
    # so what it finds is very nearly the whole window — and configuring that
    # figure would make every real request fail, since a real prompt occupies part
    # of the same budget. The recommendation therefore keeps most of the window
    # free for input, which is where extraction actually needs it.
    recommended = 1 << (good // 3).bit_length() - 1
    print('The figure above is close to the model\'s whole context window, not an output-only cap:')
    print('a real request must fit prompt + max_tokens inside it, so configuring that value would')
    print('fail on every extraction large enough to matter. Leave the input room it needs:')
    print()
    print('  llm:')
    print(f'    max_tokens: {recommended}')
    print()
    print(f'That leaves roughly {good - recommended} tokens for the prompt, and is far more output')
    print('than extraction ever produces — the ceiling only has to stop truncation, not be reached.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
