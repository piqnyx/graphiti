#!/usr/bin/env python3
"""Find out whether this endpoint accepts a reasoning-effort setting, and whether it changes anything.

Extraction spends most of its output tokens on reasoning the task does not need:
merging two sentences produced tens of thousands of output tokens against
deepseek-v4-flash, enough to hit the configured ceiling and truncate the JSON.
Turning that down would cut cost and latency — if the provider takes the setting
at all.

There is no single way to ask. OpenAI takes `reasoning_effort`, its Responses API
takes a `reasoning` object, Anthropic takes `thinking`, and vLLM-hosted models
take `chat_template_kwargs.enable_thinking`. A wrapper may honour one, ignore one
silently, or reject one outright — and "ignored silently" is the case worth
catching, because it looks like success while changing nothing.

So each candidate is sent with an identical prompt and the *output token count*
is compared against a baseline with no setting at all. Accepted and effective
shows up as fewer output tokens; accepted and ignored shows up as the same count.

Costs a handful of small completions. Reads GRAPHITI_LLM_* from the environment
and prints no credential.

    set -a; . ~/memory/secrets/graphiti/graphiti.env; set +a
    python3 tools/probe_reasoning.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 180
# Large enough that a reasoning model is not truncated mid-thought — a truncated
# answer would make the token counts meaningless — and small enough to be cheap.
MAX_TOKENS = 8192
USER_AGENT = os.environ.get('PROBE_USER_AGENT', '').strip() or 'graphiti-reasoning-probe/1'

# A task shaped like the ones extraction actually performs: short input, a
# mechanical judgement, a one-line structured answer. If reasoning effort matters
# anywhere, it matters here.
PROMPT = (
    'Two statements: "Вит живёт в Григолети" and "Вит живёт в селе Григолети рядом с Батуми". '
    'Do they describe the same fact? Answer with JSON only: {"same": true or false}.'
)

CANDIDATES: list[tuple[str, dict]] = [
    ('baseline (nothing sent)', {}),
    ('reasoning_effort=low', {'reasoning_effort': 'low'}),
    ('reasoning_effort=minimal', {'reasoning_effort': 'minimal'}),
    ('reasoning_effort=none', {'reasoning_effort': 'none'}),
    ('reasoning={"effort":"low"}', {'reasoning': {'effort': 'low'}}),
    ('thinking={"type":"disabled"}', {'thinking': {'type': 'disabled'}}),
    ('enable_thinking=false', {'enable_thinking': False}),
    (
        'chat_template_kwargs.enable_thinking=false',
        {'chat_template_kwargs': {'enable_thinking': False}},
    ),
    ('extra_body-style reasoning=low', {'reasoning': 'low'}),
]


def ask(url: str, key: str, model: str, extra: dict) -> dict:
    """Send the probe once and report what came back, without raising."""
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': PROMPT}],
        'max_tokens': MAX_TOKENS,
        **extra,
    }
    request = urllib.request.Request(
        url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', 'replace')
        return {'ok': False, 'why': f'HTTP {error.code}: {detail[:200]}'}
    except Exception as error:  # noqa: BLE001 - a failure is a result here
        return {'ok': False, 'why': f'{type(error).__name__}: {error}'}

    usage = body.get('usage') or {}
    choice = (body.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    # Providers report reasoning tokens in different places, when they report them
    # at all; the total output count is the figure that is always comparable.
    details = usage.get('completion_tokens_details') or {}
    return {
        'ok': True,
        'output': usage.get('completion_tokens'),
        'reasoning': details.get('reasoning_tokens'),
        'answer': ' '.join(str(message.get('content') or '').split())[:80],
        'finish': choice.get('finish_reason'),
    }


def main() -> int:
    url = os.environ.get('GRAPHITI_LLM_API_URL', '').strip()
    key = os.environ.get('GRAPHITI_LLM_API_KEY', '').strip()
    model = os.environ.get('GRAPHITI_LLM_MODEL', '').strip()
    if not (url and key and model):
        print('set GRAPHITI_LLM_API_URL, GRAPHITI_LLM_API_KEY and GRAPHITI_LLM_MODEL', file=sys.stderr)
        return 2

    print(f'model: {model}')
    print('sending the same short extraction-shaped task with each candidate setting\n')

    baseline: int | None = None
    rows: list[tuple[str, str]] = []
    for label, extra in CANDIDATES:
        result = ask(url, key, model, extra)
        if not result['ok']:
            rows.append((label, f'rejected — {result["why"]}'))
            print(f'  {label}: rejected')
            continue

        output = result['output']
        if baseline is None and not extra:
            baseline = output
        change = ''
        if baseline and isinstance(output, int) and baseline > 0:
            delta = round((output - baseline) / baseline * 100)
            change = f' ({delta:+d}% vs baseline)'
        reasoning = f', reasoning {result["reasoning"]}' if result['reasoning'] is not None else ''
        rows.append((label, f'accepted, output {output}{reasoning}{change}, finish={result["finish"]}'))
        print(f'  {label}: output {output}{reasoning}{change}')

    print()
    print('=' * 72)
    for label, outcome in rows:
        print(f'{label:44} {outcome}')
    print('=' * 72)
    print()
    print('How to read this. "rejected" means the provider refuses the field: unusable.')
    print('"accepted" with an output count close to the baseline means it was taken and')
    print('ignored — which is worse than rejection, because it looks like it worked.')
    print('A clearly lower output count is the one setting worth wiring into the client.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
