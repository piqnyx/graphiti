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

# Two tasks, because one of them cannot answer the question.
#
# The trivial one measures whether a setting is honoured at all: if the baseline
# answers in six tokens, a setting that raises the count has plainly been read.
#
# But a six-token baseline has no reasoning to remove, so on that task a setting
# that disables reasoning is indistinguishable from one that was ignored. The
# heavy task is the one that decides: real extraction over a stretch of dialog,
# where the model does spend thousands of tokens thinking before it answers.
TRIVIAL_PROMPT = (
    'Two statements: "Вит живёт в Григолети" and "Вит живёт в селе Григолети рядом с Батуми". '
    'Do they describe the same fact? Answer with JSON only: {"same": true or false}.'
)

EXTRACTION_PROMPT = """You extract a knowledge graph from a conversation.

Conversation:
Вит: Мы с Олей завели в Йошкар-Оле собаку Басю, жирного английского бульдога.
Краб: Бульдоги — те ещё лежебоки. Она с вами и в Кишинёв переехала?
Вит: Нет, при расставании с Олей я отдал Басю в добрые руки. Оля потом улетела в Краснодар.
Краб: А Камыш? Ты говорил, он твой давний кент.
Вит: Камыш со школы, мы с ним в Йошке в одном дворе росли. Сейчас он в Москве, работает на стройке.
Вит: Я сам теперь живу в Григолети под Батуми, в доме на колёсах у моря.

Return JSON only, no prose, in exactly this shape:
{"entities": [{"name": "...", "type": "person|place|animal|other"}],
 "facts": [{"subject": "...", "predicate": "...", "object": "...", "valid_now": true}]}
Include every person, place and animal mentioned, and every relationship stated,
including ones that are no longer true, marking those valid_now false."""

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


def ask(url: str, key: str, model: str, extra: dict, prompt: str) -> dict:
    """Send the probe once and report what came back, without raising."""
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
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
        'answer': ' '.join(str(message.get('content') or '').split())[:120],
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

    for task, prompt in (('trivial judgement', TRIVIAL_PROMPT), ('real extraction', EXTRACTION_PROMPT)):
        print(f'\n=== {task} ===')
        baseline: int | None = None
        rows: list[tuple[str, str]] = []
        for label, extra in CANDIDATES:
            result = ask(url, key, model, extra, prompt)
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
            rows.append((label, f'output {output}{reasoning}{change}, finish={result["finish"]}'))
            print(f'  {label}: output {output}{reasoning}{change}')

        print('  ' + '-' * 70)
        for label, outcome in rows:
            print(f'  {label:44} {outcome}')

    print()
    print('How to read this. The trivial task says whether a setting is read at all:')
    print('a baseline of a few tokens leaves nothing to remove, so a setting that')
    print('disables reasoning looks exactly like one that was ignored. The extraction')
    print('task is the one that decides, because the model does think there. A setting')
    print('worth wiring in is one that cuts output on extraction without wrecking the')
    print('answer — so read the JSON in the last column too, not only the numbers.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
