"""The reasoning ladder (fork, 12.11): a call that runs into its output limit is asked again
at once, one level of reasoning lower, down to the lowest.

Why: an episode of `main` failed three times running on gemini-3.1-flash-lite -- 62 912 tokens
of reasoning and 2 609 of answer against a limit of 65 536, finish "length" -- and the durable
queue sent it again every 300 s at the same level, so the graph stood behind it. The level a call
starts at stays the deployment's (`GRAPHITI_REASONING_EFFORT`, `..._BY_PROMPT`); the ladder only
steps down for the call that hit the limit, and only when `GRAPHITI_REASONING_FALLBACK` is on.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import OutputLimitError, RefusalError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    foo: str


def _choice(content, finish_reason):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(completion_tokens=2609, prompt_tokens=7643, total_tokens=73164),
    )


LIMIT = _choice('{"foo": "half', 'length')
WHOLE = _choice('{"foo": "bar"}', 'stop')
REFUSED = SimpleNamespace(
    choices=[SimpleNamespace(finish_reason='content_filter: PROHIBITED_CONTENT')], usage=None
)


class ScriptedCompletions:
    """Answers each request with the next scripted reply and keeps what was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


class DummyClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def _client(completions):
    return OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model', max_tokens=65536),
        client=DummyClient(completions),
        max_tokens=65536,
    )


async def _ask(client, prompt_name='extract_nodes.extract_json'):
    return await client.generate_response(
        [Message(role='system', content='system'), Message(role='user', content='user')],
        response_model=ResponseModel,
        group_id='main',
        prompt_name=prompt_name,
    )


def _levels(completions):
    return [call.get('reasoning_effort') for call in completions.calls]


@pytest.fixture
def env(monkeypatch):
    for name in (
        'GRAPHITI_REASONING_EFFORT',
        'GRAPHITI_REASONING_EFFORT_BY_PROMPT',
        'GRAPHITI_REASONING_FALLBACK',
        'GRAPHITI_THINKING',
        'GRAPHITI_THINKING_BY_PROMPT',
        'GRAPHITI_LLM_TRACE_FILE',
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.asyncio
async def test_with_the_switch_off_the_limit_goes_to_the_queue_after_one_request(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    completions = ScriptedCompletions([LIMIT, WHOLE])
    with pytest.raises(OutputLimitError):
        await _ask(_client(completions))
    assert _levels(completions) == ['high']


@pytest.mark.asyncio
async def test_high_that_runs_into_the_limit_is_asked_again_at_medium(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, WHOLE])
    assert await _ask(_client(completions)) == {'foo': 'bar'}
    assert _levels(completions) == ['high', 'medium']


@pytest.mark.asyncio
async def test_the_ladder_goes_down_one_step_at_a_time_to_none(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'medium')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, LIMIT, WHOLE])
    assert await _ask(_client(completions)) == {'foo': 'bar'}
    assert _levels(completions) == ['medium', 'low', 'none']


@pytest.mark.asyncio
async def test_when_every_step_runs_into_the_limit_the_error_goes_to_the_queue(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, LIMIT, LIMIT, LIMIT, WHOLE])
    with pytest.raises(OutputLimitError):
        await _ask(_client(completions))
    assert _levels(completions) == ['high', 'medium', 'low', 'none']


@pytest.mark.asyncio
@pytest.mark.parametrize('lowest', ['minimal', 'none', 'MINIMAL'])
async def test_a_stage_already_at_the_lowest_level_has_nowhere_to_step(env, lowest):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv(
        'GRAPHITI_REASONING_EFFORT_BY_PROMPT', f'dedupe_nodes={lowest},extract_edges=minimal'
    )
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, WHOLE])
    with pytest.raises(OutputLimitError):
        await _ask(_client(completions), prompt_name='dedupe_nodes.nodes')
    assert _levels(completions) == [lowest]


@pytest.mark.asyncio
async def test_the_level_of_the_stage_is_where_the_ladder_starts(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_nodes=minimal,extract_edges=low')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, WHOLE])
    assert await _ask(_client(completions), prompt_name='extract_edges.edge') == {'foo': 'bar'}
    assert _levels(completions) == ['low', 'none']


@pytest.mark.asyncio
@pytest.mark.parametrize('configured', [None, 'turbo'])
async def test_an_unset_or_unknown_level_is_asked_again_at_medium(env, configured):
    if configured is not None:
        env.setenv('GRAPHITI_REASONING_EFFORT', configured)
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, WHOLE])
    assert await _ask(_client(completions)) == {'foo': 'bar'}
    assert _levels(completions) == [configured, 'medium']


@pytest.mark.asyncio
async def test_the_lowered_level_does_not_outlive_the_call(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, WHOLE, WHOLE])
    client = _client(completions)
    await _ask(client)
    await _ask(client)
    assert _levels(completions) == ['high', 'medium', 'high']


@pytest.mark.asyncio
async def test_every_step_asks_with_the_same_messages(env):
    # generate_response writes the schema and the language instruction into the messages
    # once; asking again must not write them a second time.
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, LIMIT, WHOLE])
    await _ask(_client(completions))
    first, *rest = [call['messages'] for call in completions.calls]
    assert all(messages == first for messages in rest)
    assert first[-1]['content'].count('The following is a JSON Schema.') == 1


@pytest.mark.asyncio
async def test_a_refusal_on_a_lower_step_goes_up_as_it_is(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([LIMIT, REFUSED, WHOLE])
    with pytest.raises(RefusalError):
        await _ask(_client(completions))
    assert _levels(completions) == ['high', 'medium']


@pytest.mark.asyncio
async def test_a_refusal_is_not_asked_again_at_another_level(env):
    env.setenv('GRAPHITI_REASONING_EFFORT', 'high')
    env.setenv('GRAPHITI_REASONING_FALLBACK', 'on')
    completions = ScriptedCompletions([REFUSED, WHOLE])
    with pytest.raises(RefusalError):
        await _ask(_client(completions))
    assert _levels(completions) == ['high']
