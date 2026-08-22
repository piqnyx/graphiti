import json
from types import SimpleNamespace

import openai
import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, OutputLimitError, RateLimitError
from graphiti_core.llm_client.openai_generic_client import (
    _TRACE_CONTEXT,
    OpenAIGenericClient,
    _reasoning_kwargs,
    _thinking_kwargs,
)
from graphiti_core.prompts.models import Message


class DummyChatCompletions:
    def __init__(
        self,
        content: str = '{}',
        error: Exception | None = None,
        finish_reason: str | None = 'stop',
        usage=None,
    ):
        self.create_calls: list[dict] = []
        self._content = content
        self._error = error
        self._finish_reason = finish_reason
        self._usage = usage

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice], usage=self._usage)


class DummyChat:
    def __init__(self, completions: DummyChatCompletions):
        self.completions = completions


class DummyClient:
    def __init__(self, completions: DummyChatCompletions):
        self.chat = DummyChat(completions)


class ResponseModel(BaseModel):
    foo: str


def _messages() -> list[Message]:
    return [
        Message(role='system', content='system message'),
        Message(role='user', content='user message'),
    ]


def _make_client(
    content: str = '{"foo": "bar"}',
    error: Exception | None = None,
    finish_reason: str | None = 'stop',
    usage=None,
    **kwargs,
):
    completions = DummyChatCompletions(
        content=content,
        error=error,
        finish_reason=finish_reason,
        usage=usage,
    )
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model'),
        client=DummyClient(completions),
        **kwargs,
    )
    return client, completions


@pytest.mark.asyncio
async def test_defaults_to_json_schema_response_format():
    client, completions = _make_client()

    await client.generate_response(_messages(), response_model=ResponseModel)

    response_format = completions.create_calls[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'ResponseModel'
    assert response_format['json_schema']['schema'] == ResponseModel.model_json_schema()


@pytest.mark.asyncio
async def test_json_schema_mode_also_injects_schema_into_prompt():
    """json_schema is a hint, not a guarantee.

    Gateways that accept the response_format and then ignore it leave the prompt as the
    only thing steering key names, so the schema is stated in both modes.
    """
    client, completions = _make_client()
    messages = _messages()

    await client.generate_response(messages, response_model=ResponseModel)

    sent_user_content = completions.create_calls[0]['messages'][-1]['content']
    assert 'The following is a JSON Schema' in sent_user_content
    assert json.dumps(ResponseModel.model_json_schema()) in sent_user_content


@pytest.mark.asyncio
async def test_json_object_mode_uses_json_object_and_injects_schema():
    client, completions = _make_client(structured_output_mode='json_object')

    await client.generate_response(_messages(), response_model=ResponseModel)

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    sent_user_content = call['messages'][-1]['content']
    assert 'The following is a JSON Schema' in sent_user_content
    assert 'is not a template to copy' in sent_user_content
    assert json.dumps(ResponseModel.model_json_schema()) in sent_user_content


@pytest.mark.asyncio
async def test_no_response_model_uses_json_object_without_injection():
    client, completions = _make_client(content='{"any": "thing"}')

    result = await client.generate_response(_messages())

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    assert 'The following is a JSON Schema' not in call['messages'][-1]['content']
    assert result == {'any': 'thing'}


@pytest.mark.asyncio
async def test_rate_limit_error_is_translated():
    rate_limit = openai.RateLimitError(
        message='slow down',
        response=SimpleNamespace(status_code=429, headers={}, request=None),
        body=None,
    )
    client, _ = _make_client(error=rate_limit)

    # Assert translation at the _generate_response level. Going through generate_response
    # would invoke the inherited tenacity retry wrapper (RateLimitError is retryable), which
    # adds real backoff sleeps and would make this unit test slow.
    with pytest.raises(RateLimitError):
        await client._generate_response(_messages(), response_model=ResponseModel)


@pytest.mark.asyncio
async def test_empty_content_raises_empty_response_error():
    # Empty body from a flaky/refusing endpoint is a transient class. A provider that
    # explicitly says finish_reason=length is handled separately as OutputLimitError.
    client, _ = _make_client(content='')

    with pytest.raises(EmptyResponseError):
        await client._generate_response(_messages(), response_model=ResponseModel)


def test_empty_response_error_is_retryable():
    from graphiti_core.llm_client.client import is_server_or_retry_error

    assert is_server_or_retry_error(EmptyResponseError('empty')) is True


@pytest.mark.asyncio
async def test_output_limit_is_not_immediately_retried():
    client, completions = _make_client(
        content='{"foo":"truncated',
        finish_reason='length',
    )

    with pytest.raises(OutputLimitError, match='max_tokens=4096'):
        await client.generate_response(
            _messages(),
            response_model=ResponseModel,
            max_tokens=4096,
            group_id='main',
            prompt_name='extract_edges.edge',
        )

    # An unchanged length-limited request will deterministically burn the same budget.
    # The durable episode queue owns the later retry, after backoff/config changes.
    assert len(completions.create_calls) == 1


@pytest.mark.asyncio
async def test_hidden_reasoning_that_spends_the_whole_budget_before_content_is_output_limit():
    # Some compatible gateways do not report finish_reason=length when hidden
    # reasoning consumes the entire completion budget. The provider usage is the
    # second signal and prevents four identical full-budget retries.
    usage = SimpleNamespace(prompt_tokens=3581, completion_tokens=4096, total_tokens=7677)
    client, completions = _make_client(content='', finish_reason='stop', usage=usage)

    with pytest.raises(OutputLimitError, match='budget exhausted before content'):
        await client.generate_response(
            _messages(),
            response_model=ResponseModel,
            max_tokens=4096,
            group_id='main',
            prompt_name='extract_edges.edge',
        )

    assert len(completions.create_calls) == 1


@pytest.mark.asyncio
async def test_full_budget_malformed_json_is_output_limit_not_json_retry():
    usage = SimpleNamespace(prompt_tokens=3581, completion_tokens=4096, total_tokens=7677)
    client, completions = _make_client(
        content='{"foo":"never closed',
        finish_reason='stop',
        usage=usage,
    )

    with pytest.raises(OutputLimitError, match='malformed JSON'):
        await client.generate_response(
            _messages(),
            response_model=ResponseModel,
            max_tokens=4096,
            group_id='main',
            prompt_name='extract_edges.edge',
        )

    assert len(completions.create_calls) == 1


@pytest.mark.asyncio
async def test_exact_jsonl_trace_records_real_request_and_raw_response(tmp_path, monkeypatch):
    trace_path = tmp_path / 'llm-trace.jsonl'
    monkeypatch.setenv('GRAPHITI_LLM_TRACE_FILE', str(trace_path))
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT', 'none')

    # Deliberately odd usage data proves the trace records what the gateway says,
    # rather than reconstructing token counts locally. A valid structured body is
    # still accepted even if a compatible proxy reports surprising usage metadata.
    usage = SimpleNamespace(prompt_tokens=3581, completion_tokens=65536, total_tokens=69117)
    client, completions = _make_client(usage=usage)

    result = await client.generate_response(
        _messages(),
        response_model=ResponseModel,
        max_tokens=4096,
        group_id='main',
        prompt_name='extract_edges.edge',
    )
    assert result == {'foo': 'bar'}

    records = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    assert [record['event'] for record in records] == ['request', 'response']

    request = records[0]
    sent = completions.create_calls[0]
    assert request['prompt_name'] == 'extract_edges.edge'
    assert request['group_id'] == 'main'
    assert request['request'] == sent
    assert request['request']['model'] == 'test-model'
    assert request['request']['max_tokens'] == 4096
    assert request['request']['temperature'] == client.temperature
    assert request['request']['reasoning_effort'] == 'none'
    assert request['request']['response_format']['type'] == 'json_schema'
    # The prompt carries the schema as prose as well as in ``response_format``,
    # because the gateways this client targets accept the field and ignore it.
    last = request['request']['messages'][-1]
    assert last['role'] == 'user'
    assert last['content'].startswith('user message')
    assert 'JSON Schema' in last['content']

    response = records[1]
    assert response['request_id'] == request['request_id']
    assert response['prompt_name'] == 'extract_edges.edge'
    assert response['finish_reason'] == 'stop'
    assert response['usage'] == {
        'prompt_tokens': 3581,
        'completion_tokens': 65536,
        'total_tokens': 69117,
    }
    assert response['content'] == '{"foo": "bar"}'
    assert response['content_chars'] == len('{"foo": "bar"}')


@pytest.mark.asyncio
async def test_strips_markdown_code_fence_before_parsing():
    fenced = '```json\n{"foo": "bar"}\n```'
    client, _ = _make_client(content=fenced)

    result = await client.generate_response(_messages(), response_model=ResponseModel)

    assert result == {'foo': 'bar'}


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried():
    client, completions = _make_client(error=ValueError('bad response'))

    with pytest.raises(ValueError):
        await client.generate_response(_messages(), response_model=ResponseModel)

    assert len(completions.create_calls) == 1


def _with_prompt(name: str):
    """Enter the tracing context the request builder reads the stage name from."""
    return _TRACE_CONTEXT.set({'prompt_name': name})


def test_thinking_mode_without_overrides_is_the_global_one(monkeypatch):
    monkeypatch.setenv('GRAPHITI_THINKING', 'enabled')
    monkeypatch.delenv('GRAPHITI_THINKING_BY_PROMPT', raising=False)
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _thinking_kwargs() == {'extra_body': {'thinking': {'type': 'enabled'}}}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_a_matching_stage_override_wins_over_the_global_mode(monkeypatch):
    monkeypatch.setenv('GRAPHITI_THINKING', 'enabled')
    monkeypatch.setenv('GRAPHITI_THINKING_BY_PROMPT', 'dedupe_nodes=disabled')
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _thinking_kwargs() == {'extra_body': {'thinking': {'type': 'disabled'}}}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_a_stage_the_overrides_do_not_name_keeps_the_global_mode(monkeypatch):
    monkeypatch.setenv('GRAPHITI_THINKING', 'enabled')
    monkeypatch.setenv('GRAPHITI_THINKING_BY_PROMPT', 'dedupe_nodes=disabled')
    token = _with_prompt('dedupe_edges.resolve_edge')
    try:
        assert _thinking_kwargs() == {'extra_body': {'thinking': {'type': 'enabled'}}}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_overrides_are_ignored_when_the_stage_is_unknown(monkeypatch):
    """No prompt name means no way to tell the stages apart, so the global mode stands."""
    monkeypatch.setenv('GRAPHITI_THINKING', 'enabled')
    monkeypatch.setenv('GRAPHITI_THINKING_BY_PROMPT', 'dedupe_nodes=disabled')
    token = _TRACE_CONTEXT.set(None)
    try:
        assert _thinking_kwargs() == {'extra_body': {'thinking': {'type': 'enabled'}}}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_an_unset_switch_still_sends_nothing(monkeypatch):
    """The behaviour before any of this existed, and the one gateways expect."""
    monkeypatch.delenv('GRAPHITI_THINKING', raising=False)
    monkeypatch.setenv('GRAPHITI_THINKING_BY_PROMPT', 'dedupe_nodes=nonsense')
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _thinking_kwargs() == {}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_reasoning_effort_without_overrides_is_the_global_one(monkeypatch):
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT', 'medium')
    monkeypatch.delenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', raising=False)
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _reasoning_kwargs() == {'reasoning_effort': 'medium'}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_a_matching_stage_override_wins_over_the_global_effort(monkeypatch):
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT', 'medium')
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_nodes=minimal')
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _reasoning_kwargs() == {'reasoning_effort': 'minimal'}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_a_stage_the_effort_overrides_do_not_name_keeps_the_global_one(monkeypatch):
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT', 'medium')
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_nodes=minimal')
    token = _with_prompt('dedupe_edges.resolve_edge')
    try:
        assert _reasoning_kwargs() == {'reasoning_effort': 'medium'}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_effort_overrides_are_ignored_when_the_stage_is_unknown(monkeypatch):
    """No prompt name means no way to tell the stages apart, so the global effort stands."""
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT', 'medium')
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_nodes=minimal')
    token = _TRACE_CONTEXT.set(None)
    try:
        assert _reasoning_kwargs() == {'reasoning_effort': 'medium'}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_an_unset_effort_still_sends_nothing(monkeypatch):
    """The behaviour before any of this existed, and the one gateways expect."""
    monkeypatch.delenv('GRAPHITI_REASONING_EFFORT', raising=False)
    monkeypatch.delenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', raising=False)
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _reasoning_kwargs() == {}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_an_unmatched_stage_override_leaves_the_effort_unset(monkeypatch):
    """A stage the overrides do not name falls back to the global value, empty or not."""
    monkeypatch.delenv('GRAPHITI_REASONING_EFFORT', raising=False)
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_edges=high')
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _reasoning_kwargs() == {}
    finally:
        _TRACE_CONTEXT.reset(token)


def test_a_matching_stage_override_sends_even_with_the_global_unset(monkeypatch):
    """A named stage speaks for itself, and that is a trap worth stating.

    Clearing the global switch does not silence a stage the overrides still name:
    the override is a value for that stage, not a modifier of a value that must
    already exist. Measured 2026-08-22 on the thinking switch, which behaves the
    same way -- clearing GRAPHITI_THINKING while leaving
    GRAPHITI_THINKING_BY_PROMPT=dedupe_nodes=disabled kept sending the thinking
    object on exactly that one stage, and the gateway rejected every dedupe_nodes
    call with `Unknown name "thinking"` while every other stage went through.

    Moving between gateways therefore means clearing both variables, not one.
    """
    monkeypatch.delenv('GRAPHITI_REASONING_EFFORT', raising=False)
    monkeypatch.setenv('GRAPHITI_REASONING_EFFORT_BY_PROMPT', 'dedupe_nodes=minimal')
    token = _with_prompt('dedupe_nodes.nodes')
    try:
        assert _reasoning_kwargs() == {'reasoning_effort': 'minimal'}
    finally:
        _TRACE_CONTEXT.reset(token)
