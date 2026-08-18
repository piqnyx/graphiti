from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    foo: str


class DummyCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"foo":"bar"}'),
                    finish_reason='stop',
                )
            ],
            usage=None,
        )


class DummyClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


@pytest.mark.asyncio
async def test_explicit_prompt_budget_cannot_exceed_deployment_ceiling():
    completions = DummyCompletions()
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model', max_tokens=4096),
        client=DummyClient(completions),
        max_tokens=4096,
    )

    result = await client.generate_response(
        [Message(role='system', content='system'), Message(role='user', content='user')],
        response_model=ResponseModel,
        # Mirrors the accidental hard-coded edge extraction override that used to
        # make a 4K deployment silently request 128K output.
        max_tokens=131072,
        group_id='main',
        prompt_name='extract_edges.edge',
    )

    assert result == {'foo': 'bar'}
    assert len(completions.calls) == 1
    assert completions.calls[0]['max_tokens'] == 4096
