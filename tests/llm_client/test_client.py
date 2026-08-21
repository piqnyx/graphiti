"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from graphiti_core.llm_client.client import LLMClient, get_extraction_language_instruction
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class MockLLMClient(LLMClient):
    """Concrete implementation of LLMClient for testing"""

    async def _generate_response(self, messages, response_model=None):
        return {'content': 'test'}


def test_clean_input():
    client = MockLLMClient(LLMConfig())

    test_cases = [
        # Basic text should remain unchanged
        ('Hello World', 'Hello World'),
        # Control characters should be removed
        ('Hello\x00World', 'HelloWorld'),
        # Newlines, tabs, returns should be preserved
        ('Hello\nWorld\tTest\r', 'Hello\nWorld\tTest\r'),
        # Invalid Unicode should be removed
        ('Hello\udcdeWorld', 'HelloWorld'),
        # Zero-width characters should be removed
        ('Hello\u200bWorld', 'HelloWorld'),
        ('Test\ufeffWord', 'TestWord'),
        # Multiple issues combined
        ('Hello\x00\u200b\nWorld\udcde', 'Hello\nWorld'),
        # Empty string should remain empty
        ('', ''),
        # Form feed and other control characters from the error case
        ('{"edges":[{"relation_typ...\f\x04Hn\\?"}]}', '{"edges":[{"relation_typ...Hn\\?"}]}'),
        # More specific control character tests
        ('Hello\x0cWorld', 'HelloWorld'),  # form feed \f
        ('Hello\x04World', 'HelloWorld'),  # end of transmission
        # Combined JSON-like string with control characters
        ('{"test": "value\f\x00\x04"}', '{"test": "value"}'),
    ]

    for input_str, expected in test_cases:
        assert client._clean_input(input_str) == expected, f'Failed for input: {repr(input_str)}'


def test_attribute_extraction_preamble_no_op_when_disabled():
    client = MockLLMClient(LLMConfig())
    messages = [Message(role='system', content='base'), Message(role='user', content='hi')]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=False)
    assert messages[0].content == 'base'
    assert messages[1].content == 'hi'


def test_attribute_extraction_preamble_appends_to_system():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='You are helpful.'),
        Message(role='user', content='hi'),
    ]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages[0].content.startswith('You are helpful.')
    assert 'ATTRIBUTE EXTRACTION:' in messages[0].content
    assert 'NEVER themselves valid values' in messages[0].content
    assert messages[1].content == 'hi'  # user message untouched


def test_attribute_extraction_preamble_is_idempotent():
    client = MockLLMClient(LLMConfig())
    messages = [
        Message(role='system', content='You are helpful.'),
        Message(role='user', content='hi'),
    ]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    once = messages[0].content
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages[0].content == once, 'second call must not double-append'


def test_attribute_extraction_preamble_falls_back_to_first_message_if_no_system():
    client = MockLLMClient(LLMConfig())
    messages = [Message(role='user', content='hi')]
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert 'ATTRIBUTE EXTRACTION:' in messages[0].content
    assert messages[0].content.endswith('hi')
    # Sentinel must be at the front so the idempotency check finds it.
    assert messages[0].content.startswith('<<graphiti.attr_extraction.preamble.v1>>')


def test_attribute_extraction_preamble_handles_empty_messages():
    client = MockLLMClient(LLMConfig())
    messages: list[Message] = []
    client._apply_attribute_extraction_preamble(messages, attribute_extraction=True)
    assert messages == []


def test_language_instruction_defaults_to_inferring_from_the_source(monkeypatch):
    monkeypatch.delenv('GRAPHITI_OUTPUT_LANGUAGE', raising=False)
    instruction = get_extraction_language_instruction()
    assert 'same language as it was written in' in instruction
    assert 'SCREAMING_SNAKE_CASE' not in instruction


def test_a_pinned_language_still_leaves_relation_types_in_english(monkeypatch):
    """The pinned language is prose only.

    Search and every downstream consumer match on relation types, so a type in the
    source script is unusable. This test exists because saying "keep them English"
    was not enough on its own: a word with no English equivalent left the model
    choosing which of two instructions to break.
    """
    monkeypatch.setenv('GRAPHITI_OUTPUT_LANGUAGE', 'Russian')
    instruction = get_extraction_language_instruction()
    assert 'Russian' in instruction
    assert 'SCREAMING_SNAKE_CASE' in instruction
    # Both branches, or the impossible case is unaddressed again.
    assert 'LIKES' in instruction and 'KHACHAPURI' in instruction


class _StatusError(Exception):
    """An SDK exception carrying a status the way openai's APIStatusError does."""

    def __init__(self, status_code: int):
        super().__init__(f'status {status_code}')
        self.status_code = status_code


def test_a_provider_500_is_retryable_however_the_sdk_raised_it():
    from graphiti_core.llm_client.client import is_server_or_retry_error

    # openai.InternalServerError is not an httpx exception, so it went unrecognised:
    # the call failed, the episode failed with it, and the whole extraction was
    # replayed from the beginning to reach the one stage that had broken.
    assert is_server_or_retry_error(_StatusError(500))
    assert is_server_or_retry_error(_StatusError(503))


def test_a_client_error_is_not_retried():
    from graphiti_core.llm_client.client import is_server_or_retry_error

    # Retrying these changes nothing and costs a full extraction each time.
    assert not is_server_or_retry_error(_StatusError(400))
    assert not is_server_or_retry_error(_StatusError(404))
    assert not is_server_or_retry_error(ValueError('unrelated'))


def test_an_httpx_500_is_still_retryable():
    import httpx

    from graphiti_core.llm_client.client import is_server_or_retry_error

    request = httpx.Request('POST', 'http://example.invalid')
    error = httpx.HTTPStatusError(
        'boom', request=request, response=httpx.Response(502, request=request)
    )
    assert is_server_or_retry_error(error)
