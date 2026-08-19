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

import json
import logging
import os
import re
import threading
import typing
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, OutputLimitError, RateLimitError

logger = logging.getLogger(__name__)

_TRACE_LOCK = threading.Lock()
_TRACE_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    'graphiti_openai_generic_trace_context', default=None
)


def _reasoning_kwargs() -> dict[str, Any]:
    """Reasoning effort for this request, if the deployment asked for one.

    This is intentionally an environment switch because OpenAI-compatible gateways
    differ in which reasoning values they accept. The exact request/response trace
    can be enabled separately to verify what was actually sent and what the gateway
    returned instead of inferring provider behaviour from token totals.
    """
    effort = os.environ.get('GRAPHITI_REASONING_EFFORT', '').strip()
    return {'reasoning_effort': effort} if effort else {}


def _trace_value(value: Any) -> Any:
    """Convert SDK response objects to JSON-compatible diagnostic values."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        try:
            return model_dump(mode='json')
        except TypeError:
            return model_dump()
    try:
        return vars(value)
    except TypeError:
        return str(value)


def _completion_tokens(usage: Any) -> int | None:
    """Return provider-reported completion tokens without assuming an SDK shape."""
    if usage is None:
        return None
    value = usage.get('completion_tokens') if isinstance(usage, dict) else getattr(
        usage, 'completion_tokens', None
    )
    return value if isinstance(value, int) and value >= 0 else None


def _trace_llm_event(event: dict[str, Any]) -> None:
    """Append an opt-in JSONL event containing the real LLM request or response.

    Set GRAPHITI_LLM_TRACE_FILE to a writable path to enable it. Conversation text
    and raw model output are intentionally included because this trace exists to
    reproduce malformed extraction calls. API credentials are never included.
    Trace I/O is best-effort and must never break ingestion.
    """
    path = os.environ.get('GRAPHITI_LLM_TRACE_FILE', '').strip()
    if not path:
        return

    record = {'timestamp': datetime.now(timezone.utc).isoformat(), **event}
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str, separators=(',', ':'))
        with _TRACE_LOCK:
            with open(path, 'a', encoding='utf-8') as trace_file:
                trace_file.write(line)
                trace_file.write('\n')
                trace_file.flush()
            try:
                os.chmod(path, 0o600)
            except OSError:
                # Permissions are best-effort diagnostics, never ingestion state.
                pass
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break ingestion
        logger.warning('Failed to write GRAPHITI_LLM_TRACE_FILE: %s', exc)


DEFAULT_MODEL = 'gpt-4.1-mini'

StructuredOutputMode = Literal['json_schema', 'json_object']


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    This client targets any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    vLLM, llama.cpp, Ollama, DeepSeek, Together, etc.). It defaults to native
    ``json_schema`` structured output (constrained decoding) and can fall back to
    ``json_object`` for the minority of providers that do not support ``json_schema``.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): Deployment-wide maximum output budget.
        structured_output_mode (StructuredOutputMode): How structured output is requested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_schema',
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): Deployment-wide maximum output budget. Per-prompt requests may
                ask for less but cannot silently raise this ceiling.
            structured_output_mode (StructuredOutputMode): Whether to request structured
                output via native ``json_schema`` (the default, uses constrained decoding)
                or to fall back to ``json_object``. Set to ``'json_object'`` for providers
                that do not support the ``json_schema`` response format (e.g. DeepSeek); in
                that mode the schema is injected into the prompt instead of being enforced
                by the API.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # The factory passes deployment config here. Treat it as an upper bound so
        # one extraction helper cannot accidentally turn a 4K deployment into 128K.
        self.max_tokens = max_tokens
        self.structured_output_mode: StructuredOutputMode = structured_output_mode

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _build_response_format(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        """Build the ``response_format`` payload for the chat completion request.

        Uses native ``json_schema`` when a response model is provided and the client is in
        ``json_schema`` mode; otherwise falls back to ``json_object``. In ``json_object``
        mode the schema is not enforced by the API — ``generate_response`` injects it into
        the prompt instead.
        """
        if response_model is None or self.structured_output_mode == 'json_object':
            return {'type': 'json_object'}

        # Native json_schema. We intentionally omit "strict": true — strict mode requires
        # the schema to meet OpenAI's strict subset (additionalProperties: false, every
        # field required), which raw model_json_schema() routinely violates (that's why the
        # dedicated OpenAIClient uses responses.parse() instead). So adherence is best-effort
        # on OpenAI-proper; constrained-decoding servers (vLLM, llama.cpp) still enforce it.
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': getattr(response_model, '__name__', 'structured_response'),
                'schema': response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a wrapping markdown code fence from a JSON payload.

        OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap JSON in a
        ```json … ``` fence even when a json_schema/json_object response_format is requested,
        which breaks a bare ``json.loads``. No-op when there is no fence.
        """
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
            stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
        return stripped.strip()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        request_id = uuid4().hex
        request_kwargs: dict[str, Any] = {
            'model': self.model or DEFAULT_MODEL,
            'messages': openai_messages,
            'temperature': self.temperature,
            'max_tokens': max_tokens,
            'response_format': self._build_response_format(response_model),
            **_reasoning_kwargs(),
        }
        trace_context = _TRACE_CONTEXT.get() or {}
        _trace_llm_event(
            {
                'event': 'request',
                'request_id': request_id,
                'base_url': str(getattr(self.client, 'base_url', '')),
                'structured_output_mode': self.structured_output_mode,
                'response_model': getattr(response_model, '__name__', None),
                **trace_context,
                # This exact dict is expanded into create() below. The trace is
                # therefore the request, not a separately reconstructed approximation.
                'request': request_kwargs,
            }
        )

        try:
            response = await self.client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            result = choice.message.content or ''
            finish_reason = getattr(choice, 'finish_reason', None)
            usage = getattr(response, 'usage', None)
            completion_tokens = _completion_tokens(usage)
            _trace_llm_event(
                {
                    'event': 'response',
                    'request_id': request_id,
                    **trace_context,
                    'finish_reason': finish_reason,
                    'usage': _trace_value(usage),
                    'content_chars': len(result),
                    # Preserve the exact body before fences are stripped or JSON is parsed.
                    'content': result,
                }
            )

            # Explicit length is definitive. Some OpenAI-compatible gateways have also
            # been observed to report stop/empty content while billing exactly the entire
            # completion budget to hidden reasoning. Treat that as the same failure when
            # the body is absent or cannot be parsed below.
            budget_exhausted = completion_tokens is not None and completion_tokens >= max_tokens
            if finish_reason == 'length':
                raise OutputLimitError(
                    f'LLM output limit reached (max_tokens={max_tokens}, completion_tokens={completion_tokens}, content_chars={len(result)})'
                )

            if not result:
                if budget_exhausted:
                    raise OutputLimitError(
                        f'LLM completion budget exhausted before content (max_tokens={max_tokens}, completion_tokens={completion_tokens})'
                    )
                raise EmptyResponseError('LLM returned an empty response')

            try:
                parsed = json.loads(self._strip_code_fences(result))
            except json.JSONDecodeError as exc:
                if budget_exhausted:
                    raise OutputLimitError(
                        f'LLM completion budget exhausted with malformed JSON (max_tokens={max_tokens}, completion_tokens={completion_tokens}, content_chars={len(result)})'
                    ) from exc
                raise

            if response_model is not None:
                try:
                    return response_model.model_validate(parsed).model_dump()
                except Exception as exc:
                    if budget_exhausted:
                        raise OutputLimitError(
                            f'LLM completion budget exhausted with incomplete structured output (max_tokens={max_tokens}, completion_tokens={completion_tokens}, content_chars={len(result)})'
                        ) from exc
                    raise
            return parsed
        except openai.RateLimitError as e:
            _trace_llm_event(
                {
                    'event': 'error',
                    'request_id': request_id,
                    **trace_context,
                    'error_type': type(e).__name__,
                    'error': str(e),
                }
            )
            raise RateLimitError from e
        except Exception as e:
            _trace_llm_event(
                {
                    'event': 'error',
                    'request_id': request_id,
                    **trace_context,
                    'error_type': type(e).__name__,
                    'error': str(e),
                }
            )
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        requested_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        # The deployment config is a hard ceiling. This makes the central
        # llm.max_tokens setting authoritative even if a helper passes 131072.
        max_tokens = min(requested_max_tokens, self.max_tokens)

        # The schema goes into the prompt in BOTH modes. Not every OpenAI-compatible
        # gateway honours json_schema: opencode.ai/zen accepts the response_format for
        # mimo-v2.5 and then returns whatever key names the model preferred
        # ({"entities": ...} for ExtractedEntities, {"facts": ...} for ExtractedEdges),
        # which fails validation on every attempt. Where json_schema is genuinely
        # enforced the extra text is merely redundant.
        #
        # The wording carries as much weight as the schema itself. "Respond with a JSON
        # object in the following format: {schema}" reads as an instruction to return
        # that document: roughly one call in three came back as the schema, $defs and
        # all. Naming it as a schema and demanding an instance removes that reading.
        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            top_level_keys = ', '.join(response_model.model_fields)
            messages[-1].content += (
                '\n\nThe following is a JSON Schema. It describes the shape of the answer; '
                'it is not a template to copy and it is not the answer itself:'
                f'\n\n{serialized_model}\n\n'
                'Reply with a single JSON object that is an instance of that schema and '
                f'nothing else. Its top-level keys must be exactly: {top_level_keys}. '
                'Never emit "$defs", "properties", "required", "title" or any other JSON '
                'Schema keyword in the reply: those describe the data, they are not data.'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            token = _TRACE_CONTEXT.set(
                {
                    'prompt_name': prompt_name,
                    'group_id': group_id,
                    'model_size': model_size.value,
                    'requested_max_tokens': requested_max_tokens,
                    'effective_max_tokens': max_tokens,
                }
            )
            try:
                # Delegate to the base tenacity wrapper so genuinely transient JSON /
                # rate-limit failures get bounded backoff retries. OutputLimitError is
                # intentionally not retryable here: the outer durable queue owns that wait.
                return await self._generate_response_with_retry(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise
            finally:
                _TRACE_CONTEXT.reset(token)
