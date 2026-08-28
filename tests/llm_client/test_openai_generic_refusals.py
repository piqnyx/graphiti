"""Когда провайдер не дал ответа.

Три случая, и ни один из них не «ответ, который надо разобрать»: фильтр отказал,
ответ оборвался на потолке вывода, ответа нет вовсе. Каждый превращается в
названную ошибку, и очередь снаружи решает, что с ней делать.

Отказ больше не повторяется. Раньше он уходил второй раз тем же текстом,
расставленным иначе, -- это давало ответы там, где сжатая форма их не получала,
но это трюк против фильтра, и он стоит второго запроса из квоты, которая
считается, на промпте, про который уже известно, что он отвергнут.

Здесь же проверяется, что каждый случай виден в логе. Ничто это больше не
повторяет, и лог контейнера -- единственное место, где видно, почему батч встал.
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, RefusalError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    foo: str


class Completions:
    """Провайдер, отвечающий ровно тем, что ему велели, и считающий обращения."""

    def __init__(self, choices, usage=None):
        self._choices = choices
        self._usage = usage
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=self._choices, usage=self._usage)


class Client:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def choice(content='', finish_reason='stop'):
    return SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason
    )


def build(completions, max_tokens=4096):
    return OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model', max_tokens=max_tokens),
        client=Client(completions),
        max_tokens=max_tokens,
    )


async def ask(client, **kwargs):
    return await client.generate_response(
        [Message(role='user', content='скажи что-нибудь')],
        response_model=ResponseModel,
        group_id='main',
        prompt_name='extract_nodes.extract_json',
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_refusal_is_not_sent_a_second_time(caplog):
    # Ровно один обход провайдера. Второй был бы трюком против фильтра ценой
    # запроса из считаемой квоты, на промпте, про который уже известно.
    completions = Completions([choice(finish_reason='content_filter: PROHIBITED_CONTENT')])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RefusalError):
            await ask(build(completions))
    assert len(completions.calls) == 1
    # Именно наша строка, а не слово из текста самой ошибки: на общей проверке
    # мутация «убрать запись в лог» проходила незамеченной.
    assert 'prompt refused by the provider' in caplog.text


@pytest.mark.asyncio
async def test_a_block_at_the_prompt_level_is_a_refusal_too():
    # Гугл на блок уровня промпта отвечает пустым списком: обращение к choices[0]
    # роняло бы IndexError изнутри обработчика, написанного для этого же отказа.
    completions = Completions([])
    with pytest.raises(RefusalError):
        await ask(build(completions))
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_an_answer_cut_off_at_the_limit_is_named(caplog):
    # Тоже один обход: обрыв на потолке вывода повтором не лечится, тем же
    # запросом получится тот же обрыв.
    completions = Completions([choice(content='{"foo":', finish_reason='length')])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(Exception) as caught:
            await ask(build(completions))
    assert type(caught.value).__name__ == 'OutputLimitError'
    assert len(completions.calls) == 1
    assert 'cut off' in caplog.text


@pytest.mark.asyncio
async def test_an_empty_answer_is_named(caplog):
    # Мимо обёртки с повторами, напрямую: она считает пустой ответ временным и
    # ходит до четырёх раз с растущей паузой, так что через неё этот тест стоил
    # бы минуты. Проверяется сам страж, а политика повторов -- отдельно ниже.
    completions = Completions([choice(content='')])
    client = build(completions)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(EmptyResponseError):
            await client._generate_response(
                [Message(role='user', content='скажи что-нибудь')],
                response_model=ResponseModel,
            )
    assert 'empty answer' in caplog.text


def test_what_is_worth_a_second_request_and_what_is_not():
    """Пустой ответ повторяют, отказ и обрыв -- нет.

    Не косметика: каждый повтор -- настоящий запрос из считаемой квоты. Пустой
    ответ чаще всего икота провайдера и от повтора проходит; отказ и обрыв от
    повтора тем же текстом не меняются, и тратить на них ещё три обращения
    значит выедать минуту ключа впустую.
    """
    from graphiti_core.llm_client.client import is_server_or_retry_error
    from graphiti_core.llm_client.errors import EmptyResponseError as Empty

    assert is_server_or_retry_error(Empty('пусто'))
    assert not is_server_or_retry_error(RefusalError('отказ'))


@pytest.mark.asyncio
async def test_an_ordinary_answer_still_comes_back():
    # Стражи не должны срабатывать на обычном ответе -- иначе они не стражи, а стена.
    completions = Completions([choice(content='{"foo":"bar"}')])
    assert await ask(build(completions)) == {'foo': 'bar'}
