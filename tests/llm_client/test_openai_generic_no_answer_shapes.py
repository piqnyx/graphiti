"""Формы «ответа нет», которых ни одна подделка в наборе не строила.

Стражи в `_generate_response` существуют ради одного: превратить «провайдер не
дал ответа» в названную ошибку вместо падения внутри обработчика, написанного
ровно для этого случая. Когда падение всё же случается, оно приходит наружу как
AttributeError или ValidationError -- а это классы, которые обёртка повторов
считает временными, так что один отвергнутый промпт уходит к гуглу ещё три раза
и после этого долговечная очередь проигрывает весь эпизод заново. Один такой
батч держал голову очереди десять часов.

Соседний файл проверяет случаи, которые уже строились. Здесь -- три, которые не
строились ни разу:

* choice, у которого `message` равен None. Именно так SDK отдаёт отказ: поле есть,
  значения нет. Все подделки в наборе кладут туда объект с `content`, поэтому
  форма, ради которой страж и написан, ни одним тестом не проходилась.
* исчерпанный бюджет вывода, доехавший до проверки схемы. Пустое тело и битый
  json свои тесты имеют; ответ, который разобрался, но модели не соответствует,
  не имел.
* повтор пустого ответа: что он разрешён предикатом, проверено; что он
  действительно случается -- нет.
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from tenacity import wait_none

from graphiti_core.llm_client.client import LLMClient, is_server_or_retry_error
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, OutputLimitError, RefusalError
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
    return SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)


def build(completions, max_tokens=4096):
    return OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model', max_tokens=max_tokens),
        client=Client(completions),
        max_tokens=max_tokens,
    )


async def ask(client, **kwargs):
    """Как в проде: через обёртку повторов, чтобы было видно число обращений."""
    return await client.generate_response(
        [Message(role='user', content='скажи что-нибудь')],
        response_model=ResponseModel,
        group_id='main',
        prompt_name='extract_nodes.extract_json',
        **kwargs,
    )


async def ask_directly(client, max_tokens=4096):
    """Мимо обёртки: если страж убрать, ошибка под ним временная и обёртка будет
    ходить четыре раза с растущей паузой -- тест на сломанном коде стоил бы минут,
    а проверяется здесь сам страж."""
    return await client._generate_response(
        [Message(role='user', content='скажи что-нибудь')],
        response_model=ResponseModel,
        max_tokens=max_tokens,
    )


@pytest.mark.asyncio
async def test_a_refusal_whose_choice_carries_no_message_object(caplog):
    """Отказ приходит как choice с finish_reason и message=None.

    Поле на месте, значения в нём нет -- SDK строит объект по схеме и оставляет
    пустое пустым. Дотянуться сквозь него значит получить AttributeError на
    строке выше стража, написанного для этого самого отказа; наружу это уходит
    безымянной ошибкой, которую обёртка считает временной, и отвергнутый промпт
    уезжает к гуглу ещё трижды из считаемой квоты.
    """
    completions = Completions(
        [SimpleNamespace(message=None, finish_reason='content_filter: PROHIBITED_CONTENT')]
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RefusalError):
            await ask(build(completions))
    assert len(completions.calls) == 1
    # Имя промпта в записи -- то единственное, что говорит, какой батч встал.
    assert 'prompt=extract_nodes.extract_json' in caplog.text


@pytest.mark.asyncio
async def test_a_choice_without_a_message_field_at_all():
    """Вторая форма того же: поля message нет вовсе.

    Страж читает через getattr с умолчанием и потому переживает обе; проверяются
    обе, потому что каждая ломается своим способом обращения к полю.
    """
    completions = Completions([SimpleNamespace(finish_reason='stop')])
    completions_client = build(completions)
    with pytest.raises(EmptyResponseError):
        await ask_directly(completions_client)
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_a_full_budget_answer_that_parses_but_fails_the_schema():
    """Бюджет вывода съеден скрытым рассуждением, а тело -- разбираемый `{}`.

    Пустое тело и битый json свои ветки имеют, эта -- третья: json разобрался,
    модели не соответствует. Без неё наружу уходит ValidationError, и это ровно
    тот класс, который обёртка повторяет: четыре одинаковых запроса, каждый с
    полным бюджетом, на промпте, про который уже известно, что ответа не будет.
    """
    completions = Completions([choice(content='{}')], usage=SimpleNamespace(completion_tokens=64))
    with pytest.raises(OutputLimitError, match='incomplete structured output') as caught:
        await ask_directly(build(completions, max_tokens=64), max_tokens=64)
    assert len(completions.calls) == 1
    # Названность здесь и нужна затем, чтобы повтора не было.
    assert not is_server_or_retry_error(caught.value)


@pytest.mark.asyncio
async def test_an_empty_answer_really_is_sent_again(monkeypatch):
    """Пустой ответ доходит до провайдера второй раз, а не только разрешён.

    Что предикат зовёт пустой ответ временным, проверено отдельно; что обёртка
    этот предикат для него действительно спрашивает -- нет. Икота провайдера
    проходит от повтора, и потерять эти три попытки значит уронить эпизод там,
    где он бы дошёл. Пауза убрана: проверяется число обращений, не расписание.
    """
    monkeypatch.setattr(LLMClient._generate_response_with_retry.retry, 'wait', wait_none())
    completions = Completions([choice(content='')])
    with pytest.raises(EmptyResponseError):
        await ask(build(completions))
    assert len(completions.calls) == 4
