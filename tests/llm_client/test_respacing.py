"""The whitespace transform: same text, different setting."""

import json

from graphiti_core.llm_client.respacing import respace_json_blobs, respaced_messages


def test_spaces_the_separators_of_an_embedded_object():
    assert respace_json_blobs('a {"x":1,"y":[1,2]} b') == 'a {"x": 1, "y": [1, 2]} b'


def test_leaves_text_without_json_alone():
    assert respace_json_blobs('просто текст без скобок') == 'просто текст без скобок'


def test_leaves_a_brace_that_is_not_json_alone():
    assert respace_json_blobs('текст { не json } хвост') == 'текст { не json } хвост'


def test_preserves_meaning_exactly():
    original = json.dumps(
        {'participants': {'user': 'Вит'}, 'messages': [{'role': 'user', 'text': 'привет'}]},
        ensure_ascii=False,
        separators=(',', ':'),
    )
    spaced = respace_json_blobs(original)
    assert spaced != original
    assert json.loads(spaced) == json.loads(original)


def test_is_idempotent():
    once = respace_json_blobs('{"a":1,"b":{"c":2}}')
    assert respace_json_blobs(once) == once


def test_keeps_non_ascii_unescaped():
    # ensure_ascii would triple the byte count of Cyrillic and change what the
    # provider is asked to read.
    assert 'Вит' in respace_json_blobs('{"user":"Вит"}')


class _Msg:
    """Stands in for prompts.models.Message: only model_copy and content are used."""

    def __init__(self, content: str):
        self.content = content

    def model_copy(self, update):
        return _Msg(update['content'])


def test_respaced_messages_returns_none_when_nothing_changes():
    assert respaced_messages([_Msg('no json here'), _Msg('nor here')]) is None


def test_respaced_messages_rewrites_only_the_layout():
    original = '{"messages":[{"role":"user","text":"привет"}]}'
    spaced = respaced_messages([_Msg('prompt'), _Msg(original)])
    assert spaced is not None
    assert spaced[0].content == 'prompt'
    assert spaced[1].content != original
    assert json.loads(spaced[1].content) == json.loads(original)
