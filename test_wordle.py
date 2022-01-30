import pytest
from wordle import Wordle, render_ascii


@pytest.mark.parametrize(
    "answer,guess,expected",
    [
        ("abcde", "abcde", "+++++"),
        ("aaaab", "aaabc", "+++?_"),
        ("words", "sword", "?????"),
        ("leech", "peace", "_+_+?"),
        ("skill", "lills", "??_+?"),
        ("sense", "enses", "?????"),
        #
        # examples from https://nerdschalk.com/wordle-same-letter-twice-rules-explained-how-does-it-work/
        ("abbey", "opens", "__?__"),
        ("abbey", "babes", "??++_"),
        ("abbey", "kebab", "_?+??"),
        ("abbey", "abyss", "++?__"),
    ],
)
def test_match_words(answer, guess, expected):
    rendered = render_ascii(Wordle.match_words(answer, guess))
    assert rendered == expected
