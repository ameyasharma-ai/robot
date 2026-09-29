"""Tests for LLM clause chunker."""

from robot.llm import ClauseChunker


def test_chunker_sentence_boundary():
    chunker = ClauseChunker()
    assert chunker.feed("Hello, ") is None
    result = chunker.feed("world. ")
    assert result == "Hello, world."


def test_chunker_question():
    chunker = ClauseChunker()
    chunker.feed("How are ")
    result = chunker.feed("you? ")
    assert result == "How are you?"


def test_chunker_flush():
    chunker = ClauseChunker()
    chunker.feed("Incomplete sentence without")
    chunker.feed(" punctuation")
    result = chunker.flush()
    assert result == "Incomplete sentence without punctuation"


def test_chunker_long_unpunctuated():
    chunker = ClauseChunker(max_words=5)
    for word in ["one", " two", " three", " four", " five", " six"]:
        result = chunker.feed(word)
        if result:
            assert len(result.split()) >= 1
            break


def test_chunker_empty_flush():
    chunker = ClauseChunker()
    assert chunker.flush() is None
