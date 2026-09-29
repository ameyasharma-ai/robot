"""Tests for STT hallucination filtering."""

from robot.stt import clean_transcript


def test_clean_valid_text():
    assert clean_transcript("Hello, how are you?") == "Hello, how are you?"


def test_reject_short_text():
    assert clean_transcript("hi") == ""
    assert clean_transcript("a") == ""


def test_reject_bracket_hallucination():
    assert clean_transcript("[Music]") == ""
    assert clean_transcript("(Applause)") == ""


def test_reject_repeated_words():
    assert clean_transcript("the the the") == ""


def test_reject_common_hallucinations():
    assert clean_transcript("Thank you.") == ""
    assert clean_transcript("Thanks for watching.") == ""
    assert clean_transcript("Bye") == ""


def test_strip_leading_punctuation():
    assert clean_transcript("...Hello there") == "Hello there"


def test_pass_real_speech():
    assert clean_transcript("What is the weather like today?") != ""
    assert clean_transcript("Play my favorite song") != ""
