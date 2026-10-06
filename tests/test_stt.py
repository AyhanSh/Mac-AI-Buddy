"""Speech-to-text post-processing (no model involved)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.speech.stt import fix_name  # noqa: E402


@pytest.mark.parametrize("heard,meant", [
    ("Hey Matt, could you please open the YouTube?",
     "Hey Mac, could you please open the YouTube?"),
    ("Hey man, open YouTube.", "Hey Mac, open YouTube."),
    ("Okay Max open Figma", "Okay Mac open Figma"),
])
def test_a_misheard_name_in_a_greeting_is_the_robots(heard, meant):
    assert fix_name(heard) == meant


@pytest.mark.parametrize("text", [
    "Tell Matt I said hi",          # a person, not the robot
    "Hey Mac, how are you?",
    "Привет, как дела?",
])
def test_everything_else_is_left_alone(text):
    assert fix_name(text) == text


# -- telling the person apart from background sound -----------------

from minibot.speech.stt import MlxWhisperSTT, foreign_script  # noqa: E402


def stt(langs=("en", "ru")):
    return MlxWhisperSTT(languages=set(langs))


@pytest.mark.parametrize("text,lang", [
    ("Hey Mac, could you please open YouTube?", "en"),
    ("Открой YouTube, пожалуйста.", "ru"),     # Latin brand inside Russian
])
def test_real_speech_in_your_languages_is_kept(text, lang):
    assert stt()._reject(text, lang, -0.3, 1.0) is None


@pytest.mark.parametrize("text,lang,logprob,ratio,why", [
    # observed live while YouTube was open
    ("次の動画でお会いしましょう。", "ja", -0.4, 1.0, "not one of"),
    ("Über compromised Sierra 1981 переж Motoええええ", "en", -0.5, 1.0, "hiragana"),
    # measured: room noise decoded as gibberish at -8.7
    ("Same adjustаю ritinho com Uber", "en", -8.7, 0.9, "low confidence"),
    # music comes out as one confident phrase on a loop
    ("I'm like, " * 30, "en", -0.2, 9.0, "repeating"),
])
def test_background_sound_is_ignored(text, lang, logprob, ratio, why):
    assert why in stt()._reject(text, lang, logprob, ratio)


def test_no_language_list_accepts_any_language():
    assert MlxWhisperSTT()._reject("次の動画でお会いしましょう。", "ja", -0.4, 1.0) is None


def test_foreign_script_allows_scripts_of_your_languages():
    assert foreign_script("Привет, YouTube", {"en", "ru"}) is None
    assert foreign_script("Привет", {"en"}) == "cyrillic"
