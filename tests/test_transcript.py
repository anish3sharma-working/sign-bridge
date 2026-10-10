from services.speech_service import normalize_transcript


def test_normalizer_only_changes_spacing_and_punctuation_spacing():
    assert normalize_transcript("  hello   ,  world !  ") == "hello, world!"
    assert normalize_transcript("") == ""
