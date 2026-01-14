import pytest


def test_argos_translator_missing_model(monkeypatch):
    argos_translate = pytest.importorskip("argostranslate.translate")
    monkeypatch.setattr(argos_translate, "get_installed_languages", lambda: [])

    from all_metrics.lev import ArgosTranslator

    translator = ArgosTranslator(model_path=None)
    with pytest.raises(RuntimeError) as excinfo:
        translator.ensure_ready()
    assert "EDUTEXT_ARGOS_MODEL_PATH" in str(excinfo.value)


def test_argos_translator_translate_if_available():
    pytest.importorskip("argostranslate.translate")

    from all_metrics.lev import ArgosTranslator

    translator = ArgosTranslator(model_path=None)
    try:
        translator.ensure_ready()
    except RuntimeError:
        pytest.skip("Argos model not installed for en->ru")

    result = translator.translate("test")
    assert isinstance(result, str)
