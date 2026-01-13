from all_metrics.exercise_types import choose_prediction, detect_lang_simple


def test_detect_lang_simple():
    assert detect_lang_simple("Прочитай текст") == "ru"
    assert detect_lang_simple("Read the text") == "en"
    assert detect_lang_simple("Read и ответь") == "mixed"


def test_choose_prediction_gating():
    pred, conf, chosen, lang = choose_prediction(
        "Прочитай текст", "ru_label", 0.2, "en_label", 0.9
    )
    assert lang == "ru"
    assert pred == "ru_label"
    assert chosen == "ru_lang"
    assert conf == 0.2

    pred, conf, chosen, lang = choose_prediction(
        "Read и ответь", "ru_label", 0.1, "en_label", 0.8
    )
    assert lang == "mixed"
    assert pred == "en_label"
    assert chosen == "en_conf"
