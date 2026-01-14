from all_metrics import metrics


def test_textstat_metrics_cached(monkeypatch):
    metrics._clear_textstat_cache()
    calls = {"count": 0}

    def fake_metrics(text: str):
        calls["count"] += 1
        return {
            "flesch_reading_ease": 1.0,
            "flesch_kincaid_grade": 1.0,
            "words_total": 1,
            "sentences_total": 1,
            "syllables_total": 1,
            "avg_words_per_sentence": 1.0,
        }

    monkeypatch.setattr(metrics, "_compute_textstat_metrics_uncached", fake_metrics)

    first = metrics.compute_textstat_metrics("Hello world")
    second = metrics.compute_textstat_metrics("Hello world")

    assert calls["count"] == 1
    assert first == second
