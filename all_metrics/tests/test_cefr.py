from pathlib import Path

from all_metrics.cefr import load_cefr_lexicon


def test_load_cefr_lexicon_dedup(tmp_path: Path):
    csv_content = "\n".join(
        [
            "Word,CEFR Level",
            "apple,A2",
            "apple,B1",
            "banana,C1",
            "banana,A1",
            "cat,B2",
        ]
    )
    path = tmp_path / "cefr.csv"
    path.write_text(csv_content, encoding="utf-8")

    levels = load_cefr_lexicon(str(path))
    assert levels["apple"] == "A2"
    assert levels["banana"] == "A1"
    assert levels["cat"] == "B2"
