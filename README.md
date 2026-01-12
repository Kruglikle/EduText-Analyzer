# EduText-Analyzer
EduText Analyzer is a Python-based tool for automated analysis of school textbooks and educational texts in foreign languages. The project combines corpus linguistics, NLP, and machine learning to extract lexical, grammatical, and communicative features from texts.

## Textbook exercise classification
- Place OCR `.txt` files in `data/` or the repository root (see `enjoy-2.txt` as an example). Page separators like `----- PAGE 32 -----` or `=== Страница 32 ===` are detected automatically.
- Run classification:
  - `python scripts/classify_textbooks.py --input data/enjoy-2.txt --output outputs`
  - `python scripts/classify_textbooks.py --input data/textbooks_folder --output outputs --dump-blocks`
  - Optional: `--pipeline best_en_pipeline.pkl` to load a saved sklearn Pipeline instead of separate `--vectorizer tfidf_vectorizer.pkl` and `--model "logreg_model (1).pkl"`.
- Outputs:
  - `<name>_predictions.csv` with `exercise_id, page, section, prompt, label, confidence, review`.
  - `<name>_summary.csv` with totals by label and counts per page/section; `--dump-blocks` also writes `<name>_blocks.csv` with raw segments.
- `review=true` marks low-confidence predictions (default `< 0.60`); skim these prompts/blocks manually to confirm or correct the label.
- Segmentation now filters title/credits/contents blocks more aggressively and tightens exercise-start detection to avoid splitting song lyrics into standalone exercises.
