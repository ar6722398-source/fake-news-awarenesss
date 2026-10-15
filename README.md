# Fake News Awareness & Detection System

A student mini-project implementing the proposal with Python, Streamlit, NLP via TF-IDF, Logistic Regression, and SQLite.

## Dataset
Place a labelled CSV at `data/news.csv` with:
- `text`: news headline/article text
- `label`: `REAL` or `FAKE`

The bundled `data/news.csv` is a tiny synthetic demo dataset for testing the workflow only. It is not suitable for judging real news. For meaningful experiments, use a legitimately obtained labelled dataset and review its licence, provenance, class balance, duplicates and label quality.

## Quick start (Windows)
From the project folder, run:
```powershell
.\run.bat
```
The launcher creates `.venv` if needed, installs/checks dependencies, trains the model from `data/news.csv`, and starts the Streamlit app. Open the local URL shown in the terminal. Press `Ctrl+C` in that terminal to stop the app.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train_model.py --data data/news.csv
```
This writes `fake_news_pipeline.joblib` and `model_metrics.txt`.

## Run
```bash
streamlit run app.py
```

## Features
- Text input and REAL/POTENTIALLY FAKE prediction
- TF-IDF + Logistic Regression pipeline
- Probability-based model confidence where available
- SQLite prediction history
- Awareness checklist
- Evaluation report from a held-out test split

## Important limitation
This is an educational text-classification prototype, not a professional fact-checker. Dataset bias, topic drift, satire, copied reporting, short text and unfamiliar events can all produce misleading predictions. Verify important claims against primary evidence and reputable sources.
