# TruthLens — Fake News Detection

A multi-file BS AI project using the ISOT Fake News dataset.

## 1. Project structure

```text
fake_news_detection_project/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
├── train_model.py
├── predictor.py
├── app.py
└── requirements.txt
```

## 2. Put the dataset in the correct place

Extract the downloaded Kaggle ZIP. Copy `Fake.csv` and `True.csv` into the `data` folder.

Do not rename the files.

## 3. Open in VS Code

Open the `fake_news_detection_project` folder in VS Code.

Open Terminal → New Terminal, then create and activate a virtual environment (recommended).

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 4. Install packages

```bash
python -m pip install -r requirements.txt
```

## 5. Train the model

```bash
python train_model.py
```

Training creates:
- `models/fake_news_model.joblib`
- `models/metrics.json`

It prints accuracy, precision, recall, F1-score, classification report, and confusion matrix for a held-out test split.

## 6. Launch the GUI

```bash
streamlit run app.py
```

A browser tab should open with the TruthLens interface.

## Important academic note

The ISOT dataset is a research dataset, not a complete ground-truth database for all news on the internet. Its labels and writing/source patterns may make the test score look better than real-world performance. This project classifies text patterns; it does not independently establish whether a claim is true. For a stronger final-year study, add out-of-dataset testing, reliable fact-checking sources, and careful error analysis.
