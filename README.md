# Drug Type Prediction using Machine Learning

This repository implements the provided Mini Project: Drug Type Prediction using Machine Learning.

## Project structure

- `Drug_Type_Prediction_FIXED.ipynb` — corrected Task 1–11 implementation
- `app.py` — Streamlit prediction UI
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes credentials and generated Python files

## Required dataset

Place the professor's `drug200.csv` file in:

`/content/drive/MyDrive/ML Project/drug200.csv`

or in the notebook working directory.

## Key corrections from the previous implementation

1. Consistent 80/20 stratified split: 160 training / 40 testing.
2. No preprocessing fitted on the full dataset before the split.
3. SMOTENC is used for mixed numerical/categorical augmentation.
4. Exactly 1,000 augmented training samples are generated: 840 synthetic + 160 original training samples.
5. Target labels remain `DrugA`, `DrugB`, `DrugC`, `DrugX`, `DrugY`.
6. One complete preprocessing + model pipeline is saved.
7. Streamlit receives raw patient values and uses the same pipeline as training.
8. Probability labels use `model.classes_`, preventing class-order mistakes.
9. The model comparison table contains each model once; the best model is selected separately.
10. Tuning results are reported exactly as measured, including negative or zero improvement.

## Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The file `best_drug_prediction_pipeline.pkl` must be available in the project directory (or in the configured model folder).

## Security

Do not place ngrok tokens, API keys, passwords, or Google credentials in the notebook or GitHub repository.
The previous notebook contained an ngrok token; it should be revoked/rotated before public upload.

## Academic limitation

This is an educational machine-learning mini-project based on a small dataset. Model results are not clinical validation and should not be presented as a real-world medical prescribing system.
