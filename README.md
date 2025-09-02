# AI-Powered Sleep Health Analyzer 😴

An interactive Streamlit app that analyzes lifestyle & sleep data to classify sleep quality (Good / Moderate / Poor), visualize patterns, and provide simple, personalized recommendations.

## Features
- Personal assessment form → instant prediction & tips
- Upload CSVs → batch scoring with downloadable results
- Built-in sample dataset for quick demo
- Basic EDA (distributions, label balance)
- RandomForest model trained on synthetic data

## Tech Stack
- Python, Streamlit, Pandas, NumPy
- scikit-learn (RandomForest)
- Matplotlib for charts

## Quickstart (Local)
1. Install Python 3.10+.
2. Create & activate a virtual environment.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
5. Open the URL shown in the terminal (usually http://localhost:8501).

## Data
- `sample_data.csv` is included for demo only.
- To use your own dataset, prepare a CSV with columns:
  `age, sleep_hours, caffeine_mg, screen_time_hours, exercise_minutes, stress_level, sleep_latency_minutes, awakenings`

## Deploy on Streamlit Community Cloud
1. Push this repo (already done if you can see this).
2. Go to https://share.streamlit.io
3. Select this repo, set `app.py` as the entry point, and deploy.

## Notes
- Replace synthetic data with your real data and consider validation before real-world use.
- This app is educational and not medical advice.
