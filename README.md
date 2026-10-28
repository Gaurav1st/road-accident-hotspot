# AI-Based Road Accident Hotspot Prediction — Demo

A runnable Streamlit mini-project prototype with an interactive hotspot map, risk prediction form, sample data table, and model feature importance.

## Important limitation
The app generates synthetic demonstration data. It is **not trained on official accident records**, and its scores must not be used for real-world road-safety decisions. Replace the demo data with a credible local dataset before making real-world claims.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish a public link (usually a few minutes after setup)
1. Create/sign in to GitHub and create a new public repository.
2. Upload `app.py` and `requirements.txt` from this folder.
3. Open https://share.streamlit.io/ and connect GitHub.
4. Choose the repository, branch `main`, and main file `app.py`, then click Deploy.
5. Share the `*.streamlit.app` URL once deployment finishes.
