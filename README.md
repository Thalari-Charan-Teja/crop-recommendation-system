# Smart Agriculture Assistant (ML + Gemini)

A Streamlit crop recommendation application that combines an existing scikit-learn
model bundle with a Gemini-powered assistant for explanations and general farming
education.

## Project files

```text
smart_crop_github/
├── app.py
├── crop_bundle.joblib       # Add your own trained model bundle; not included here
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── assets/
    └── .gitkeep
```

## 1. Add your trained model

Copy your existing `crop_bundle.joblib` into the same folder as `app.py`.

The app expects a joblib dictionary with these keys:

- `model`
- `scaler`
- `nn`
- `threshold`
- `lo`
- `hi`
- `features`

The model must support `predict_proba`, and the scaler / nearest-neighbor
object must be compatible with the seven input features in this order:

1. Nitrogen (N)
2. Phosphorus (P)
3. Potassium (K)
4. Temperature (°C)
5. Humidity (%)
6. Soil pH
7. Rainfall (mm)

**Important:** The model file was not included in this starter package because it
was not present among the files available when this package was created. Add the
same `crop_bundle.joblib` that your original app uses.

## 2. Install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
```

## 3. Configure Gemini API key locally

Create `.streamlit/secrets.toml` (this real secrets file is ignored by Git):

```toml
GEMINI_API_KEY = "PASTE_YOUR_PRIVATE_API_KEY_HERE"
```

Get a key from Google AI Studio: https://aistudio.google.com/api-keys

Do not commit, share, or paste your real API key into source code or GitHub.

A placeholder example is provided as `.streamlit/secrets.toml.example`.
Rename/copy it to `secrets.toml` and replace the placeholder locally.

## 4. Run the app

```bash
streamlit run app.py
```

Without a Gemini key, the ML crop prediction can still load, but Gemini-powered
features will display a configuration message.

## 5. Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Ensure `crop_bundle.joblib` is included if it is below GitHub's ordinary
   100 MB per-file limit. For larger files, use Git LFS or external model storage.
3. Create a Streamlit Community Cloud app pointing to `app.py`.
4. In the app's **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "YOUR_PRIVATE_API_KEY"
```

Never upload `.streamlit/secrets.toml` to GitHub.

## Notes and limitations

- The ML model's confidence score is not a guaranteed probability of crop success.
- The LLM generates general educational information and can make mistakes.
- Fertilizer and irrigation decisions should be checked against a laboratory soil
  test and local agricultural guidance.
- Live weather integration is not included yet; the app uses the values entered
  by the user.
