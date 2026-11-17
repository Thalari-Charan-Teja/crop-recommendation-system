# app.py  -- the web page. Run with:  streamlit run app.py
import numpy as np, joblib, streamlit as st

b = joblib.load("crop_bundle.joblib")
model, scaler, nn = b["model"], b["scaler"], b["nn"]
threshold, lo, hi, FEATURES = b["threshold"], b["lo"], b["hi"], b["features"]

def check(raw):
    """Return (status, top3). status = 'unusual' | 'uncertain' | 'ok'"""
    raw = np.array([raw], dtype=float)
    out_range = ((raw < lo) | (raw > hi)).any()
    scaled = scaler.transform(raw)
    odd_combo = nn.kneighbors(scaled, n_neighbors=5)[0].mean() > threshold
    proba = model.predict_proba(scaled)[0]
    top = np.argsort(proba)[::-1][:3]
    top3 = [(model.classes_[i], proba[i]) for i in top]
    if out_range or odd_combo:
        return "unusual", top3
    return ("ok" if top3[0][1] >= 0.60 else "uncertain"), top3

st.title("Crop Recommendation with Confidence Check")
st.write("Enter soil and weather values. The app also warns you when the input looks unusual.")

c1, c2 = st.columns(2)
N  = c1.number_input("Nitrogen (N)", 0.0, 300.0, 90.0)
P  = c1.number_input("Phosphorus (P)", 0.0, 300.0, 42.0)
K  = c1.number_input("Potassium (K)", 0.0, 300.0, 43.0)
T  = c1.number_input("Temperature (°C)", 0.0, 60.0, 21.0)
H  = c2.number_input("Humidity (%)", 0.0, 100.0, 82.0)
PH = c2.number_input("pH", 0.0, 14.0, 6.5)
R  = c2.number_input("Rainfall (mm)", 0.0, 6000.0, 200.0)

if st.button("Recommend crop"):
    status, top3 = check([N, P, K, T, H, PH, R])
    if status == "unusual":
        st.error("⚠️ Unusual input: it does not look like the data this model learned from. No reliable recommendation.")
    elif status == "uncertain":
        st.warning("🤔 The model is not sure. Treat these as possibilities, not an answer.")
    else:
        st.success(f"Recommended crop: **{top3[0][0]}** ({top3[0][1]*100:.0f}% confident)")
    st.subheader("Top 3 possible crops")
    for crop, p in top3:
        st.write(f"{crop}: {p*100:.1f}%")
        st.progress(float(p))
