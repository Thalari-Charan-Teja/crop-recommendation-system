import numpy as np
import joblib
import streamlit as st
from google import genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Agriculture Assistant",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Agriculture Assistant")
st.caption("Machine-learning crop recommendations with a Gemini AI farming assistant.")

# ---------------- LOAD EXISTING ML MODEL ----------------
@st.cache_resource
def load_bundle():
    return joblib.load("crop_bundle.joblib")

try:
    b = load_bundle()
    model = b["model"]
    scaler = b["scaler"]
    nn = b["nn"]
    threshold = b["threshold"]
    lo = b["lo"]
    hi = b["hi"]
except Exception as e:
    st.error("Could not load crop_bundle.joblib. Keep it in the same folder as app.py.")
    st.exception(e)
    st.stop()


def check(raw):
    """Return status and top 3 predictions from the existing ML model."""
    raw_array = np.array([raw], dtype=float)
    out_range = ((raw_array < lo) | (raw_array > hi)).any()
    scaled = scaler.transform(raw_array)
    odd_combo = nn.kneighbors(scaled, n_neighbors=5)[0].mean() > threshold
    probabilities = model.predict_proba(scaled)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    top3 = [(str(model.classes_[i]), float(probabilities[i])) for i in top_indices]

    if out_range or odd_combo:
        return "unusual", top3
    if top3[0][1] < 0.60:
        return "uncertain", top3
    return "ok", top3


# ---------------- GEMINI CLIENT ----------------
def get_gemini_client():
    """Read the API key from Streamlit secrets; never hard-code it."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def ask_gemini(prompt):
    """Send a prompt to Gemini and return its text response."""
    client = get_gemini_client()
    if client is None:
        return (
            "Gemini API key not found. Create "
            ".streamlit/secrets.toml and add GEMINI_API_KEY = "
            '"YOUR_API_KEY" (do not share your key).'
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text or "The AI did not return text. Please try again."
    except Exception as e:
        return f"Gemini request failed: {e}"


# ---------------- SIDEBAR ----------------
st.sidebar.header("🌾 Project information")
st.sidebar.write(
    "The ML model predicts crop possibilities. Gemini explains the results "
    "and answers general farming questions."
)
language = st.sidebar.selectbox("AI response language", ["English", "Telugu"])
st.sidebar.info(
    "AI guidance is general information, not a substitute for a laboratory "
    "soil test or advice from a local agricultural officer."
)

# ---------------- INPUTS ----------------
st.subheader("1. Enter soil and weather values")
left, right = st.columns(2)

with left:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=42.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=43.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=21.0)

with right:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=6000.0, value=200.0)

if st.button("🌱 Recommend crops", type="primary"):
    status, top3 = check([N, P, K, temperature, humidity, ph, rainfall])
    st.session_state["crop_result"] = {
        "status": status,
        "top3": top3,
        "inputs": {
            "Nitrogen": N,
            "Phosphorus": P,
            "Potassium": K,
            "Temperature_C": temperature,
            "Humidity_percent": humidity,
            "Soil_pH": ph,
            "Rainfall_mm": rainfall
        }
    }
    # Reset old AI answers when a new prediction is made.
    st.session_state.pop("ai_report", None)
    st.session_state["chat_history"] = []

# ---------------- RESULTS ----------------
result = st.session_state.get("crop_result")

if result:
    st.subheader("2. Machine-learning prediction")
    status = result["status"]
    top3 = result["top3"]

    if status == "unusual":
        st.error(
            "Unusual input: these values or their combination differ from "
            "the training data. Treat the predictions as unreliable."
        )
    elif status == "uncertain":
        st.warning(
            "The model confidence is limited. Treat these crops as possibilities, "
            "not a confirmed recommendation."
        )
    else:
        st.success(
            f"Top prediction: {top3[0][0]} "
            f"({top3[0][1] * 100:.1f}% model score)"
        )

    st.write("Top 3 model predictions")
    for crop, probability in top3:
        st.write(f"**{crop}** — {probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    st.subheader("3. AI agriculture tools")
    tab1, tab2, tab3, tab4 = st.tabs([
        "AI crop advisor",
        "Fertilizer guidance",
        "Water management",
        "Compare crops"
    ])

    input_summary = result["inputs"]
    crop_summary = ", ".join(
        [f"{crop} ({prob * 100:.1f}%)" for crop, prob in top3]
    )
    lang_instruction = (
        "Respond in simple Telugu." if language == "Telugu"
        else "Respond in clear, simple English."
    )
    safety_note = (
        "Do not invent exact fertilizer doses, yield guarantees, prices, or "
        "local conditions. Explain when a soil test or local agricultural "
        "expert is needed. Clearly separate known input values from assumptions."
    )

    with tab1:
        st.write("Ask Gemini to explain the model result using the actual inputs.")
        if st.button("Generate AI crop explanation"):
            prompt = f"""
You are a cautious agricultural information assistant.
{lang_instruction}
Soil and weather inputs supplied by the user: {input_summary}
Existing machine-learning model top predictions: {crop_summary}
Model status: {status}

Explain:
1. What the model predicted.
2. What the supplied soil/weather values mean in general.
3. Why these values may be relevant to crop suitability, without claiming
   that you know the model's internal reasoning.
4. What the farmer should verify locally before planting.

{ safety_note }
This is an educational recommendation, not a guarantee.
"""
            with st.spinner("Gemini is preparing the crop explanation..."):
                st.session_state["ai_report"] = ask_gemini(prompt)

        if st.session_state.get("ai_report"):
            st.markdown(st.session_state["ai_report"])

    with tab2:
        st.write("Get general nutrient-management information based on the entered values.")
        if st.button("Generate fertilizer guidance"):
            prompt = f"""
You are a cautious soil and crop information assistant.
{lang_instruction}
Entered soil values: {input_summary}
ML model top crop predictions: {crop_summary}

Give general educational guidance:
- Explain what N, P, K and pH represent.
- Explain why a soil test is needed before choosing fertilizer type and quantity.
- Suggest questions the farmer should ask a local agricultural officer.
- Do not prescribe exact fertilizer quantities or claim a deficiency from these
  numbers alone. Units, soil test method, crop stage and local conditions matter.

{ safety_note }
"""
            with st.spinner("Preparing general fertilizer guidance..."):
                st.markdown(ask_gemini(prompt))

    with tab3:
        st.write("Get general water-management considerations from the supplied values.")
        if st.button("Generate water-management guidance"):
            prompt = f"""
You are a cautious agricultural water-management information assistant.
{lang_instruction}
Entered values: {input_summary}
Top ML crop predictions: {crop_summary}

Explain general considerations for irrigation and water management for these
possible crops. Do not invent a local weather forecast, irrigation schedule,
or exact litres/mm requirement. Explain that soil type, crop stage, field
conditions and local forecast are also required. Recommend checking local
agricultural guidance.
"""
            with st.spinner("Preparing water-management guidance..."):
                st.markdown(ask_gemini(prompt))

    with tab4:
        st.write("Compare the top three crops predicted by the ML model.")
        if st.button("Compare top 3 crops with AI"):
            prompt = f"""
You are a neutral agricultural information assistant.
{lang_instruction}
The model returned these possible crops: {crop_summary}
Entered values: {input_summary}

Compare these crops at a high level in a table with:
- general growing conditions,
- water considerations,
- soil considerations,
- what local information must be checked.

Do not invent precise local suitability, profits, yields, market prices, or
guarantees. Explain that the model scores are not probabilities of success.
{ safety_note }
"""
            with st.spinner("Comparing crops..."):
                st.markdown(ask_gemini(prompt))

    # ---------------- FARMING CHATBOT ----------------
    st.divider()
    st.subheader("4. Ask the AI farming assistant")
    st.caption(
        "Ask general questions about crops, soil, irrigation or farming. "
        "The assistant can use your latest input values and model predictions."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Example: What does soil pH mean?")
    if question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        history_text = "\n".join(
            f'{m["role"]}: {m["content"]}'
            for m in st.session_state["chat_history"][-8:]
        )
        prompt = f"""
You are a safe, practical agriculture education assistant.
{lang_instruction}
Latest user soil/weather values: {input_summary}
Latest ML model predictions: {crop_summary}
Recent conversation:
{history_text}

Answer the latest question clearly and simply. Do not claim to be a certified
agronomist. Do not invent live weather, local prices, soil test results,
fertilizer doses or yield guarantees. If important details are missing, say
what should be checked with a local agricultural expert.
"""
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_gemini(prompt)
                st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Enter the soil and weather values, then click **Recommend crops** to unlock the AI tools.")

st.divider()
st.caption(
    "Note: ML confidence scores are model outputs, not guaranteed chances of crop success. "
    "AI-generated guidance can be wrong; verify important decisions with soil testing "
    "and local agricultural experts."
)
