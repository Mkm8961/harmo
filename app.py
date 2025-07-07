import streamlit as st
import pandas as pd
from difflib import get_close_matches

st.set_page_config(page_title="Material Validator", layout="wide")
st.title("⚡ Smart Material Description Validator & Suggester")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your material CSV file", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Define valid material types
valid_types = ["PIPE", "NIPPLE", "VALVE", "BOLT", "STUD", "FLANGE", "STOPPER", "STOPPLE", "ELL", "ELBOW"]
df['UPPER_DESC'] = df['MATERIAL_NUMBER_TEXT'].astype(str).str.upper()
filtered_df = df[df['UPPER_DESC'].str.split(',').str[0].str.strip().isin(valid_types)]

# --- Validation logic ---
def validate_material(desc):
    parts = [p.strip().upper() for p in desc.split(",")]
    type_ = parts[0] if parts else ""

    if type_ == "PIPE":
        return (
            any("W" in p for p in parts) and
            any(x in parts for x in ["X", "Y", "API"]) and
            any(x in parts for x in ["COAT", "WRAP", "DUALCOAT"])
        )

    if type_ == "NIPPLE":
        return (
            any("X" in p for p in parts) and
            "TOE" in parts and
            any(x in parts for x in ["BLK", "GALV", "SC", "ZINC"])
        )

    if type_ == "FLANGE":
        return (
            any(x in parts for x in ["WN", "SLIPON", "THRD", "RF", "FF", "RTJ"]) and
            any("150" in p or "300" in p for p in parts) and
            any(x in parts for x in ["CS", "CARBON", "STL"])
        )

    if type_ in ["BOLT", "STUD"]:
        return (
            any("X" in p for p in parts) and
            any("STEEL" in p or "CARBON" in p for p in parts)
        )

    if type_ == "VALVE":
        return (
            any(x in parts for x in ["BALL", "GATE", "CHECK", "PLUG"]) and
            any(p.isdigit() or '"' in p for p in parts) and
            any(x in parts for x in ["150A", "300A", "600A"]) and
            any(x in parts for x in ["FE RF", "FLANGED", "THRD"]) and
            any(x in parts for x in ["FP", "FULL PORT"])
        )

    if type_ in ["ELL", "ELBOW"]:
        return (
            any(x in parts for x in ["45", "90"]) and
            any(x in parts for x in ["WELD", "THRD", "SOCKET"]) and
            any(x in parts for x in ["LR", "SR"])
        )

    if type_ in ["STOPPER", "STOPPLE"]:
        return (
            any("WELD" in p or "THRD" in p for p in parts) and
            any(p.replace('"', '').isdigit() for p in parts) and
            any("#" in p or "150A" in p or "ANSI" in p for p in parts)
        )

    return False

# --- Suggest closest ---
@st.cache_data
def get_suggestions_map(df):
    suggestions = {}
    for t in valid_types:
        descs = df[df['MATERIAL_NUMBER_TEXT'].str.upper().str.startswith(t)]['MATERIAL_NUMBER_TEXT'].tolist()
        suggestions[t] = descs
    return suggestions

suggestion_map = get_suggestions_map(df)

def get_suggestion(desc, type_):
    options = suggestion_map.get(type_, [])
    match = get_close_matches(desc.upper(), options, n=1, cutoff=0.3)
    return match[0] if match else "No suggestion"

# --- UI Dropdown for performance ---
selected = st.selectbox("🔽 Select a material to validate", filtered_df['MATERIAL_NUMBER_TEXT'])

status = "✅ Valid" if validate_material(selected) else "❌ Invalid"
type_ = selected.split(",")[0].strip().upper()
suggestion = get_suggestion(selected, type_)

st.markdown(f"**🔍 Type Detected:** `{type_}`")
st.markdown(f"**📊 Status:** {status}")
if status == "❌ Invalid":
    st.markdown(f"**💡 Suggested Completion:** `{suggestion}`")

# Allow edit and save
corrected = st.text_input("✏️ Edit the description if needed", value=selected)
if "corrections" not in st.session_state:
    st.session_state["corrections"] = []

if st.button("💾 Save Correction"):
    st.session_state["corrections"].append({
        "Original": selected,
        "Corrected": corrected
    })
    st.success("Correction saved!")

# Show all saved corrections
if st.session_state["corrections"]:
    st.subheader("📘 Saved Corrections")
    st.dataframe(pd.DataFrame(st.session_state["corrections"]))
