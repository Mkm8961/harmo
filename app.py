import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

st.set_page_config(page_title="Smart Material Validator", layout="wide")
st.title("📦 Smart Material Description Validator (Merged with Roberta Fallback)")

# --- Upload CSV ---
st.sidebar.header("📤 Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.warning("Upload a file to continue.")
    st.stop()
df = pd.read_csv(uploaded_file)

# --- Valid material types (for rule-based logic) ---
valid_types = [
    "ADAPTER", "CONNECTOR", "BARRICADE", "BEND", "BUSHING", "CAP", "CATHODIC PROTECTION", "CLAMP",
    "COUPLING", "ELBOW", "FLANGE", "FLANGE GASKET", "GAS METER", "GAS METER AND METERSET PARTS",
    "GAUGE", "INSULATOR", "NIPPLE", "OPERATIONS ITEM", "OUTLET", "PLASTIC PIPE", "STEEL PIPE",
    "PLUG", "INSERT PROTECTOR", "REDUCER", "REGULATOR", "REGULATOR PARTS", "RELIEF VALVE",
    "SIGNAGE/LOCATE MATERIAL", "SLEEVE", "STIFFENER", "STOPPER", "STRAINER", "STUDS & SCREWS",
    "TEE", "TRANSITION", "UNION", "VALVE", "VALVE PARTS"
]

# --- Preprocess descriptions ---
df['UPPER_DESC'] = df['MATERIAL_NUMBER_TEXT'].astype(str).str.upper()
df['TYPE'] = df['UPPER_DESC'].str.split(',').str[0].str.strip()
filtered_df = df.copy()

# --- Load Roberta ---
@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")
    model = AutoModel.from_pretrained("sentence-transformers/all-roberta-large-v1")
    return tokenizer, model

tokenizer, roberta_model = load_roberta()

def get_roberta_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = roberta_model(**tokens)
    return outputs.last_hidden_state[:, 0].numpy()

# --- Rule-based validation function ---
def get_missing_parts(mat_type: str, parts: list[str]) -> list[str]:
    parts = [p.strip().upper() for p in parts]
    miss = []

    if mat_type not in valid_types:
        miss.append("TYPE NOT RECOGNISED")
        return miss

    if mat_type in {"STEEL PIPE", "PLASTIC PIPE"}:
        if not any(x in parts for x in ["COAT", "WRAP", "DUALCOAT", "PE", "FBE"]):
            miss.append("coating")
        if not any("W" in p for p in parts):
            miss.append("wall thickness")
        if not any(x in parts for x in ["X", "Y", "API"]):
            miss.append("grade")
        if not any(p.replace(".", "", 1).isdigit() for p in parts):
            miss.append("diameter")

    elif mat_type == "NIPPLE":
        if not any("X" in p for p in parts):
            miss.append("size")
        if "TOE" not in parts:
            miss.append("TOE")
        if not any(x in parts for x in ["BLK", "GALV", "ZINC", "SC"]):
            miss.append("material")

    elif mat_type == "VALVE":
        if not any(x in parts for x in ["BALL", "GATE", "CHECK", "PLUG"]):
            miss.append("valve type")
        if not any(x in parts for x in ["150A", "300A", "600A", "CL150", "CL300"]):
            miss.append("pressure rating")
        if not any(x in parts for x in ["FE RF", "FLANGED", "THRD", "SOCKET"]):
            miss.append("end type")
        if not any(x in parts for x in ["FP", "FULL PORT", "RP"]):
            miss.append("port type")

    elif mat_type == "FLANGE":
        if not any(x in parts for x in ["WN", "SLIPON", "THRD", "SOCKET", "LAP JOINT"]):
            miss.append("flange type")
        if not any(x in parts for x in ["RF", "FF", "RTJ"]):
            miss.append("face type")
        if not any(x in parts for x in ["150A", "300A", "600A", "CL150", "CL300", "CL600"]):
            miss.append("rating")
        if not any(x in parts for x in ["CS", "CARBON", "STL"]):
            miss.append("material")

    elif mat_type in {"ELBOW", "TEE", "BEND"}:
        if not any(x in parts for x in ["90", "45"]):
            miss.append("angle")
        if not any(x in parts for x in ["LR", "SR"]):
            miss.append("radius")
        if not any(x in parts for x in ["WELD", "THRD", "SOCKET"]):
            miss.append("connection type")

    elif mat_type == "STOPPER":
        if not any(x in parts for x in ["WELD", "THRD"]):
            miss.append("end type")
        if not any(p.replace('"', "").isdigit() for p in parts):
            miss.append("size")
        if not any(x in parts for x in ["#150", "CL150", "150A"]):
            miss.append("pressure rating")

    elif mat_type == "CAP":
        if not any(x in parts for x in ["THRD", "SOCKET", "WELD"]):
            miss.append("connection type")
        if not any(p.replace('"', "").isdigit() for p in parts):
            miss.append("size")

    elif mat_type == "COUPLING":
        if not any(x in parts for x in ["INSULATING", "COMP", "COMPRESSION", "TRANSITION"]):
            miss.append("type")
        if not any(p.replace('"', "").isdigit() for p in parts):
            miss.append("size")

    elif mat_type == "REDUCER":
        if not any("X" in p for p in parts):
            miss.append("size format")
        if not any(x in parts for x in ["ECC", "CONC"]):
            miss.append("concentric/eccentric type")

    elif mat_type == "UNION":
        if not any(x in parts for x in ["THRD", "SOCKET"]):
            miss.append("end type")
        if not any(p.replace('"', "").isdigit() for p in parts):
            miss.append("size")

    elif mat_type == "PLUG":
        if not any(x in parts for x in ["HEX", "ROUND"]):
            miss.append("head type")
        if not any(x in parts for x in ["THRD", "SOCKET"]):
            miss.append("end type")

    elif mat_type == "GAUGE":
        if not any(x in parts for x in ["PSI", "BAR", "MMWC"]):
            miss.append("unit")
        if not any(p.replace(".", "", 1).isdigit() for p in parts):
            miss.append("pressure range")

    elif mat_type == "CLAMP":
        if not any(x in parts for x in ["WELD", "THRD"]):
            miss.append("end type")
        if not any(p.replace('"', "").isdigit() for p in parts):
            miss.append("size")
        if not any(x in parts for x in ["#150", "CL150", "150A"]):
            miss.append("pressure rating")

    return miss

# --- Roberta fallback suggestion ---
def suggest_similar_roberta(input_text, type_):
    input_emb = get_roberta_embedding(input_text)
    candidates = filtered_df[filtered_df['TYPE'] == type_]['MATERIAL_NUMBER_TEXT'].tolist()
    if not candidates:
        return "No similar material found"
    candidate_embs = np.vstack([get_roberta_embedding(c) for c in candidates])
    scores = np.dot(candidate_embs, input_emb.T).squeeze()
    best_match = np.argmax(scores)
    return candidates[best_match]

# --- Suggestion Builder ---
def build_suggestion(desc, type_, missing_parts):
    if type_ in valid_types:
        candidates = filtered_df[filtered_df['TYPE'] == type_]['MATERIAL_NUMBER_TEXT'].tolist()
        parts = [p.strip().upper() for p in desc.split(",") if p.strip()]
        for c in candidates:
            cand_parts = [p.strip().upper() for p in c.split(",") if p.strip()]
            if all(any(k in cp for cp in cand_parts) for k in missing_parts):
                for token in cand_parts:
                    if token not in parts:
                        parts.append(token)
                return ", ".join(parts)
    return suggest_similar_roberta(desc, type_)

# --- Validation Loop ---
results = []
for _, row in filtered_df.iterrows():
    desc = row['MATERIAL_NUMBER_TEXT']
    parts = [p.strip().upper() for p in desc.split(",") if p.strip()]
    type_ = row['TYPE']
    missing = get_missing_parts(type_, parts)
    valid = len(missing) == 0
    suggestion = build_suggestion(desc, type_, missing) if not valid else ""
    results.append({
        "Material Number": row['MATERIAL_NUMBER'],
        "Description": desc,
        "Type": type_,
        "Status": "Valid" if valid else "Invalid",
        "Missing Parts": ", ".join(missing) if not valid else "",
        "Suggested Completion": suggestion
    })

results_df = pd.DataFrame(results)

# --- Setup session state for corrections ---
if "corrections" not in st.session_state:
    st.session_state["corrections"] = []

# --- UI Tabs ---
tabs = st.tabs(["✅ Valid", "❌ Invalid", "📘 Corrections", "📊 Summary"])

with tabs[0]:
    st.subheader("✅ Valid Material Descriptions")
    st.dataframe(results_df[results_df['Status'] == "Valid"])

with tabs[1]:
    st.subheader("❌ Invalid Descriptions")
    filter_type = st.selectbox("Filter by Material Type", sorted(results_df["Type"].unique()))
    inv_df = results_df[(results_df['Status'] == "Invalid") & (results_df['Type'] == filter_type)]
    for _, row in inv_df.iterrows():
        with st.expander(f"{row['Material Number']} - {row['Description']}"):
            st.markdown(f"**Missing Parts:** {row['Missing Parts']}")
            st.markdown(f"**Suggestion:** `{row['Suggested Completion']}`")
            corrected = st.text_input("✏️ Edit if needed", value=row['Suggested Completion'], key=row['Material Number'])
            if st.button("💾 Save Correction", key=f"btn_{row['Material Number']}"):
                st.session_state["corrections"].append({
                    "Material Number": row["Material Number"],
                    "Original": row["Description"],
                    "Corrected": corrected
                })
                st.success("Saved!")

with tabs[2]:
    st.subheader("📘 Saved Corrections")
    if st.session_state["corrections"]:
        st.dataframe(pd.DataFrame(st.session_state["corrections"]))
    else:
        st.info("No corrections saved yet.")

with tabs[3]:
    st.subheader("📊 Validation Summary")
    summary = results_df.groupby(["Type", "Status"]).size().unstack(fill_value=0)
    st.dataframe(summary)
