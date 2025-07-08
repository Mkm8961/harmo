import streamlit as st
import pandas as pd
from rapidfuzz import process

st.set_page_config(page_title="Material Validator", layout="wide")
st.title("Material Validator & Suggester")

# --- Inject custom CSS for beautification ---
st.markdown("""
    <style>
        .material-box {
            border: 1px solid #CCC;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 15px;
            background-color: #F9F9F9;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        }
        .status-valid {
            color: green;
            font-weight: bold;
        }
        .status-invalid {
            color: red;
            font-weight: bold;
        }
        .header-section {
            background-color: #e8f4fa;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 600;
            border-left: 4px solid #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

# 📤 Upload CSV
st.markdown('<div class="header-section">📤 Upload Your Material File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your material CSV file", type="csv")
if not uploaded_file:
    st.stop()
df = pd.read_csv(uploaded_file)

# 🔧 Setup
valid_types = ["PIPE", "NIPPLE", "VALVE", "BOLT", "STUD", "FLANGE", "STOPPER", "STOPPLE", "ELL", "ELBOW"]
df["UPPER_DESC"] = df["MATERIAL_NUMBER_TEXT"].astype(str).str.upper()
filtered_df = df[df["UPPER_DESC"].str.split(",").str[0].str.strip().isin(valid_types)].copy()
filtered_df["Type"] = filtered_df["UPPER_DESC"].str.split(",").str[0].str.strip()

@st.cache_data
def get_suggestions_map(df):
    suggestions = {}
    for t in valid_types:
        descs = df[df["MATERIAL_NUMBER_TEXT"].str.upper().str.startswith(t)]["MATERIAL_NUMBER_TEXT"].tolist()
        suggestions[t] = descs
    return suggestions

suggestion_map = get_suggestions_map(df)

def validate_with_reason(desc):
    parts = [p.strip().upper() for p in desc.split(",")]
    type_ = parts[0] if parts else ""

    def match_any(keywords):
        return any(any(k in p for k in keywords) for p in parts)

    missing = []

    if type_ == "PIPE":
        if not match_any(["W", "WT"]): missing.append("wall thickness")
        if not match_any(["X", "Y", "API"]): missing.append("grade")
        if not match_any(["COAT", "WRAP", "DUALCOAT", "PEB", "ERW", "FBE"]): missing.append("coating")
        return (len(missing) == 0), missing

    if type_ == "NIPPLE":
        if not match_any(["X"]): missing.append("size")
        if "TOE" not in parts: missing.append("TOE")
        if not match_any(["BLK", "GALV", "ZINC", "SC"]): missing.append("material")
        return (len(missing) == 0), missing

    if type_ == "FLANGE":
        if not match_any(["WN", "SLIPON", "THRD", "RF", "FF", "RTJ"]): missing.append("flange type")
        if not match_any(["150", "300"]): missing.append("pressure rating")
        if not match_any(["CS", "CARBON", "STL"]): missing.append("material")
        return (len(missing) == 0), missing

    if type_ in ["BOLT", "STUD"]:
        if not match_any(["X"]): missing.append("size")
        if not match_any(["STEEL", "CARBON"]): missing.append("material")
        return (len(missing) == 0), missing

    if type_ == "VALVE":
        if not match_any(["BALL", "GATE", "CHECK", "PLUG"]): missing.append("valve type")
        if not match_any(["150A", "300A", "600A"]): missing.append("pressure rating")
        if not match_any(["FE RF", "FLANGED", "THRD"]): missing.append("end type")
        if not match_any(["FP", "FULL PORT"]): missing.append("port type")
        return (len(missing) == 0), missing

    if type_ in ["ELL", "ELBOW"]:
        if not match_any(["45", "90"]): missing.append("angle")
        if not match_any(["WELD", "THRD", "SOCKET"]): missing.append("connection type")
        if not match_any(["LR", "SR"]): missing.append("radius")
        return (len(missing) == 0), missing

    if type_ in ["STOPPER", "STOPPLE"]:
        if not match_any(["WELD", "THRD"]): missing.append("weld/thread")
        if not any(p.replace('"', '').isdigit() for p in parts): missing.append("size")
        if not match_any(["#", "150A", "ANSI", "CL600"]): missing.append("pressure rating")
        return (len(missing) == 0), missing

    return False, ["unknown or unsupported type"]

def build_full_suggestion(desc, type_, missing_parts):
    candidates = suggestion_map.get(type_, [])
    keyword_map = {
        "wall thickness": ["W", "WT"],
        "grade": ["X", "Y", "API"],
        "coating": ["COAT", "WRAP", "DUALCOAT", "PEB", "ERW", "FBE"],
        "size": ["X", '"', "INCH", "MM"],
        "TOE": ["TOE"],
        "material": ["BLK", "GALV", "ZINC", "SC", "STEEL", "CARBON", "CS", "STL"],
        "flange type": ["WN", "SLIPON", "THRD", "RF", "FF", "RTJ"],
        "pressure rating": ["150", "150A", "300", "300A", "600A", "#", "ANSI", "CL600"],
        "valve type": ["BALL", "GATE", "CHECK", "PLUG"],
        "end type": ["FE RF", "FLANGED", "THRD"],
        "port type": ["FP", "FULL PORT"],
        "angle": ["45", "90"],
        "connection type": ["WELD", "THRD", "SOCKET"],
        "radius": ["LR", "SR"],
        "weld/thread": ["WELD", "THRD"]
    }

    original_parts = [p.strip().upper() for p in desc.split(",") if p.strip()]
    original_set = set(original_parts)

    miss_keywords = []
    for m in missing_parts:
        miss_keywords += keyword_map.get(m.lower(), [])

    for c in candidates:
        candidate_parts = [p.strip().upper() for p in c.split(",")]
        if all(any(k in p for p in candidate_parts) for k in miss_keywords):
            for token in candidate_parts:
                if token not in original_set:
                    original_parts.append(token)
            return ", ".join(original_parts)

    return "No complete match found"

@st.cache_data
def classify_status(df):
    df = df.copy()
    df["Validation_Status"] = df["MATERIAL_NUMBER_TEXT"].apply(
        lambda x: "Valid" if validate_with_reason(x)[0] else "Invalid"
    )
    return df

# --- Explore & Validate with Filter ---
st.markdown('<div class="header-section">🔍 Explore & Validate</div>', unsafe_allow_html=True)
validation_filter = st.radio("Filter materials by status", ["All", "Valid", "Invalid"], horizontal=True)

filtered_status_df = classify_status(filtered_df)

# Combine material number and description
filtered_status_df["Display"] = (
    filtered_status_df["MATERIAL_NUMBER"].astype(str) +
    " - " +
    filtered_status_df["MATERIAL_NUMBER_TEXT"]
)

if validation_filter != "All":
    filtered_status_df = filtered_status_df[filtered_status_df["Validation_Status"] == validation_filter]

# Dropdown and mapping
selected_display = st.selectbox("Select a material description", filtered_status_df["Display"])
selected_row = filtered_status_df[filtered_status_df["Display"] == selected_display].iloc[0]
selected = selected_row["MATERIAL_NUMBER_TEXT"]
material_number = selected_row["MATERIAL_NUMBER"]
sel_type = selected.split(",")[0].strip().upper()

valid, missing_parts = validate_with_reason(selected)
suggestion = build_full_suggestion(selected, sel_type, missing_parts)

with st.container():
    st.markdown("#### ✨ Validation Result")
    st.markdown(f"**Material Number:** `{material_number}`")
    st.markdown(f"**Selected Material:** `{selected}`")

    if valid:
        st.markdown('<span class="status-valid">✅ Valid - Passes all rules</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-invalid">❌ Invalid</span>', unsafe_allow_html=True)
        st.markdown(f"**Missing Parts:** {', '.join(missing_parts)}")

        if suggestion != "No complete match found":
            st.markdown(f"🔧 **Suggested Full Description:** `{suggestion}`")
        else:
            st.warning("⚠️ No full suggestion found in historical data.")

corrected = st.text_input("✏️ Edit if needed", value=suggestion if suggestion != "No complete match found" else selected)
if "corrections" not in st.session_state:
    st.session_state["corrections"] = []

if st.button("💾 Save Correction"):
    st.session_state["corrections"].append({"Material Number": material_number, "Original": selected, "Corrected": corrected})
    st.success("✅ Correction saved!")

if st.session_state["corrections"]:
    st.markdown('<div class="header-section">📘 Saved Corrections</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state["corrections"]))

if st.checkbox("📊 Show validation summary"):
    @st.cache_data
    def get_summary(df):
        summary_df = df.copy()
        summary_df["Status"] = summary_df["MATERIAL_NUMBER_TEXT"].apply(
            lambda x: "Valid" if validate_with_reason(x)[0] else "Invalid"
        )
        return summary_df.groupby(["Type", "Status"]).size().unstack(fill_value=0)
    st.dataframe(get_summary(filtered_df))
