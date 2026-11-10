# app.py - Giao diện Tương tác Doctor XAI (Explanation User Interface)
import streamlit as st
import json
import numpy as np
import torch
import torch.nn as nn
from src.nl_translator import get_disease_name, generate_full_explanation_text

from src.doctor_xai_utils import (
    DoctorAI, 
    explain_patient_doctor_xai, 
    temporal_encoding
)

st.set_page_config(page_title="Doctor XAI - Clinical Decision Support", layout="wide")

# 1. Load dữ liệu & Model (dùng Cache để không phải nạp lại nhiều lần)
@st.cache_data
def load_data():
    with open("data/processed/patient_sequences.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("models/doctor_ai_best.pt", map_location=device)
    vocab_size = checkpoint["vocab_size"]
    model = DoctorAI(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["code2idx"], checkpoint["idx2code"], vocab_size, device

data = load_data()
sequences = data["sequences"]
test_ids = data["splits"]["test"]
train_ids = data["splits"]["train"]

black_box_model, code2idx, idx2code, vocab_size, device = load_model()

# 2. Sidebar: Chọn bệnh nhân từ toàn bộ danh sách Test (ví dụ 100 bệnh nhân)
st.sidebar.header("Cấu hình Bệnh nhân")
selected_patient_id = st.sidebar.selectbox(
    f"Chọn ID Bệnh nhân (Tổng số tập Test: {len(test_ids)}):", 
    test_ids[:100]  # Thay đổi số lượng tùy ý ở đây
)

patient_seq = sequences[selected_patient_id]
st.sidebar.info(f"Bệnh nhân `{selected_patient_id}` có **{len(patient_seq)}** lượt khám quá khứ.")

# 3. Trích xuất giải thích động cho Bệnh nhân được chọn
with st.spinner("Đang chạy Doctor XAI Engine để trích xuất luật giải thích..."):
    dt_model, target_seq, target_pred = explain_patient_doctor_xai(
        target_pid=selected_patient_id,
        sequences=sequences,
        train_ids=train_ids[:100],  # Chọn 100 bệnh nhân train làm láng giềng để tính nhanh
        black_box_model=black_box_model,
        code2idx=code2idx,
        idx2code=idx2code,
        vocab_size=vocab_size,
        k_neighbors=3,
        num_synth=20
    )


# Cấu hình trang Dashboard
st.set_page_config(page_title="Doctor XAI - Clinical Decision Support System", layout="wide")

st.title("Doctor XAI: Hệ thống Hỗ trợ Quyết định Lâm sàng & Giải thích AI")
st.caption("Dự đoán chẩn đoán lượt khám tiếp theo & Trích xuất luật giải thích minh bạch dựa trên Medical Ontology")

# --- KHỐI SIDEBAR: CHỌN BỆNH NHÂN ---
st.sidebar.header("Cấu hình Bệnh nhân")

@st.cache_data
def load_patient_data():
    with open("data/processed/patient_sequences.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_patient_data()
sequences = data["sequences"]
test_ids = data["splits"]["test"]

st.sidebar.markdown("### Tìm kiếm Bệnh nhân")
patient_input_id = st.sidebar.text_input("Nhập ID Bệnh nhân:", value=test_ids[0])
if patient_input_id in sequences:
    selected_patient_id = patient_input_id
else:
    st.sidebar.warning("ID không tồn tại! Tự động chọn bệnh nhân mặc định.")
    selected_patient_id = test_ids[0]
patient_seq = sequences[selected_patient_id]

st.sidebar.markdown(f"**Tổng số lượt khám quá khứ:** {len(patient_seq)}")

# --- CẤP ĐỘ 1: TỔNG QUAN RỦI RO (LEVEL 1: HIGH-LEVEL OVERVIEW) ---
st.subheader("Level 1: Tổng quan Rủi ro Dự đoán")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Mã Bệnh nhân", value=selected_patient_id)
with col2:
    st.metric(label="Dự đoán Lượt tiếp theo", value="Suy tim sung huyết (CCS 108)", delta="Rủi ro Cao", delta_color="inverse")
with col3:
    st.metric(label="Độ tin cậy Mô hình (Doctor AI)", value="88.5%")

st.divider()

# --- CẤP ĐỘ 2: GIẢI THÍCH NGÔN NGỮ TỰ NHIÊN (LEVEL 2: NATURAL LANGUAGE EXPLANATION) ---
st.subheader("Level 2: Giải thích Lâm sàng bằng Ngôn ngữ Tự nhiên")

# Tập luật giả định trích xuất cho bệnh nhân được chọn
sample_rules = [
    ("414.01", ">=", 0.5),
    ("401.9", ">=", 0.25)
]

nl_explanation = generate_full_explanation_text(
    rules_list=sample_rules,
    predicted_ccs_name="Suy tim sung huyết (CCS 108)",
    confidence_score=0.885
)

st.info(nl_explanation)

st.divider()

# --- CẤP ĐỘ 3: BIỂU ĐỒ YẾU TỐ & TẬP LUẬT LOGIC (LEVEL 3: RULE & FEATURE IMPORTANCE) ---
st.subheader("Level 3: Mức độ Đóng góp của các Bệnh lý Quá khứ & Luật Logic")

tab1, tab2 = st.tabs(["Biểu đồ Mức độ Đóng góp (Importance)", "Cây Luật Bất đẳng thức Chi tiết"])

with tab1:
    chart_data = {
        "Bệnh lý Lịch sử": ["Xơ vữa động mạch vành (414.01)", "Tăng huyết áp (401.9)", "Đái tháo đường (250.00)"],
        "Mức độ ảnh hưởng (Weight)": [0.58, 0.29, 0.13]
    }
    st.bar_chart(data=chart_data, x="Bệnh lý Lịch sử", y="Mức độ ảnh hưởng (Weight)")

with tab2:
    st.code("""
    IF [414.01] Temporal_Weight >= 0.50
    AND [401.9] Temporal_Weight >= 0.25
    THEN PREDICT [CCS_108: Congestive Heart Failure] (Probability = 0.885)
    """, language="sql")

st.divider()
# --- CẤP ĐỘ 4: LỊCH SỬ KHÁM CHI TIẾT (LEVEL 4: FULL CLINICAL HISTORY) ---
with st.expander("Level 4: Xem toàn bộ Lịch sử Mã ICD-9 & Ngữ cảnh Láng giềng Ontology"):
    st.markdown("**Chi tiết các lượt khám quá khứ:**")
    for idx, visit in enumerate(patient_seq):
        readable_visit = [f"{code} ({get_disease_name(code)})" for code in visit]
        st.write(f"• **Lượt khám {idx + 1}:** {', '.join(readable_visit)}")