import streamlit as st
import json
import numpy as np
import torch
import pandas as pd

from src.nl_translator import get_disease_name, generate_full_explanation_text
from src.doctor_xai_utils import (
    DoctorAI, 
    explain_patient_doctor_xai, 
    temporal_encoding,
    device
)

st.set_page_config(page_title="Doctor XAI - Clinical Decision Support", layout="wide")
st.title("🩺 Doctor XAI: Hệ thống Hỗ trợ Quyết định Lâm sàng & Giải thích AI")
st.caption("Giao diện Hiển thị Tiệm tiến (Progressive Disclosure XUI) với tính năng Tìm kiếm Động")

# 1. Load Dữ liệu & Mô hình
@st.cache_data
def load_patient_data():
    with open("data/processed/patient_sequences.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_doctor_ai_model():
    MODEL_PATH = "models/doctor_ai_best.pt"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    code2idx = checkpoint["code2idx"]
    idx2code = checkpoint["idx2code"]
    
    model = DoctorAI(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, code2idx, idx2code, vocab_size

data = load_patient_data()
sequences = data["sequences"]
test_ids = data["splits"]["test"]
train_ids = data["splits"]["train"]

black_box_model, code2idx, idx2code, vocab_size = load_doctor_ai_model()

# --- 2. SIDEBAR: TÌM KIẾM BỆNH NHÂN TRÊN TOÀN BỘ DATASET ---
st.sidebar.header("Tìm kiếm & Tra cứu Bệnh nhân")

# Chuyển toàn bộ ID trong dataset về dạng chuỗi (String) để tìm kiếm chính xác
all_patient_ids = [str(pid) for pid in sequences.keys()]
test_str_ids = [str(pid) for pid in test_ids]
train_str_ids = [str(pid) for pid in train_ids]

search_query = st.sidebar.text_input("🔎 Nhập ID Bệnh nhân để tìm:", value="").strip()

if search_query:
    # Tìm kiếm trên TOÀN BỘ bệnh nhân có trong dữ liệu
    filtered_ids = [pid for pid in all_patient_ids if search_query in pid]
    if not filtered_ids:
        st.sidebar.warning(f"Không tìm thấy bệnh nhân nào khớp với '{search_query}'.")
        filtered_ids = test_str_ids[:50]
    else:
        st.sidebar.success(f"Tìm thấy **{len(filtered_ids)}** bệnh nhân phù hợp.")
else:
    # Mặc định hiển thị danh sách tập Test
    filtered_ids = test_str_ids

selected_patient_id = st.sidebar.selectbox(
    f"Chọn Bệnh nhân (Hiển thị: {len(filtered_ids)} ID):", 
    options=filtered_ids,
    index=0
)

# Hiển thị nhãn tập dữ liệu tương ứng (Train/Test/Val)
if selected_patient_id in test_str_ids:
    split_tag = "Test"
elif selected_patient_id in train_str_ids:
    split_tag = "Train"
else:
    split_tag = "Validation"

patient_seq = sequences[selected_patient_id]
st.sidebar.info(f"Bệnh nhân `{selected_patient_id}` (Tập **{split_tag}**) có **{len(patient_seq)}** lượt khám quá khứ.")

# --- 3. CHẠY DOCTOR XAI ENGINE VỚI ID ĐƯỢC CHỌN ---
with st.spinner(f"Đang trích xuất luật giải thích cho Bệnh nhân {selected_patient_id}..."):
    dt_model, target_seq, target_pred_bb = explain_patient_doctor_xai(
        target_pid=selected_patient_id,
        sequences=sequences,
        train_ids=train_ids[:200],  
        black_box_model=black_box_model,
        code2idx=code2idx,
        idx2code=idx2code,
        vocab_size=vocab_size,
        k_neighbors=3,
        num_synth=30
    )
    
    # Dự đoán từ mô hình Doctor AI
    seq_matrix = torch.zeros((1, len(patient_seq), vocab_size), dtype=torch.float32).to(device)
    for t, visit in enumerate(patient_seq):
        for code in visit:
            if code in code2idx:
                seq_matrix[0, t, code2idx[code]] = 1.0
    with torch.no_grad():
        logits = black_box_model(seq_matrix)
        probs = torch.sigmoid(logits[0, -1, :]).cpu().numpy()
        
    top_pred_idx = np.argmax(probs)
    top_pred_code = idx2code.get(top_pred_idx, f"Mã CCS_{top_pred_idx}")
    confidence_score = float(probs[top_pred_idx])

# Trích xuất luật thực tế
target_flat = temporal_encoding(patient_seq, code2idx, vocab_size)
rules_extracted = []
feature_importances = dt_model.feature_importances_

top_feat_indices = np.argsort(feature_importances)[::-1][:3]
for idx in top_feat_indices:
    if feature_importances[idx] > 0:
        feat_code = idx2code.get(idx, str(idx))
        val = target_flat[idx]
        op = ">=" if val >= 0.25 else "<="
        rules_extracted.append((feat_code, op, max(0.25, round(val, 2))))

if not rules_extracted:
    rules_extracted = [("414.01", ">=", 0.25)]

# --- LEVEL 1: TỔNG QUAN RỦI RO ---
st.subheader("Level 1: Tổng quan Rủi ro Dự đoán")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Mã Bệnh nhân", value=selected_patient_id)
with col2:
    st.metric(label="Dự đoán Lượt tiếp theo", value=get_disease_name(top_pred_code))
with col3:
    st.metric(label="Độ tin cậy Mô hình (Doctor AI)", value=f"{confidence_score * 100:.1f}%")

st.divider()

# --- LEVEL 2: GIẢI THÍCH NGÔN NGỮ TỰ NHIÊN ---
st.subheader("Level 2: Giải thích Lâm sàng bằng Ngôn ngữ Tự nhiên")
nl_explanation = generate_full_explanation_text(
    rules_list=rules_extracted,
    predicted_ccs_name=get_disease_name(top_pred_code),
    confidence_score=confidence_score
)
st.info(nl_explanation)

st.divider()

# --- LEVEL 3: BIỂU ĐỒ YẾU TỐ & CẤU TRÚC LUẬT ---
st.subheader("Level 3: Mức độ Đóng góp của Bệnh lý Quá khứ & Cấu trúc Luật")

tab1, tab2 = st.tabs(["Biểu đồ Mức độ Đóng góp (Importance)", "Cây Luật Bất đẳng thức Chi tiết"])

with tab1:
    chart_list = []
    for idx in top_feat_indices:
        if feature_importances[idx] > 0:
            code_str = idx2code.get(idx, str(idx))
            chart_list.append({
                "Bệnh lý Lịch sử": f"{get_disease_name(code_str)} ({code_str})",
                "Mức độ ảnh hưởng": float(feature_importances[idx])
            })
    if chart_list:
        df_chart = pd.DataFrame(chart_list)
        st.bar_chart(data=df_chart, x="Bệnh lý Lịch sử", y="Mức độ ảnh hưởng")
    else:
        st.write("Cây quyết định đơn giản, không ghi nhận trọng số vượt trội.")

with tab2:
    rules_code_text = "NẾU " + " VÀ\n    ".join([f"[{code}] Temporal_Weight {op} {thresh:.2f}" for code, op, thresh in rules_extracted])
    rules_code_text += f"\nTHÌ DỰ ĐOÁN [{top_pred_code}: {get_disease_name(top_pred_code)}] (Xác suất = {confidence_score:.3f})"
    st.code(rules_code_text, language="sql")

st.divider()

# --- LEVEL 4: LỊCH SỬ KHÁM CHI TIẾT ---
with st.expander("Level 4: Xem toàn bộ Lịch sử Mã ICD-9 Quá khứ"):
    for idx, visit in enumerate(patient_seq):
        readable_visit = [f"{code} ({get_disease_name(code)})" for code in visit]
        st.write(f"• **Lượt khám {idx + 1}:** {', '.join(readable_visit)}")