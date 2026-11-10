# Từ điển ánh xạ mã ICD-9 tiêu biểu sang Tên bệnh tiếng Việt
ICD9_DESCRIPTIONS = {
    "414.01": "Xơ vữa động mạch vành (Coronary atherosclerosis)",
    "250.00": "Đái tháo đường týp 2 không biến chứng",
    "401.9": "Tăng huyết áp vô căn (Essential hypertension)",
    "428.0": "Suy tim sung huyết (Congestive heart failure)",
    "585.9": "Bệnh thận mãn tính (Chronic kidney disease)",
    "272.4": "Tăng lipid máu hỗn hợp (Hyperlipidemia)",
    "530.81": "Trào ngược dạ dày thực quản (GERD)",
    "493.90": "Hen suyễn không xác định (Asthma)",
    "518.81": "Suy hô hấp cấp (Acute respiratory failure)",
    "571.5": "Xơ gan không do rượu (Cirrhosis of liver)"
}

def get_disease_name(code):
    """Lấy tên mô tả bệnh từ mã ICD-9."""
    code_str = str(code).strip()
    return ICD9_DESCRIPTIONS.get(code_str, f"Mã bệnh ICD-9 [{code_str}]")

def decode_temporal_weight(threshold):
    """
    Chuyển đổi ngưỡng bất đẳng thức Temporal Encoding EN(c, P)
    thành ngữ cảnh thời gian khám bệnh.
    """
    if threshold >= 0.5:
        return "xuất hiện ở lượt khám gần nhất (lượt t)"
    elif threshold >= 0.25:
        return "xuất hiện trong 2 lượt khám gần đây (lượt t hoặc t-1)"
    elif threshold >= 0.125:
        return "xuất hiện trong 3 lượt khám gần nhất"
    else:
        return "đã từng được ghi nhận trong lịch sử khám quá khứ"

def translate_rule_to_natural_language(feature_name, threshold, operator=">="):
    """
    Biến đổi một điều kiện tiền đề trong Cây quyết định thành câu tự nhiên.
    """
    disease_name = get_disease_name(feature_name)
    time_context = decode_temporal_weight(threshold)
    
    if operator in [">=", ">"]:
        return f"Bệnh nhân **đã có** chẩn đoán *'{disease_name}'* {time_context}"
    else:
        return f"Bệnh nhân **chưa từng/ít có** ghi nhận chẩn đoán *'{disease_name}'* {time_context}"

def generate_full_explanation_text(rules_list, predicted_ccs_name, confidence_score):
    """
    Tổng hợp toàn bộ tập luật thành báo cáo giải thích lâm sàng hoàn chỉnh.
    """
    explanation_sentences = []
    for feat, op, thresh in rules_list:
        explanation_sentences.append(translate_rule_to_natural_language(feat, thresh, op))
        
    rules_text = "\n  • " + "\n  • ".join(explanation_sentences)
    
    summary_text = (
        f"**Dự đoán của Doctor AI:** Rủi ro cao mắc **{predicted_ccs_name}** "
        f"ở lượt khám tiếp theo (Độ tin cậy: **{confidence_score * 100:.1f}%**).\n\n"
        f"**Cơ sở giải thích (Doctor XAI Rule):**\n"
        f"Dự đoán này được đưa ra dựa trên các yếu tố lịch sử bệnh lý sau:\n"
        f"{rules_text}"
    )
    return summary_text