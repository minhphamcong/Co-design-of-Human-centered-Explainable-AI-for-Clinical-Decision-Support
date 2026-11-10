ICD9_DESCRIPTIONS = {
    # Tim mạch & Mạch máu
    "414.01": "Xơ vữa động mạch vành (Coronary atherosclerosis)",
    "414.00": "Bệnh tim thiếu máu cục bộ mãn tính",
    "401.9": "Tăng huyết áp vô căn (Essential hypertension)",
    "428.0": "Suy tim sung huyết (Congestive heart failure)",
    "428.22": "Suy tim tâm thu mãn tính",
    "428.32": "Suy tim tâm trương mãn tính",
    "427.31": "Rung nhĩ (Atrial fibrillation)",
    "410.71": "Nhồi máu cơ tim cấp vùng dưới nội mạc (NSTEMI)",
    "412": "Nhồi máu cơ tim cũ (Old MI)",
    "424.0": "Rối loạn van hai lá (Mitral valve disorders)",
    "434.91": "Tai biến mạch máu não / Đột quỵ thiếu máu cục bộ",
    
    # Thận & Tiết niệu
    "584.9": "Suy thận cấp không xác định (Acute kidney failure)",
    "585.9": "Bệnh thận mãn tính (Chronic kidney disease)",
    "599.0": "Nhiễm trùng đường tiết niệu (UTI)",
    
    # Nội tiết & Chuyển hóa
    "250.00": "Đái tháo đường týp 2 không biến chứng",
    "250.02": "Đái tháo đường týp 2 kiểm soát kém",
    "272.4": "Tăng lipid máu hỗn hợp (Hyperlipidemia)",
    "272.0": "Tăng cholesterol máu",
    "276.1": "Giảm natri máu (Hyponatremia)",
    "276.2": "Nhiễm toan chuyển hóa (Acidosis)",
    
    # Hô hấp & Tiêu hóa
    "518.81": "Suy hô hấp cấp (Acute respiratory failure)",
    "511.9": "Tràn dịch màng phổi (Pleural effusion)",
    "486": "Viêm phổi không xác định (Pneumonia)",
    "493.90": "Hen suyễn không xác định (Asthma)",
    "496": "Bệnh phổi tắc nghẽn mãn tính (COPD)",
    "530.81": "Trào ngược dạ dày thực quản (GERD)",
    "571.5": "Xơ gan (Cirrhosis of liver)",
    
    # Nhiễm trùng & Khác
    "995.92": "Nhiễm trùng huyết nặng (Severe sepsis)",
    "038.9": "Nhiễm trùng máu không xác định (Septicemia)",
    "780.39": "Co giật (Seizures)",
    "V45.81": "Tiền sử phẫu thuật bắc cầu động mạch vành (CABG)",
    "V45.82": "Tiền sử can thiệp động mạch vành qua da (PTCA)",
    "V58.61": "Tiền sử dùng thuốc chống đông máu lâu dài",
    
    # Danh mục Tiền tố (Prefix Grouping)
    "414": "Bệnh tim thiếu máu cục bộ / Xơ vữa vành",
    "401": "Tăng huyết áp",
    "428": "Suy tim",
    "427": "Rối loạn nhịp tim",
    "584": "Suy thận cấp",
    "585": "Bệnh thận mãn tính",
    "250": "Đái tháo đường",
    "272": "Rối loạn chuyển hóa Lipid",
    "518": "Suy hô hấp",
    "482": "Viêm phổi vi khuẩn",
    "434": "Đột quỵ / Tắc mạch não"
}

# 2. Từ điển ánh xạ Mã phân loại CCS (Dự đoán lượt khám tiếp theo của Doctor AI)
CCS_DESCRIPTIONS = {
    "100": "Nhồi máu cơ tim cấp (Acute Myocardial Infarction)",
    "101": "Xơ vữa động mạch vành & Bệnh tim thiếu máu cục bộ",
    "108": "Suy tim sung huyết (Congestive Heart Failure)",
    "106": "Rung nhĩ & Loạn nhịp tim (Cardiac Dysrhythmias)",
    "98": "Tăng huyết áp vô căn (Essential Hypertension)",
    "157": "Suy thận cấp (Acute Renal Failure)",
    "158": "Bệnh thận mãn tính (Chronic Kidney Disease)",
    "50": "Đái tháo đường (Diabetes Mellitus)",
    "53": "Rối loạn lipid máu (Disorders of Lipid Metabolism)",
    "122": "Viêm phổi (Pneumonia)",
    "127": "Bệnh phổi tắc nghẽn mãn tính (COPD)",
    "109": "Bệnh mạch máu brains / Đột quỵ",
    "2": "Nhiễm trùng máu (Septicemia)",
    "131": "Suy hô hấp cấp (Acute Respiratory Failure)",
    "159": "Nhiễm trùng đường tiết niệu (UTI)"
}

def get_disease_name(code):
    if code is None:
        return "Không xác định"
        
    code_str = str(code).strip()
    
    # 1. Kiểm tra nếu là Mã CCS (ví dụ: CCS_108 hoặc 108)
    clean_ccs = code_str.replace("CCS_", "").strip()
    if clean_ccs in CCS_DESCRIPTIONS:
        return f"{CCS_DESCRIPTIONS[clean_ccs]} [Mã CCS {clean_ccs}]"
        
    # 2. Khớp chính xác mã ICD-9 trong từ điển
    if code_str in ICD9_DESCRIPTIONS:
        return ICD9_DESCRIPTIONS[code_str]
        
    # 3. Khớp theo tiền tố 3 chữ số (Ví dụ: 414.01 -> 414)
    prefix = code_str.split(".")[0] if "." in code_str else code_str[:3]
    if prefix in ICD9_DESCRIPTIONS:
        return f"{ICD9_DESCRIPTIONS[prefix]} (ICD-9: {code_str})"
        
    # 4. Trả về tên định dạng sạch nếu không có trong từ điển
    return f"Bệnh lý ICD-9 [{code_str}]"

def decode_temporal_weight(threshold):
    """Chuyển đổi trọng số thời gian EN(c, P) thành văn cảnh lâm sàng."""
    if threshold >= 0.5:
        return "xuất hiện ở lượt khám gần nhất (lượt t)"
    elif threshold >= 0.25:
        return "xuất hiện trong 2 lượt khám gần đây (lượt t hoặc t-1)"
    elif threshold >= 0.125:
        return "xuất hiện trong 3 lượt khám gần nhất"
    else:
        return "đã từng được ghi nhận trong lịch sử khám quá khứ"

def translate_rule_to_natural_language(feature_name, threshold, operator=">="):
    """Tạo câu điều kiện tự nhiên cho từng tiền đề luật."""
    disease_name = get_disease_name(feature_name)
    time_context = decode_temporal_weight(threshold)
    
    if operator in [">=", ">"]:
        return f"Bệnh nhân **đã có** ghi nhận *'{disease_name}'* {time_context}"
    else:
        return f"Bệnh nhân **chưa từng/ít có** chẩn đoán *'{disease_name}'* {time_context}"

def generate_full_explanation_text(rules_list, predicted_ccs_name, confidence_score):
    """Tổng hợp thành báo cáo giải thích lâm sàng tiếng Việt hoàn chỉnh."""
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