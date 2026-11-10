import numpy as np
import torch
import torch.nn as nn
from functools import lru_cache
from sklearn.tree import DecisionTreeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Class mô hình Hộp đen Doctor AI
class DoctorAI(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super(DoctorAI, self).__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        emb = self.dropout(self.relu(self.embedding(x)))
        gru_out, _ = self.gru(emb)
        return self.fc_out(gru_out)

# 2. Các hàm đo khoảng cách Ontology & DTW
def get_icd_prefix(code_str):
    if "." in code_str:
        return code_str.split(".")[0]
    return code_str[:3] if len(code_str) >= 3 else code_str

@lru_cache(maxsize=None)
def wualmer_similarity(code1, code2):
    if code1 == code2:
        return 1.0
    prefix1 = get_icd_prefix(code1)
    prefix2 = get_icd_prefix(code2)
    if prefix1 == prefix2:
        return 0.75
    return 0.0

@lru_cache(maxsize=None)
def visit_distance_levenshtein(visit1_tuple, visit2_tuple):
    if not visit1_tuple and not visit2_tuple: return 0.0
    if not visit1_tuple: return float(len(visit2_tuple))
    if not visit2_tuple: return float(len(visit1_tuple))
    
    n, m = len(visit1_tuple), len(visit2_tuple)
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    for i in range(n + 1): dp[i, 0] = float(i)
    for j in range(m + 1): dp[0, j] = float(j)
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 1.0 - wualmer_similarity(visit1_tuple[i-1], visit2_tuple[j-1])
            dp[i, j] = min(dp[i-1, j] + 1.0, dp[i, j-1] + 1.0, dp[i-1, j-1] + cost)
    return dp[n, m]

def dtw_patient_distance(seq1, seq2):
    N, M = len(seq1), len(seq2)
    dtw_matrix = np.zeros((N + 1, M + 1), dtype=np.float32)
    dtw_matrix[0, 1:] = float("inf")
    dtw_matrix[1:, 0] = float("inf")
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = visit_distance_levenshtein(tuple(seq1[i-1]), tuple(seq2[j-1]))
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[N, M]

def find_k_real_neighbors(target_patient_id, sequences, train_ids, k=10):
    target_seq = sequences[target_patient_id]
    distances = []
    for pid in train_ids:
        if pid == target_patient_id:
            continue
        dist = dtw_patient_distance(target_seq, sequences[pid])
        distances.append((pid, dist))
    distances.sort(key=lambda x: x[1])
    return [pid for pid, _ in distances[:k]]

def ontological_perturbation(patient_sequence, num_samples=50):
    synthetic_samples = [patient_sequence]
    all_prefixes = set()
    for visit in patient_sequence:
        for code in visit:
            all_prefixes.add(get_icd_prefix(code))
    all_prefixes = list(all_prefixes)
    if not all_prefixes:
        return synthetic_samples
    for _ in range(num_samples):
        masked_prefix = np.random.choice(all_prefixes)
        new_seq = []
        for visit in patient_sequence:
            new_visit = [code for code in visit if get_icd_prefix(code) != masked_prefix]
            new_seq.append(new_visit)
        synthetic_samples.append(new_seq)
    return synthetic_samples

def temporal_encoding(patient_sequence, code2idx, vocab_size):
    n_visits = len(patient_sequence)
    flat_vector = np.zeros(vocab_size, dtype=np.float32)
    for i, visit in enumerate(patient_sequence):
        weight = (0.5) ** (n_visits - i)
        for code in visit:
            if code in code2idx:
                flat_vector[code2idx[code]] += weight
    return flat_vector

# 3. Thuật toán chính Doctor XAI Explainer
def explain_patient_doctor_xai(target_pid, sequences, train_ids, black_box_model, code2idx, idx2code, vocab_size, k_neighbors=3, num_synth=30):
    target_seq = sequences[target_pid]
    real_neighbor_ids = find_k_real_neighbors(target_pid, sequences, train_ids, k=k_neighbors)
    all_synth_seqs = []
    for pid in [target_pid] + real_neighbor_ids:
        synth_seqs = ontological_perturbation(sequences[pid], num_samples=num_synth // max(1, len(real_neighbor_ids)))
        all_synth_seqs.extend([s for s in synth_seqs if s])
        
    X_flat = np.array([temporal_encoding(seq, code2idx, vocab_size) for seq in all_synth_seqs])
    max_len = max(len(seq) for seq in all_synth_seqs)
    batch_tensor = torch.zeros((len(all_synth_seqs), max_len, vocab_size), dtype=torch.float32).to(device)
    
    for b_idx, seq in enumerate(all_synth_seqs):
        for t, visit in enumerate(seq):
            for code in visit:
                if code in code2idx:
                    batch_tensor[b_idx, t, code2idx[code]] = 1.0
                    
    with torch.no_grad():
        logits = black_box_model(batch_tensor)
        last_logits = logits[:, -1, :]
        probs = torch.sigmoid(last_logits).cpu().numpy()
        Y_black_box = (probs >= 0.3).astype(int)
        
    explainer_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    explainer_dt.fit(X_flat, Y_black_box)
    
    target_flat = temporal_encoding(target_seq, code2idx, vocab_size).reshape(1, -1)
    target_pred_bb = explainer_dt.predict(target_flat)
    
    return explainer_dt, target_seq, target_pred_bb