# app.py - Deep Learning RA - Sistem Absensi Face Recognition (Modern Clean UI)
import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime
import streamlit as st
from io import BytesIO
import time

# ============================================================================
# ARSITEKTUR MODEL YANG SAMA DENGAN TRAINING
# ============================================================================
class ResNet50Embedding(nn.Module):
    def __init__(self, embed_dim=512, p_drop=0.5):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_feats = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.bn = nn.BatchNorm1d(in_feats)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_feats, embed_dim)
    
    def forward(self, x):
        f = self.backbone(x)
        f = self.bn(f)
        f = self.drop(f)
        return self.fc(f)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, labels=None):
        return x

# ============================================================================
# SETUP STREAMLIT
# ============================================================================
st.set_page_config(page_title="Face Recognition Attendance", layout="wide", initial_sidebar_state="collapsed")
DEVICE = "cpu"
MODEL_PATH = "best_gacor.pth"
ATTENDANCE_FILE = "attendance.json"

# Modern Clean CSS - Blue & White Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(37, 99, 235, 0.2);
    }
    
    .app-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(37, 99, 235, 0.1);
    }
    
    .card-header {
        font-size: 18px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Status Box */
    .status-box {
        text-align: center;
        padding: 3rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #bfdbfe;
    }
    
    .status-icon {
        font-size: 64px;
        margin-bottom: 1rem;
    }
    
    .status-title {
        font-size: 20px;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .status-text {
        font-size: 14px;
        color: #64748b;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 2rem;
        border: 2px solid #e0e7ff;
        margin: 1.5rem 0;
    }
    
    .result-name {
        font-size: 28px;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    .result-info {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .info-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2563eb;
    }
    
    .info-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    
    .info-value {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Attendance List */
    .attendance-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #10b981;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .attendance-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 14px;
    }
    
    .attendance-time {
        font-size: 12px;
        color: #64748b;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Upload Area */
    .uploadedFile {
        border: 2px dashed #93c5fd;
        border-radius: 8px;
        background: #eff6ff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
        color: #1e40af;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #64748b;
        font-size: 14px;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #2563eb !important;
    }
    
    /* Prediction Badge */
    .prediction-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Success Alert */
    .success-alert {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #6ee7b7;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Warning Alert */
    .warning-alert {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #fcd34d;
        color: #78350f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load MTCNN
mtcnn = None
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)
except:
    pass

# ============================================================================
# LOAD MODEL
# ============================================================================
model = None
arc = None
idx_to_class = None

if not os.path.exists(MODEL_PATH):
    st.error("Model 'best_gacor.pth' tidak ditemukan!")
else:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        raw_map = checkpoint["idx_to_class"]
        
        if isinstance(raw_map, list):
            idx_to_class = {i: name for i, name in enumerate(raw_map)}
        elif isinstance(raw_map, dict):
            if all(isinstance(v, int) for v in raw_map.values()):
                idx_to_class = {v: k for k, v in raw_map.items()}
            else:
                idx_to_class = raw_map
        else:
            raise ValueError("Format idx_to_class tidak dikenali")
        
        num_classes = len(idx_to_class)
        
        model = ResNet50Embedding(embed_dim=512, p_drop=0.5)
        model.load_state_dict(checkpoint["model"])
        
        arc = ArcMarginProduct(512, num_classes)
        arc.load_state_dict(checkpoint["arc"])
        
        model.eval()
        arc.eval()
        
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")

# Load student data
def load_student_data():
    data = {}
    if os.path.exists("labels-nim.csv"):
        try:
            with open("labels-nim.csv", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        # Format: filename, nama, nim, kelas
                        name = parts[1].strip()
                        nim = parts[2].strip()
                        kelas = parts[3].strip()
                        data[name] = {"nim": nim, "kelas": kelas}
        except Exception as e:
            st.error(f"Error loading student data: {str(e)}")
    return data

STUDENT_DATA = load_student_data()

# Attendance functions
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            return json.load(f)
    return []

def save_attendance(records):
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f, indent=2)

# ============================================================================
# FACE DETECTION & PREDICTION
# ============================================================================
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def detect_and_crop_face(image_array):
    if mtcnn is not None:
        try:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                h, w = image_array.shape[:2]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                return image_array[y1:y2, x1:x2], (x1, y1, x2, y2)
        except:
            pass
    h, w = image_array.shape[:2]
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return image_array[y1:y1+size, x1:x1+size], None

def predict_image(face_img):
    if model is None or arc is None or idx_to_class is None:
        return []
    
    if isinstance(face_img, np.ndarray):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
    
    inp = infer_tfms(face_img).unsqueeze(0)
    
    with torch.no_grad():
        emb = model(inp)
        cos_logits = F.linear(F.normalize(emb), F.normalize(arc.weight)) * 30.0
        probs = F.softmax(cos_logits, dim=1)
    
    top3_probs, top3_idxs = torch.topk(probs, min(3, len(idx_to_class)))
    results = []
    for p, idx in zip(top3_probs[0], top3_idxs[0]):
        label = idx_to_class.get(int(idx), f"Unknown_{idx}")
        results.append({"label": label, "confidence": round(p.item() * 100, 2)})
    return results

# ============================================================================
# UI LAYOUT
# ============================================================================

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">Face Recognition Attendance System</div>
    <div class="app-subtitle">Sistem Absensi Berbasis Deep Learning</div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col1, col2, col3 = st.columns([1, 2, 1])

# LEFT COLUMN - Attendance History
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Riwayat Kehadiran</div>', unsafe_allow_html=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    attendance = load_attendance()
    today_records = [r for r in attendance if r["date"] == today]
    
    if today_records:
        for r in today_records:
            st.markdown(f"""
            <div class="attendance-item">
                <div>
                    <div class="attendance-name">{r['label']}</div>
                    <div class="attendance-time">NIM: {r.get('nim', 'N/A')}</div>
                </div>
                <div class="attendance-time">{r['time']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-text" style="text-align: center; padding: 2rem;">Belum ada kehadiran hari ini</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# CENTER COLUMN - Main Interface
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Verifikasi Wajah</div>', unsafe_allow_html=True)
    
    option = st.radio("Pilih Metode Input", ("Upload Foto", "Gunakan Kamera"), horizontal=True, label_visibility="collapsed")
    
    img = None
    if option == "Upload Foto":
        uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        img_file = st.camera_input("Ambil foto", label_visibility="collapsed")
        if img_file:
            img = Image.open(img_file).convert("RGB")
    
    if img is not None:
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Memproses wajah..."):
            time.sleep(0.5)
        
        face_crop, bbox = detect_and_crop_face(img_bgr)
        
        if face_crop is None:
            st.markdown('<div class="warning-alert">Tidak ada wajah terdeteksi dalam gambar</div>', unsafe_allow_html=True)
        else:
            preds = predict_image(face_crop)
            
            if not preds:
                st.error("Model tidak tersedia")
            else:
                top = preds[0]
                name = top["label"]
                conf = top["confidence"]
                student = STUDENT_DATA.get(name, {"nim": "N/A", "kelas": "N/A"})
                
                # Display Image with better sizing
                col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
                with col_img2:
                    st.image(img, use_column_width=True, caption="Foto Terdeteksi")
                
                # Result Card
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-name">{name}</div>
                    <div class="result-info">
                        <div class="info-item">
                            <div class="info-label">NIM</div>
                            <div class="info-value">{student['nim']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Kelas</div>
                            <div class="info-value">{student['kelas']}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Confidence</div>
                            <div class="info-value">{conf:.2f}%</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Waktu</div>
                            <div class="info-value">{datetime.now().strftime("%H:%M:%S")}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Attendance Button
                attendance_list = load_attendance()
                already_marked = any(r["label"] == name and r["date"] == today for r in attendance_list)
                
                if already_marked:
                    st.markdown(f'<div class="success-alert">Kehadiran untuk {name} sudah tercatat hari ini</div>', unsafe_allow_html=True)
                else:
                    if st.button("Catat Kehadiran"):
                        new_record = {
                            "id": len(attendance_list) + 1,
                            "label": name,
                            "nim": student["nim"],
                            "kelas": student["kelas"],
                            "confidence": conf,
                            "timestamp": datetime.now().isoformat(),
                            "date": today,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "status": "present"
                        }
                        attendance_list.append(new_record)
                        save_attendance(attendance_list)
                        st.markdown(f'<div class="success-alert">Kehadiran untuk {name} berhasil dicatat</div>', unsafe_allow_html=True)
                        st.rerun()
                
                # Top 3 Predictions
                with st.expander("Lihat Detail Prediksi"):
                    for i, p in enumerate(preds, 1):
                        student_info = STUDENT_DATA.get(p['label'], {"nim": "N/A", "kelas": "N/A"})
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                            <strong>{i}. {p['label']}</strong><br>
                            <small>NIM: {student_info['nim']} | Kelas: {student_info['kelas']} | Confidence: {p['confidence']:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN - Status
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Status Kehadiran</div>', unsafe_allow_html=True)
    
    if today_records:
        st.markdown(f"""
        <div class="status-box">
            <div class="status-icon">&#10004;</div>
            <div class="status-title">Aktif</div>
            <div class="status-text">Total kehadiran hari ini</div>
            <div style="font-size: 36px; font-weight: 700; color: #1e40af; margin-top: 1rem;">{len(today_records)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box">
            <div class="status-icon">&#8987;</div>
            <div class="status-title">Menunggu</div>
            <div class="status-text">Belum ada kehadiran yang tercatat hari ini</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    Deep Learning RA - Trio Kwek Kwek Teams
</div>
""", unsafe_allow_html=True)