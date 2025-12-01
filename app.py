# app.py - Streamlit untuk resnet50_arcface_best_gacor.pth
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

# ============================================================================
# ARSITEKTUR MODEL YANG SAMA PERSIS DENGAN TRAINING
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
        # Tidak digunakan di inferensi, hanya untuk init weight
        return x

# ============================================================================
# SETUP
# ============================================================================
st.set_page_config(page_title="🎓 AI Absensi Wajah (ArcFace)", layout="centered")
DEVICE = "cpu"
MODEL_PATH = "best_gacor.pth"
ATTENDANCE_FILE = "attendance.json"

# Load MTCNN
mtcnn = None
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)
    st.sidebar.success("✅ MTCNN loaded")
except:
    st.sidebar.warning("⚠️ MTCNN not available – using center crop")

# ============================================================================
# LOAD MODEL (PERBAIKAN UNTUK BUG idx_to_class)
# ============================================================================
model = None
arc = None
idx_to_class = None

if not os.path.exists(MODEL_PATH):
    st.sidebar.error("❌ Model 'best_gacor.pth' tidak ditemukan!")
else:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # PERBAIKAN: HANDLE BUG DI idx_to_class
        raw_map = checkpoint["idx_to_class"]
        
        if isinstance(raw_map, list):
            # Format: ["nama1", "nama2", ...]
            idx_to_class = {i: name for i, name in enumerate(raw_map)}
            st.sidebar.info("ℹ️ idx_to_class dari list")
            
        elif isinstance(raw_map, dict):
            # Cek: apakah nilai-nilai integer? → berarti ini class_to_idx (bug)
            if all(isinstance(v, int) for v in raw_map.values()):
                # Balik mapping: {nama: idx} → {idx: nama}
                idx_to_class = {v: k for k, v in raw_map.items()}
                st.sidebar.info("ℹ️ idx_to_class dibalik (bug training)")
            else:
                # Format normal: {idx: nama}
                idx_to_class = raw_map
                st.sidebar.info("ℹ️ idx_to_class normal")
        else:
            raise ValueError("Format idx_to_class tidak dikenali")
        
        num_classes = len(idx_to_class)
        st.sidebar.write(f"Kelas: {num_classes}")
        st.sidebar.write(f"Contoh: {list(idx_to_class.items())[:3]}")

        # Muat model
        model = ResNet50Embedding(embed_dim=512, p_drop=0.5)
        model.load_state_dict(checkpoint["model"])
        
        arc = ArcMarginProduct(512, num_classes)
        arc.load_state_dict(checkpoint["arc"])
        
        model.eval()
        arc.eval()
        st.sidebar.success("✅ Model ArcFace dimuat!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        model = None

# Load student data
def load_student_data():
    data = {}
    if os.path.exists("labels-nim.csv"):
        try:
            with open("labels-nim.csv", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        nim = parts[1].strip()
                        kelas = parts[2].strip()
                        data[name] = {"nim": nim, "kelas": kelas}
        except Exception as e:
            st.sidebar.warning(f"⚠️ Gagal baca CSV: {e}")
    return data

STUDENT_DATA = load_student_data()

# Attendance
def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            return json.load(f)
    return []

def save_attendance(records):
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(records, f, indent=2)

# ============================================================================
# FACE DETECTION & PREDICTION (SAMA DENGAN INFERENCE DI COLAB)
# ============================================================================
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
                return image_array[y1:y2, x1:x2]
        except:
            pass
    h, w = image_array.shape[:2]
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return image_array[y1:y1+size, x1:x1+size]

# Transformasi yang sama dengan validasi
infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def predict_image(face_img):
    if model is None or arc is None or idx_to_class is None:
        return []
    
    if isinstance(face_img, np.ndarray):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
    
    inp = infer_tfms(face_img).unsqueeze(0)
    
    with torch.no_grad():
        emb = model(inp)
        # COSINE LOGITS DENGAN SCALE 30.0 (SESUAI INFERENCE DI COLAB)
        cos_logits = F.linear(F.normalize(emb), F.normalize(arc.weight)) * 30.0
        probs = F.softmax(cos_logits, dim=1)
    
    top3_probs, top3_idxs = torch.topk(probs, min(3, len(idx_to_class)))
    results = []
    for p, idx in zip(top3_probs[0], top3_idxs[0]):
        label = idx_to_class.get(int(idx), f"Unknown_{idx}")
        results.append({"label": label, "confidence": round(p.item() * 100, 2)})
    return results

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.title("🎓 AI Absensi Wajah Mahasiswa (ArcFace)")
st.markdown("Ambil foto atau unggah gambar untuk absen otomatis.")

option = st.radio("Pilih sumber gambar:", ("📸 Ambil Foto", "📤 Upload Gambar"), horizontal=True)
img = None

if option == "📸 Ambil Foto":
    img_file = st.camera_input("Ambil foto wajah Anda")
    if img_file:
        img = Image.open(img_file).convert("RGB")
elif option == "📤 Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

if img is not None:
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    face_crop = detect_and_crop_face(img_bgr)
    if face_crop is None:
        st.error("❌ Tidak ada wajah terdeteksi.")
    else:
        preds = predict_image(face_crop)
        if not preds:
            st.error("❌ Model tidak tersedia.")
        else:
            top = preds[0]
            name = top["label"]
            conf = top["confidence"]
            student = STUDENT_DATA.get(name, {"nim": "N/A", "kelas": "N/A"})

            # Tampilkan hasil
            st.image(img, caption="Gambar Asli", use_column_width=True)
            st.subheader(f"👤 {name}")
            st.write(f"**NIM**: {student['nim']}  \n**Kelas**: {student['kelas']}  \n**Confidence**: {conf:.2f}%")

            # Absensi
            today = datetime.now().strftime("%Y-%m-%d")
            attendance = load_attendance()
            already_marked = any(r["label"] == name and r["date"] == today for r in attendance)

            if already_marked:
                st.warning(f"✅ {name} sudah absen hari ini.")
            else:
                if st.button("✅ Absen Sekarang"):
                    new_record = {
                        "id": len(attendance) + 1,
                        "label": name,
                        "nim": student["nim"],
                        "kelas": student["kelas"],
                        "confidence": conf,
                        "timestamp": datetime.now().isoformat(),
                        "date": today,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "status": "present"
                    }
                    attendance.append(new_record)
                    save_attendance(attendance)
                    st.success(f"✔️ Absensi untuk **{name}** berhasil dicatat!")

            with st.expander("📊 Top 3 Prediksi"):
                for i, p in enumerate(preds, 1):
                    st.write(f"{i}. **{p['label']}** – {p['confidence']:.2f}%")

# Sidebar
with st.sidebar:
    st.header("📋 Absensi Hari Ini")
    today = datetime.now().strftime("%Y-%m-%d")
    records = [r for r in load_attendance() if r["date"] == today]
    if records:
        for r in records:
            st.write(f"✅ {r['label']} ({r['time']})")
    else:
        st.write("Belum ada absensi.")