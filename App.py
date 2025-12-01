# app.py — Streamlit Road Crack Detection (Improved theme + GT preview + robust IO)
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io

# ---------------------------
# CONFIG / METRICS (EDIT MANUAL)
# ---------------------------
eval_fcn_loss = 0.0763
eval_fcn_acc  = 0.9740
eval_fcn_iou  = 0.3587

eval_unet_loss = 0.0693
eval_unet_acc  = 0.9763
eval_unet_iou  = 0.3762

# ---------------------------
# Helper functions
# ---------------------------
def load_model_safe(path):
    try:
        m = tf.keras.models.load_model(path, compile=False)
        return m, None
    except Exception as e:
        return None, str(e)

def preprocess_image_for_model(pil_img, size=(128,128)):
    img = pil_img.convert("RGB")
    arr = np.array(img)
    arr_resized = cv2.resize(arr, (size[1], size[0]))
    arr_norm = arr_resized.astype("float32") / 255.0
    arr_batch = np.expand_dims(arr_norm, axis=0)
    return arr_batch, arr_resized

def binarize_mask(pred, thresh=0.5):
    return (pred > thresh).astype(np.uint8)

def overlay_mask_on_image(image_np, mask_np, color=(255,0,0), alpha=0.5):
    im = image_np.copy().astype(np.uint8)
    mask = (mask_np.squeeze() > 0).astype(np.uint8)
    if mask.ndim == 2:
        mask3 = np.stack([mask]*3, axis=-1)
    else:
        mask3 = mask
    color_arr = np.zeros_like(im)
    color_arr[..., 0] = int(color[0])
    color_arr[..., 1] = int(color[1])
    color_arr[..., 2] = int(color[2])
    overlay = np.where(mask3, (im*(1-alpha) + color_arr*alpha).astype(np.uint8), im)
    return overlay

def compute_iou_from_masks(gt_mask, pred_mask):
    gt = (gt_mask > 0).astype(np.uint8)
    pr = (pred_mask > 0).astype(np.uint8)
    intersection = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# STREAMLIT UI & THEME (robust)
# ---------------------------
st.set_page_config(page_title="Road Crack Detection · U-Net vs FCN",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Force a consistent baseline and disable browser/OS auto dark overrides that break contrast.
# We still provide a manual Light/Dark theme switch below.
base_reset_css = """
<style>
/* try to stop browser forced dark heuristics */
:root { color-scheme: light !important; }
html, body, .stApp { background-color: #f4f7fb !important; }
</style>
"""
st.markdown(base_reset_css, unsafe_allow_html=True)

# Precise theme CSS for light + dark (high-contrast, overrides many Streamlit internal classes).
LIGHT_THEME = """
<style>
/* MAIN */
.stApp { background-color: #f4f7fb !important; color: #0b1b2b !important; }

/* container */
.reportview-container .main .block-container {
    background-color: #ffffff !important;
    padding: 20px 24px !important;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(20,40,80,0.06);
}

/* sidebar */
.sidebar .sidebar-content {
    background-color: #ffffff !important;
    border-right: 1px solid #e6eef8;
}

/* headings */
h1, h2, h3, h4 { color: #0b1b2b !important; font-weight:700 !important; }

/* uploader border wrapper (applied via st.markdown wrapper) */
.uploader-box { border: 2px dashed #c6d5e8 !important; padding: 12px; border-radius:8px; }

/* buttons */
.stButton>button {
    background-color: #1565c0 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    box-shadow: none !important;
    border: none !important;
}

/* metrics style */
div[data-testid="metric-container"] {
    background: linear-gradient(90deg, #eef8ff, #f7fbff) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* file uploader text color */
div[role="listitem"] { color: #123 !important; }

/* footer small */
footer { color: #5b6b7b !important; }
</style>
"""

DARK_THEME = """
<style>
.stApp { background-color: #0b0f14 !important; color: #e8eef6 !important; }

.reportview-container .main .block-container {
    background-color: #0f1418 !important;
    padding: 20px 24px !important;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
}

/* sidebar */
.sidebar .sidebar-content {
    background-color: #0b0f14 !important;
    border-right: 1px solid #202428;
}

/* headings */
h1, h2, h3, h4 { color: #f7fbff !important; font-weight:700 !important; }

/* uploader border wrapper */
.uploader-box { border: 2px dashed #444 !important; padding: 12px; border-radius:8px; }

/* buttons */
.stButton>button {
    background-color: #1f6feb !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* metrics style */
div[data-testid="metric-container"] {
    background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

footer { color: #9aa6b2 !important; }
</style>
"""

# Sidebar theme switcher (manual)
with st.sidebar:
    st.markdown("## Tema Aplikasi")
    theme_choice = st.radio("Pilih tema:", ("Light", "Dark"))
    st.markdown("---")

# Apply chosen theme
if theme_choice == "Dark":
    st.markdown(DARK_THEME, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME, unsafe_allow_html=True)

# ---------------------------
# Header & Sidebar Info
# ---------------------------
st.title("🛣️ Road Crack Detection — U-Net vs FCN")
st.write("Perbandingan kinerja dua model segmentasi untuk mendeteksi retak jalan. Upload gambar, lihat hasil, dan dapatkan rekomendasi otomatis berdasarkan metrik.")

with st.sidebar:
    st.header("Model & Info")
    st.markdown("**Model files:** `unet_model.h5`, `fcn_model.h5`")
    st.info("Jika model besar, loading bisa memakan waktu beberapa detik.")
    st.markdown("---")
    st.subheader("Evaluasi (training)")
    col_a, col_b = st.columns(2)
    col_a.metric("U-Net IoU", f"{eval_unet_iou:.3f}")
    col_b.metric("FCN IoU", f"{eval_fcn_iou:.3f}")
    st.markdown("---")
    st.subheader("Rekomendasi (otomatis)")
    if eval_unet_iou > eval_fcn_iou:
        st.success("U-Net direkomendasikan (IoU lebih tinggi).")
    elif eval_unet_iou < eval_fcn_iou:
        st.success("FCN direkomendasikan (IoU lebih tinggi).")
    else:
        st.info("Hasil imbang — pertimbangkan kecepatan vs akurasi.")
    st.markdown("---")
    st.caption("Tips: Gunakan gambar resolusi baik. Upload GT mask untuk evaluasi per gambar.")

# ---------------------------
# Load models
# ---------------------------
with st.spinner("Memuat model U-Net & FCN..."):
    unet_model, err_unet = load_model_safe("unet_model.h5")
    fcn_model, err_fcn   = load_model_safe("fcn_model.h5")

if err_unet or err_fcn:
    st.error("Beberapa model gagal dimuat. Pastikan file `.h5` ada di folder project.")
    if err_unet:
        st.write("U-Net load error:", err_unet)
    if err_fcn:
        st.write("FCN load error:", err_fcn)
    st.stop()

st.success("Model dimuat ✅")

# ---------------------------
# Upload area
# ---------------------------
col_main, col_eval = st.columns([2.2, 1])

with col_main:
    st.subheader("1) Upload Gambar Jalan")
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih gambar (jpg/jpeg/png) untuk dideteksi retaknya", type=["jpg","jpeg","png"])
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<small>Model input size: 128×128 (image akan di-resize untuk inferensi)</small>", unsafe_allow_html=True)

    st.caption("Opsional: upload juga file mask ground-truth (hitam-putih) di bawah untuk menghitung IoU per gambar.")
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    gt_file = st.file_uploader("Upload Ground-Truth Mask (opsional, png/jpg)", type=["png","jpg","jpeg"], key="gt")
    st.markdown('</div>', unsafe_allow_html=True)

with col_eval:
    st.subheader("2) Ringkasan Evaluasi (training)")
    st.write("**U-Net**")
    st.write(f"- Loss: {eval_unet_loss:.4f}")
    st.write(f"- Accuracy: {eval_unet_acc:.4f}")
    st.write(f"- IoU: {eval_unet_iou:.4f}")
    st.markdown("---")
    st.write("**FCN**")
    st.write(f"- Loss: {eval_fcn_loss:.4f}")
    st.write(f"- Accuracy: {eval_fcn_acc:.4f}")
    st.write(f"- IoU: {eval_fcn_iou:.4f}")

# ---------------------------
# Prediction + Display
# ---------------------------
if uploaded:
    # read uploaded image robustly
    try:
        uploaded_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception as e:
        st.error("Gagal membaca gambar upload: " + str(e))
        st.stop()

    input_batch, img_resized = preprocess_image_for_model(pil_img, size=(128,128))

    # inference
    pred_unet = unet_model.predict(input_batch)
    pred_fcn  = fcn_model.predict(input_batch)

    mask_unet = binarize_mask(pred_unet)[0]
    mask_fcn  = binarize_mask(pred_fcn)[0]

    overlay_unet = overlay_mask_on_image(img_resized, mask_unet, color=(255,40,40), alpha=0.5)
    overlay_fcn  = overlay_mask_on_image(img_resized, mask_fcn, color=(0,200,180), alpha=0.45)

    st.subheader("Hasil Deteksi")
    c1, c2, c3 = st.columns([1.1,1.1,1.1])
    c1.image(img_resized, caption="Input (resized 128×128)", use_column_width=True)
    c2.image(overlay_unet, caption="U-Net — Overlay", use_column_width=True)
    c3.image(overlay_fcn, caption="FCN — Overlay", use_column_width=True)

    # raw masks
    m1, m2, m3 = st.columns(3)
    m1.image((mask_unet.squeeze()*255).astype("uint8"), caption="Mask U-Net (biner)", use_column_width=True)
    m2.image((mask_fcn.squeeze()*255).astype("uint8"), caption="Mask FCN (biner)", use_column_width=True)

    # If GT uploaded: show GT image, GT mask (resized), and compute IoU
    if gt_file:
        try:
            gt_bytes = gt_file.read()
            gt_pil = Image.open(io.BytesIO(gt_bytes)).convert("L")
            gt_arr_raw = np.array(gt_pil)
            gt_resized = cv2.resize(gt_arr_raw, (128,128))
            gt_bin = (gt_resized > 127).astype(np.uint8)

            # Show GT image and GT mask
            st.subheader("Ground-Truth (uploaded)")
            gcol1, gcol2, gcol3 = st.columns([1,1,1])
            gcol1.image(img_resized, caption="Input")
            gcol2.image((gt_resized).astype("uint8"), caption="GT Mask (resized)")
            # overlay GT on input for clarity
            gt_overlay = overlay_mask_on_image(img_resized, np.expand_dims(gt_bin, axis=0), color=(255,255,0), alpha=0.5)
            gcol3.image(gt_overlay, caption="GT overlay on Input")

            # compute IoU per model
            iou_unet = compute_iou_from_masks(gt_bin, mask_unet.squeeze())
            iou_fcn  = compute_iou_from_masks(gt_bin, mask_fcn.squeeze())

            st.markdown("### Evaluasi per gambar (Ground-Truth)")
            st.write(f"- IoU U-Net : **{iou_unet:.4f}**")
            st.write(f"- IoU FCN  : **{iou_fcn:.4f}**")

            # Recommendation per-image
            if iou_unet > iou_fcn:
                st.success("Rekomendasi untuk gambar ini: **U-Net** lebih sesuai.")
            elif iou_fcn > iou_unet:
                st.success("Rekomendasi untuk gambar ini: **FCN** lebih sesuai.")
            else:
                st.info("Rekomendasi: hasil imbang untuk gambar ini.")
        except Exception as e:
            st.warning("Gagal memproses ground-truth: " + str(e))

    # Download mask buttons (save as PNG properly)
    try:
        buf_unet = Image.fromarray((mask_unet.squeeze()*255).astype("uint8"))
        buf_fcn  = Image.fromarray((mask_fcn.squeeze()*255).astype("uint8"))

        down_col1, down_col2, _ = st.columns([1,1,2])
        with down_col1:
            st.download_button(
                label="Download Mask U-Net",
                data=pil_to_bytes(buf_unet, fmt="PNG"),
                file_name="mask_unet.png",
                mime="image/png"
            )
        with down_col2:
            st.download_button(
                label="Download Mask FCN",
                data=pil_to_bytes(buf_fcn, fmt="PNG"),
                file_name="mask_fcn.png",
                mime="image/png"
            )
    except Exception as e:
        st.warning("Gagal menyiapkan download mask: " + str(e))

# ---------------------------
# Footer / About
# ---------------------------
st.markdown("---")
st.markdown("### Tentang Aplikasi")
st.write("""
Aplikasi ini dibuat untuk demonstrasi deteksi retak jalan menggunakan dua arsitektur segmentasi: **U-Net** dan **FCN**.
Masukkan file model `.h5` (hasil training) ke folder project sebelum menjalankan aplikasi.
""")
st.markdown("**Dibuat oleh:** Sultan Ahmad Rizki Badani")
