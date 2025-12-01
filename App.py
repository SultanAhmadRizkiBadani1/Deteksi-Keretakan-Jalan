import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageEnhance
import io

# ---------------------------
# CONFIG / METRICS (EDIT MANUAL)
# ---------------------------
# Isi dengan hasil evaluasi nyata dari training (Colab)
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
    # Convert to RGB numpy array
    img = pil_img.convert("RGB")
    arr = np.array(img)
    arr_resized = cv2.resize(arr, (size[1], size[0]))
    arr_norm = arr_resized.astype("float32") / 255.0
    arr_batch = np.expand_dims(arr_norm, axis=0)
    return arr_batch, arr_resized

def binarize_mask(pred, thresh=0.5):
    mask = (pred > thresh).astype(np.uint8)
    return mask

def overlay_mask_on_image(image_np, mask_np, color=(255,0,0), alpha=0.5):
    # image_np: HxW x3 (0-255), mask_np: HxW (0/1)
    im = image_np.copy().astype(np.uint8)
    mask = (mask_np.squeeze() > 0).astype(np.uint8)
    if mask.ndim == 2:
        mask3 = np.stack([mask]*3, axis=-1)
    else:
        mask3 = mask
    color_arr = np.zeros_like(im)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
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

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Road Crack Detection · U-Net vs FCN", layout="wide",
                   initial_sidebar_state="expanded")

# Simple custom CSS for nicer look
st.markdown("""
    <style>
    .stApp { background-color: #f7fbff; }
    .reportview-container .main .block-container{ padding-top:1.5rem; }
    .big-font { font-size:20px; color:#0b486b; font-weight:600;}
    .metric-label { color:#1f618d; font-weight:600; }
    .small-muted { color:#5d6d7e; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("Road Crack Detection — U-Net vs FCN")
st.write("Perbandingan kinerja dua model segmentasi untuk mendeteksi retak jalan. Upload gambar, lihat hasil, dan dapatkan rekomendasi otomatis berdasarkan metrik.")

# Sidebar - model status & info
with st.sidebar:
    st.header("Model & Info")
    st.markdown("**Model files (must be in same folder):** `unet_model.h5`, `fcn_model.h5`")
    st.info("Jika model besar, loading bisa memakan waktu beberapa detik.")
    st.markdown("---")
    st.subheader("Evaluasi")
    st.metric("U-Net IoU", f"{eval_unet_iou:.3f}")
    st.metric("FCN IoU", f"{eval_fcn_iou:.3f}")
    st.markdown("---")
    st.subheader("Rekomendasi (otomatis)")
    if eval_unet_iou > eval_fcn_iou:
        st.success("✔ U-Net direkomendasikan (IoU lebih tinggi).")
    elif eval_unet_iou < eval_fcn_iou:
        st.success("✔ FCN direkomendasikan (IoU lebih tinggi).")
    else:
        st.info("Hasil imbang — pertimbangkan kecepatan vs akurasi.")
    st.markdown("---")
    st.markdown("Tips:\n- Untuk hasil terbaik gunakan gambar resolusi baik.\n- Jika punya ground-truth masks, upload untuk hitung IoU real-time.")

# Load models (show spinner)
with st.spinner("Memuat model U-Net & FCN..."):
    unet_model, err_unet = load_model_safe("unet_model.h5")
    fcn_model, err_fcn   = load_model_safe("fcn_model.h5")

if err_unet or err_fcn:
    st.warning("Beberapa model gagal dimuat. Pastikan file .h5 ada di folder project.")
    if err_unet:
        st.write("U-Net load error:", err_unet)
    if err_fcn:
        st.write("FCN load error:", err_fcn)
    st.stop()

st.success("Model dimuat ✅")

# Main: Upload and predict
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Upload Gambar Jalan")
    uploaded = st.file_uploader("Pilih gambar (jpg/png) untuk dideteksi retaknya", type=["jpg","jpeg","png"])
    st.markdown("<div class='small-muted'>Model input size: 128×128 (image akan di-resize untuk inferensi)</div>", unsafe_allow_html=True)

    st.caption("Opsional: upload juga file mask ground-truth (hitam-putih) di bagian bawah untuk menghitung IoU.")

    # Optional ground-truth mask uploader
    gt_file = st.file_uploader("Upload Ground-Truth Mask (opsional, png/jpg) — satu file untuk gambar ini", type=["png","jpg","jpeg"])

with col2:
    st.subheader("2) Ringkasan Evaluasi")
    st.metric("U-Net — Loss", f"{eval_unet_loss:.4f}")
    st.metric("U-Net — Acc",  f"{eval_unet_acc:.3f}")
    st.metric("U-Net — IoU",  f"{eval_unet_iou:.3f}")
    st.markdown("---")
    st.metric("FCN — Loss", f"{eval_fcn_loss:.4f}")
    st.metric("FCN — Acc",  f"{eval_fcn_acc:.3f}")
    st.metric("FCN — IoU",  f"{eval_fcn_iou:.3f}")

# When image is uploaded
if uploaded:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Gagal membaca gambar: " + str(e))
        st.stop()

    # Preprocess and inference
    input_batch, img_resized = preprocess_image_for_model(pil_img, size=(128,128))
    pred_unet = unet_model.predict(input_batch)
    pred_fcn  = fcn_model.predict(input_batch)

    mask_unet = binarize_mask(pred_unet, thresh=0.5)[0]
    mask_fcn  = binarize_mask(pred_fcn,  thresh=0.5)[0]

    # Overlay (use resized for overlay)
    overlay_unet = overlay_mask_on_image(img_resized, mask_unet, color=(255,0,0), alpha=0.45)
    overlay_fcn  = overlay_mask_on_image(img_resized, mask_fcn, color=(0,255,255), alpha=0.45)

    # Show original + predictions side-by-side
    st.markdown("### Hasil Deteksi")
    gcol1, gcol2, gcol3 = st.columns(3)
    gcol1.image(img_resized, caption="Resized Input (128×128)", use_column_width=True)
    gcol2.image(overlay_unet, caption="U-Net — Overlay", use_column_width=True)
    gcol3.image(overlay_fcn, caption="FCN — Overlay", use_column_width=True)

    # Show mask raw
    mcol1, mcol2 = st.columns(2)
    mcol1.image(mask_unet.squeeze()*255, caption="Mask U-Net (Biner)", use_column_width=True)
    mcol2.image(mask_fcn.squeeze()*255, caption="Mask FCN (Biner)", use_column_width=True)

    # If ground-truth mask provided, compute IoU
    if gt_file:
        try:
            gt_pil = Image.open(gt_file).convert("L")
            gt_arr = np.array(gt_pil)
            # resize gt to 128x128 if needed
            gt_resized = cv2.resize(gt_arr, (128,128))
            # threshold gt
            gt_bin = (gt_resized > 127).astype(np.uint8)

            iou_unet = compute_iou_from_masks(gt_bin, mask_unet.squeeze())
            iou_fcn  = compute_iou_from_masks(gt_bin, mask_fcn.squeeze())

            st.markdown("### Evaluasi pada Ground-Truth (ter-upload)")
            st.write(f"- IoU U-Net : **{iou_unet:.4f}**")
            st.write(f"- IoU FCN  : **{iou_fcn:.4f}**")

            if iou_unet > iou_fcn:
                st.success("Rekomendasi: **U-Net** (lebih sesuai untuk gambar ini).")
            elif iou_fcn > iou_unet:
                st.success("Rekomendasi: **FCN** (lebih sesuai untuk gambar ini).")
            else:
                st.info("Hasil imbang untuk gambar ini.")
        except Exception as e:
            st.warning("Gagal memproses ground-truth mask: " + str(e))

    # Provide download buttons for masks
    buf_unet = (mask_unet.squeeze()*255).astype(np.uint8)
    buf_fcn  = (mask_fcn.squeeze()*255).astype(np.uint8)

    # Convert to PIL for download
    pil_unet_mask = Image.fromarray(buf_unet)
    pil_fcn_mask  = Image.fromarray(buf_fcn)

    col_down1, col_down2, col_down3 = st.columns([1,1,2])
    with col_down1:
        b1 = st.download_button("Download Mask U-Net", data=pil_unet_mask.tobytes(), file_name="mask_unet.png", mime="image/png")
    with col_down2:
        b2 = st.download_button("Download Mask FCN", data=pil_fcn_mask.tobytes(), file_name="mask_fcn.png", mime="image/png")
    with col_down3:
        st.markdown("**Catatan:** File mask disajikan dalam format biner (0/255). Gunakan editor gambar untuk overlay dengan original jika perlu.")

# Footer / About
st.markdown("---")
st.markdown("### Tentang Aplikasi")
st.write("""
Aplikasi ini dibuat untuk demonstrasi hasil deteksi retak jalan menggunakan dua arsitektur segmentasi: **U-Net** dan **FCN**.
Masukkan model (hasil training dari Colab) ke folder project sebelum menjalankan aplikasi.
""")
st.markdown("**Dibuat oleh:** Sultan Ahmad Rizki Badani")
