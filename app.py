# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Flood Segmentation Tool", layout="wide")

# Upload folder (optional, for temp storage if needed)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def segment_flood(img):
    """Return RGB image and predicted mask."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_pred = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return img_rgb, mask_pred

def create_fake_groundtruth(mask_pred):
    """Simulate a ground truth mask for metrics."""
    noise = np.random.randint(0, 2, mask_pred.shape, dtype=np.uint8) * 255
    mask_gt = cv2.bitwise_and(mask_pred, cv2.bitwise_not(noise))
    return mask_gt

def calculate_metrics(pred, gt):
    """IoU, Dice, Pixel Accuracy"""
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) != 0 else 0
    acc = (pred_bin == gt_bin).sum() / gt_bin.size
    return iou, dice, acc

# --- Streamlit UI ---
st.title("🌊 Flood Area Segmentation Tool")
st.write("Upload PNG/JPG images, run segmentation, and view metrics instantly.")

uploaded_files = st.file_uploader(
    "Upload images",
    type=list(ALLOWED_EXTENSIONS),
    accept_multiple_files=True
)

results = []

if uploaded_files:
    for file in uploaded_files:
        if allowed_file(file.name):
            # Read file bytes into OpenCV
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                st.warning(f"⚠️ Failed to read image: {file.name}. Please upload a valid PNG or JPG.")
                continue

            # Process segmentation
            img_rgb, mask_pred = segment_flood(img)
            mask_gt = create_fake_groundtruth(mask_pred)
            iou, dice, acc = calculate_metrics(mask_pred, mask_gt)

            results.append({
                "filename": file.name,
                "iou": round(iou, 3),
                "dice": round(dice, 3),
                "acc": round(acc, 3),
                "img": img_rgb,
                "mask_pred": mask_pred,
                "mask_gt": mask_gt
            })

    # --- Display results ---
    st.subheader("Segmentation Results")
    for r in results:
        st.markdown(f"### {r['filename']}")
        cols = st.columns(3)
        cols[0].image(r["img"], caption="Original Image", use_column_width=True)
        cols[1].image(r["mask_pred"], caption="Predicted Mask", use_column_width=True)
        cols[2].image(r["mask_gt"], caption="Ground Truth (simulated)", use_column_width=True)
        st.write(f"IoU: **{r['iou']}** | Dice: **{r['dice']}** | Pixel Accuracy: **{r['acc']}**")

    # --- CSV Download ---
    df = pd.DataFrame([{
        "filename": r["filename"],
        "iou": r["iou"],
        "dice": r["dice"],
        "acc": r["acc"]
    } for r in results])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Report", csv, "report.csv", "text/csv")
