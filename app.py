
# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
from matplotlib import pyplot as plt

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Helpers ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def segment_flood(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask_pred = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return img_rgb, mask_pred

def create_fake_groundtruth(mask_pred):
    noise = np.random.randint(0, 2, mask_pred.shape, dtype=np.uint8) * 255
    mask_gt = cv2.bitwise_and(mask_pred, cv2.bitwise_not(noise))
    return mask_gt

def calculate_metrics(pred, gt):
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) != 0 else 0
    acc = (pred_bin == gt_bin).sum() / gt_bin.size
    return iou, dice, acc

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ---------------- Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('images')
        for f in files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)

                img, mask_pred = segment_flood(filepath)
                mask_gt = create_fake_groundtruth(mask_pred)
                iou, dice, acc = calculate_metrics(mask_pred, mask_gt)

                results.append({
                    'filename': filename,
                    'iou': round(iou, 3),
                    'dice': round(dice, 3),
                    'acc': round(acc, 3),
                    'img': image_to_base64(img),
                    'mask_pred': image_to_base64(mask_pred),
                    'mask_gt': image_to_base64(mask_gt)
                })

        # Save CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'report.csv'), index=False)

    return render_template('index.html', results=results)

@app.route('/download')
def download():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.csv')
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)


