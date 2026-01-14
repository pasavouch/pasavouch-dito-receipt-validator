from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_PATH = os.path.join(BASE_DIR, "template_dito_receipt_v1.jpg")
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

if REF_IMG is None:
    raise RuntimeError("DITO template not found")

ASPECT_TOL = 0.25
DIFF_LIMIT = 55
EDGE_LIMIT = 28
SSIM_THRESHOLD = 0.72

@app.route("/validate-format", methods=["POST"])
def validate_format():

    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        h_ref, w_ref = REF_IMG.shape
        h_img, w_img = img.shape

        ratio_ref = w_ref / h_ref
        ratio_img = w_img / h_img

        # landscape only
        if abs(ratio_ref - ratio_img) > ASPECT_TOL:
            return jsonify({"ok": False, "reason": "NOT_LANDSCAPE"})

        img_resized = cv2.resize(img, (w_ref, h_ref))

        ref_blur = cv2.GaussianBlur(REF_IMG, (5, 5), 0)
        img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # core receipt area (cropped allowed)
        y1, y2 = int(h_ref * 0.18), int(h_ref * 0.82)
        x1, x2 = int(w_ref * 0.10), int(w_ref * 0.90)

        ref_crop = ref_blur[y1:y2, x1:x2]
        img_crop = img_blur[y1:y2, x1:x2]

        # detect UI / overlay / drawings
        edges = cv2.Canny(img_crop, 80, 200)
        if edges.mean() > EDGE_LIMIT:
            return jsonify({"ok": False, "reason": "UI_OR_OVERLAY_DETECTED"})

        # heavy layout change (history list / multiple rows)
        diff = cv2.absdiff(ref_crop, img_crop).mean()
        if diff > DIFF_LIMIT:
            return jsonify({"ok": False, "reason": "MULTI_TRANSACTION_OR_HISTORY"})

        # structure similarity
        ref_edge = cv2.Canny(ref_crop, 80, 200)
        img_edge = cv2.Canny(img_crop, 80, 200)

        score, _ = ssim(ref_edge, img_edge, full=True)
        score = round(score, 2)

        if score < SSIM_THRESHOLD:
            return jsonify({
                "ok": False,
                "reason": "FORMAT_MISMATCH",
                "similarity": score
            })

        return jsonify({
            "ok": True,
            "similarity": score
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "reason": "SYSTEM_ERROR",
            "msg": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
