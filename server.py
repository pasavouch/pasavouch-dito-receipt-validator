from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Setup base directory and template path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "template_dito_receipt_v1.jpg")

# Load reference image in grayscale
REF_IMG = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

if REF_IMG is None:
    raise RuntimeError("DITO template not found. Please ensure the file exists.")

# Threshold for template matching (0.0 to 1.0)
# 0.65 is a good balance for mobile screenshots
MATCH_THRESHOLD = 0.65

# Height tolerance for portrait receipts
# Allows slightly shorter images while still blocking landscape/history views
HEIGHT_TOLERANCE = 0.80

@app.route("/validate-format", methods=["POST"])
def validate_format():
    # Check if image part exists in request
    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        # Read the uploaded image from the request
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # Get dimensions to ensure template is not larger than the uploaded image
        h_ref, w_ref = REF_IMG.shape
        h_img, w_img = img.shape

        if h_img < int(h_ref * HEIGHT_TOLERANCE) or w_ref > w_img:
            # Reject if image is too short or too narrow
            return jsonify({"ok": False, "reason": "UPLOAD_TOO_SMALL"})

        # Perform Template Matching
        # This searches for the REF_IMG pattern inside the uploaded img
        res = cv2.matchTemplate(img, REF_IMG, cv2.TM_CCOEFF_NORMED)
        
        # Extract the highest similarity score found
        _, max_val, _, _ = cv2.minMaxLoc(res)
        score = round(float(max_val), 2)

        # Validate against our threshold
        if score < MATCH_THRESHOLD:
            return jsonify({
                "ok": False, 
                "reason": "FORMAT_MISMATCH",
                "similarity": score
            })

        # If successful, return the similarity score
        return jsonify({
            "ok": True,
            "similarity": score
        })

    except Exception as e:
        # Return system error for debugging
        return jsonify({
            "ok": False,
            "reason": "SYSTEM_ERROR",
            "msg": str(e)
        })

if __name__ == "__main__":
    # Run the server
    app.run(host="0.0.0.0", port=5000)
