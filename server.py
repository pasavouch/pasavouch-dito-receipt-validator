from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/validate-format", methods=["POST"])
def validate_format():
    # Check if image part exists in request
    if "image" not in request.files:
        return jsonify({"ok": False, "reason": "NO_IMAGE"})

    try:
        # Read the uploaded image from the request
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        # Use IMREAD_COLOR to better handle high-res mobile metadata
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # Get image dimensions
        h_img, w_img = img.shape[:2]

        # 1. Orientation check: Height must be greater than Width
        # Added tolerance for mobile devices with odd native resolutions
        if h_img < w_img:
            return jsonify({"ok": False, "reason": "NOT_PORTRAIT_ORIENTATION"})

        # 2. Minimum size check: Validated for small or low-end devices
        # Ensures enough pixels for OCR to read Transaction Numbers
        if h_img < 600 or w_img < 250:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_SMALL"})

        # 3. Aspect ratio check: Increased limit to 1.0
        # Prevents rejection of square-ish shots from Redmi/Tablets
        aspect_ratio = w_img / h_img
        if aspect_ratio > 1.0: 
            return jsonify({"ok": False, "reason": "INVALID_PORTRAIT_RATIO"})

        # Passed portrait format validation
        return jsonify({
            "ok": True,
            "width": w_img,
            "height": h_img,
            "ratio": round(aspect_ratio, 2),
            "mode": "PORTRAIT"
        })

    except Exception as e:
        # Return system error for debugging
        return jsonify({
            "ok": False,
            "reason": "SYSTEM_ERROR",
            "msg": str(e)
        })

if __name__ == "__main__":
    # Run the server on port 5000
    app.run(host="0.0.0.0", port=5000)
