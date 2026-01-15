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
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"ok": False, "reason": "IMAGE_READ_ERROR"})

        # Get image dimensions
        h_img, w_img = img.shape

        # Minimum size check
        # Blocks thumbnails and icons
        if w_img < 800 or h_img < 250:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_SMALL"})

        # Aspect ratio check
        # Single transaction view is wide, not tall
        aspect_ratio = w_img / h_img

        if aspect_ratio < 2.5 or aspect_ratio > 6.5:
            return jsonify({"ok": False, "reason": "INVALID_LAYOUT"})

        # If all checks pass, allow OCR stage
        return jsonify({
            "ok": True,
            "width": w_img,
            "height": h_img,
            "ratio": round(aspect_ratio, 2)
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
