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

        # Must be landscape (cropped transaction view)
        if w_img <= h_img:
            return jsonify({"ok": False, "reason": "INVALID_ORIENTATION"})

        # Minimum size check
        # Prevents icons, thumbnails, overly aggressive crops
        if w_img < 600 or h_img < 200:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_SMALL"})

        # Aspect ratio check
        # Cropped transaction views are wide but not extreme
        aspect_ratio = w_img / h_img
        if aspect_ratio < 2.0 or aspect_ratio > 7.0:
            return jsonify({"ok": False, "reason": "INVALID_LAYOUT"})

        # Passed format validation
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
