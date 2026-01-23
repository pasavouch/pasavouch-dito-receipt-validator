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

        # Orientation check: Must be Portrait (height > width)
        # This blocks landscape or wide-cropped transaction views
        if h_img <= w_img:
            return jsonify({"ok": False, "reason": "NOT_PORTRAIT_ORIENTATION"})

        # Minimum size check for full-screen screenshots
        # Ensures the image is high-resolution enough for verification
        if h_img < 800 or w_img < 300:
            return jsonify({"ok": False, "reason": "IMAGE_TOO_SMALL"})

        # Aspect ratio check for mobile screens (Width / Height)
        # Typical portrait mobile screens are between 0.45 and 0.65
        aspect_ratio = w_img / h_img
        if aspect_ratio > 0.8: 
            # If ratio > 0.8, the image is too "square" to be a standard portrait screenshot
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
    # Run the server
    app.run(host="0.0.0.0", port=5000)
