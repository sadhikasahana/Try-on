from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
from io import BytesIO
import os
import uuid
import numpy as np
import cv2
import math
import mediapipe as mp

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def detect_person_shoulder_points(image):
    image_np = np.array(image.convert("RGB"))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = pose.process(image_np)

    if results.pose_landmarks:
        left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        h, w, _ = image_np.shape
        left_px = (int(left.x * w) - 8, int(left.y * h))
        right_px = (int(right.x * w), int(right.y * h))
        return left_px, right_px
    return None, None

def detect_garment_shoulder_points(image, mode="normal"):
    img_np = np.array(image.convert("RGBA"))
    alpha = img_np[:, :, 3]
    mask = alpha > 0
    binary_mask = np.uint8(mask * 255)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    top_points = sorted(contour, key=lambda pt: pt[0][1])[:20]
    top_points_sorted = sorted(top_points, key=lambda pt: pt[0][0])
    left = tuple(top_points_sorted[0][0])
    right = tuple(top_points_sorted[-1][0])

    if mode == "puffed":
        offset_y, offset_x = 53, 52
    else:
        offset_y, offset_x = 34, 8

    left_shoulder = (left[0] - offset_x, left[1] + offset_y)
    right_shoulder = (right[0] + offset_x, right[1] + offset_y)

    return left_shoulder, right_shoulder

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommendation")
def recommendation():
    return render_template("frontend/index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = str(uuid.uuid4()) + ".png"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return jsonify({"filepath": path})

@app.route("/try-on", methods=["POST"]) 
def try_on():
    data = request.get_json()
    person_path = data.get("user_image")
    garment_path = data.get("dress_image")
    mode = data.get("mode", "normal")  # Default to "normal" if not provided

    try:
        person_img = Image.open(person_path).convert("RGB").resize((512, 768))
        garment_img = Image.open(garment_path).convert("RGBA")

        left_person, right_person = detect_person_shoulder_points(person_img)
        left_garment, right_garment = detect_garment_shoulder_points(garment_img, mode=mode)

        if not all([left_person, right_person, left_garment, right_garment]):
            return jsonify({"success": False, "error": "Could not detect shoulder points"})

        person_dist = calculate_distance(left_person, right_person)
        garment_dist = calculate_distance(left_garment, right_garment)
        scale_x = person_dist / garment_dist

        person_height = person_img.height
        garment_height = garment_img.height
        scale_y = person_height / garment_height * 0.55

        scale_factor = max(scale_x, scale_y)
        print("Scale Factor:", scale_factor)
        new_w = int(garment_img.width * scale_factor)
        new_h = int(garment_img.height * scale_factor)
        resized_garment = garment_img.resize((new_w, new_h), Image.LANCZOS)

        left_g_resized = (int(left_garment[0] * scale_factor), int(left_garment[1] * scale_factor))
        right_g_resized = (int(right_garment[0] * scale_factor), int(right_garment[1] * scale_factor))

        def center(p1, p2):
            return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        offset_x = center(left_person, right_person)[0] - center(left_g_resized, right_g_resized)[0]
        offset_y = center(left_person, right_person)[1] - center(left_g_resized, right_g_resized)[1] 

        result = Image.new("RGBA", person_img.size)
        result.paste(person_img.convert("RGBA"), (0, 0))
        result.paste(resized_garment, (offset_x, offset_y), resized_garment)

        result_filename = str(uuid.uuid4()) + ".png"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result.save(result_path)

        return jsonify({"success": True, "result_image": "/" + result_path.replace("\\", "/")})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)


