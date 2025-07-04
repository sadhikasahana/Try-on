from flask import Flask, render_template, request, jsonify, send_from_directory
import os, uuid, json
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from model import model
from operator import itemgetter

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 224, 224

# ✅ Load ResNet50 once

# ✅ Dataset paths
BASE_PATH = "static/dataset"

GARMENT_CATEGORIES = {
    'tops': os.path.join(BASE_PATH, 'Garments/Tops/'),
    'bottoms': os.path.join(BASE_PATH, 'Garments/Bottom/'),
    'saree': os.path.join(BASE_PATH, 'Garments/Saaree/'),
    'onepiece': os.path.join(BASE_PATH, 'Garments/Onepiece/')
}
ACCESSORY_CATEGORIES = {
    'Purse': os.path.join(BASE_PATH, 'Accessories/Purse/'),
    'Bracelet': os.path.join(BASE_PATH, 'Accessories/Bracelet/'),
    'Necklace': os.path.join(BASE_PATH, 'Accessories/Neclace/'),
    'Earrings': os.path.join(BASE_PATH, 'Accessories/Earings/'),
    'Goggles': os.path.join(BASE_PATH, 'Accessories/Glasses/')
}

def get_all_image_paths(base_folder):
    return [os.path.join(base_folder, f) for f in os.listdir(base_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(img_path):
    try:
        preprocessed = load_and_preprocess(img_path)
        features = model.predict(preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
    
def split_accessory_pair(img, mode='horizontal'):
    """Split paired accessory into left and right."""
    w, h = img.size
    if mode == 'horizontal':
        left = img.crop((0, 0, w // 2, h))
        right = img.crop((w // 2, 0, w, h))
    else:
        top = img.crop((0, 0, w, h // 2))
        bottom = img.crop((0, h // 2, w, h))
        left, right = top, bottom
    return left, right

@app.route("/")
def index():
    return render_template("index.html")

# ✅ 1. Upload user image
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({"user_image": filepath})

# ✅ 2. Recommend accessories
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_img_path = data.get("user_image")

    user_feat = extract_features(user_img_path)
    all_features = {}
    category_map = {}

    # Extract all features from garment + accessory dataset
    for category, path in {**GARMENT_CATEGORIES, **ACCESSORY_CATEGORIES}.items():
        for img_path in get_all_image_paths(path):
            feat = extract_features(img_path)
            if feat is not None:
                all_features[img_path] = feat
                category_map[img_path] = category

    # Detect best garment category
    best_match = None
    max_sim = -1
    for path, feat in all_features.items():
        if category_map[path] in GARMENT_CATEGORIES:
            sim = cosine_similarity([user_feat], [feat])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_match = category_map[path]
    
    #For each accessory type, get top-4
    top_k = 4
    acc_type_top_paths = {}
    for acc_type, folder in ACCESSORY_CATEGORIES.items():
        scored_paths = []
        for path in get_all_image_paths(folder):
            sim = cosine_similarity([user_feat], [all_features[path]])[0][0]
            scored_paths.append((path, sim))
        scored_paths.sort(key=itemgetter(1), reverse=True)
        acc_type_top_paths[acc_type] = [p for p, s in scored_paths[:top_k]]

    

    # Recommend 1 best matching accessory of each type
    #final_accessory_paths = {}
    #for acc_type, folder in ACCESSORY_CATEGORIES.items():
    #    best_path, best_score = None, -1
    #    for path in get_all_image_paths(folder):
    #        sim = cosine_similarity([user_feat], [all_features[path]])[0][0]
    #        if sim > best_score:
    #            best_path = path
    #            best_score = sim
    #    final_accessory_paths[acc_type] = best_path
    
    #create 4 rows of recommendations
    recommendation_rows = []
    for i in range(top_k):
        row_accessories = {}
        for acc_type in ACCESSORY_CATEGORIES:
            paths = acc_type_top_paths.get(acc_type, [])
            if len(paths) > i:
                row_accessories[acc_type] = paths[i]
            elif paths:
                row_accessories[acc_type] = paths[0]
            else:
                row_accessories[acc_type] = None
        recommendation_rows.append({
            "accessories": row_accessories
        })
    # Save recommendation results
    result = {
        "user_image": user_img_path,
        "garment": best_match,
        "accessories": acc_type_top_paths
    }
    with open("recommendation_result.json", "w") as f:
        json.dump(result, f)

    return jsonify(result)

# ✅ 3. Apply recommended accessories using pose + overlay
@app.route("/apply", methods=["POST"])
def apply():
    #get which set user selected
    data = request.get_json()
    selected_accessories = data.get("selected_accessories", {})


    with open("recommendation_result.json", "r") as f:
        rec_data = json.load(f)

    user_img_path = rec_data["user_image"]
    
    user_img_pil = Image.open(user_img_path).convert("RGBA")
    user_img_cv = cv2.cvtColor(np.array(user_img_pil), cv2.COLOR_RGBA2BGR)

    
    # Load user image
    #user_img_pil = Image.open(user_img_path).convert("RGBA")
    #user_img_cv = cv2.cvtColor(np.array(user_img_pil), cv2.COLOR_RGBA2BGR)
    earring_img = Image.open(selected_accessories["Earrings"]).convert("RGBA")
    necklace_img = Image.open(selected_accessories["Necklace"]).convert("RGBA")
    purse_img = Image.open(selected_accessories["Purse"]).convert("RGBA")
    #footwear_img = Image.open(selected_accessories["Footwear"]).convert("RGBA")
    goggle_img = Image.open(selected_accessories["Goggles"]).convert("RGBA")
    bracelet_img = Image.open(selected_accessories["Bracelet"]).convert("RGBA")


    # Detect pose landmarks
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return jsonify({"error": "No pose landmarks detected."})

        def get_xy(idx):
            lm = results.pose_landmarks.landmark[idx]
            return (int(lm.x * user_img_cv.shape[1]), int(lm.y * user_img_cv.shape[0]))
        
        # ✅ Get key body landmarks
        left_ear = get_xy(mp_pose.PoseLandmark.LEFT_EAR.value)
        right_ear = get_xy(mp_pose.PoseLandmark.RIGHT_EAR.value)
        left_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_eye = get_xy(mp_pose.PoseLandmark.LEFT_EYE.value)
        right_eye = get_xy(mp_pose.PoseLandmark.RIGHT_EYE.value)
        left_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_index = get_xy(mp_pose.PoseLandmark.RIGHT_INDEX.value)
        left_ankle = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        left_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_eye = get_xy(mp_pose.PoseLandmark.LEFT_EYE.value)
        right_eye = get_xy(mp_pose.PoseLandmark.RIGHT_EYE.value)
        right_wrist = get_xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)


        # ✅ Estimate earlobe by shifting ears downward by 15% of distance from eye to shoulder
        def offset_ear_to_earlobe(ear, eye, shoulder):
            dy = int(0.15 * abs(shoulder[1] - eye[1]))
            return (ear[0], ear[1] + dy)

        left_earlobe = offset_ear_to_earlobe(left_ear, left_eye, left_shoulder)
        right_earlobe = offset_ear_to_earlobe(right_ear, right_eye, right_shoulder)

        # ✅ Estimate necklace point: mid-point between shoulders shifted down 25% of shoulder-hip distance
        neck_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        neck_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
        shoulder_hip_dist = abs(((left_hip[1] + right_hip[1]) // 2) - neck_center_y)
        necklace_y = neck_center_y + int(0.25 * shoulder_hip_dist)
        necklace = (neck_center_x, necklace_y)
        jaw_left = right_ear   # ← yes, flipped due to camera/mirror logic
        jaw_right = left_ear



        centers = {
            "left_ear": get_xy(mp_pose.PoseLandmark.LEFT_EAR.value),
            "right_ear": get_xy(mp_pose.PoseLandmark.RIGHT_EAR.value),
            "right_index": get_xy(mp_pose.PoseLandmark.RIGHT_INDEX.value),
            "left_ankle": get_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value),
            "right_ankle": get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            "left_earlobe": left_earlobe,
            "right_earlobe": right_earlobe,
            "necklace": necklace,
            "hand": right_index,
            "left_foot": left_ankle,
            "right_foot": right_ankle,
            "left_shoulder": left_shoulder,      # ✅ Add this
            "right_shoulder": right_shoulder,
            "jaw_left" : jaw_left,
            "jaw_right" : jaw_right,
            "left_eye" : left_eye,
            "right_eye" : right_eye,
            "right_wrist" : right_wrist
        }
    
    
    # Load accessories
    #earring_img = Image.open(accessories["earrings"]).convert("RGBA")
    #necklace_img = Image.open(accessories["necklace"]).convert("RGBA")
    #purse_img = Image.open(accessories["purse"]).convert("RGBA")
    #footwear_img = Image.open(accessories["footwear"]).convert("RGBA")



    # Paste example earrings (you can use refined logic from your notebook)
    #earring_scaled = earring_img.resize((50, 50))
    #user_img_pil.alpha_composite(earring_scaled, (centers["left_ear"][0] - 25, centers["left_ear"][1]))
    #user_img_pil.alpha_composite(earring_scaled, (centers["right_ear"][0] - 25, centers["right_ear"][1]))

    left_earring, right_earring = split_accessory_pair(earring_img, mode='horizontal')
    def crop_top(img, pixels=0):
        w, h = img.size
        return img.crop((0, pixels, w, h))
    
    left_earring = crop_top(left_earring, pixels=0)
    right_earring = crop_top(right_earring, pixels=91)

    #scale earrings
    earring_scale = 0.18
    left_resized = left_earring.resize((
        int(left_earring.width * earring_scale), int(left_earring.height * earring_scale))
    )
    right_resized = right_earring.resize(
        (int(right_earring.width * earring_scale), int(right_earring.height * earring_scale))
    )

    left_shift = 10
    right_shift = -10
    vertical_shift = -5

    left_pos = (
        centers["left_earlobe"][0] - left_resized.width // 2 + left_shift,
        centers["left_earlobe"][1] + vertical_shift
    )
    right_pos = (
        centers["right_earlobe"][0] - right_resized.width // 2 + right_shift,
        centers["right_earlobe"][1] +vertical_shift
    )
    
    #paste earrings
    user_img_pil.alpha_composite(left_resized, left_pos)
    user_img_pil.alpha_composite(right_resized, right_pos)


    #Paste Bracelet
    wrist_width = abs(centers["right_wrist"][0] - centers["right_index"][0])
    target_width = int(1.2 * wrist_width)
    scale_factor = target_width / bracelet_img.width
    bracelet_resized = bracelet_img.resize(
        (target_width, int(bracelet_img.height * scale_factor)),
        resample = Image.LANCZOS
    )

    bracelet_pos = (
        centers["right_wrist"][0] - bracelet_resized.width // 2,
        centers["right_wrist"][1] - bracelet_resized.height // 2
    )

    user_img_pil.alpha_composite(bracelet_resized, bracelet_pos)

        



    # Paste example purse
    #purse_resized = purse_img.resize((100, 100))
    #user_img_pil.alpha_composite(purse_resized, (centers["right_index"][0] - 50, centers["right_index"][1]))
    purse_shift_y = 30
    purse_pos = (
        centers["hand"][0] - int(purse_img.width * 0.45) // 2,
        centers["hand"][1] + purse_shift_y    
    )

    purse_resized = purse_img.resize(
        (int(purse_img.width * 0.45), int(purse_img.height * 0.45 )),
        resample = Image.LANCZOS
    )
    user_img_pil.alpha_composite(purse_resized, purse_pos)

    # Paste footwear (left and right)
    #shoe_resized = footwear_img.resize((80, 80))
    #user_img_pil.alpha_composite(shoe_resized, (centers["left_ankle"][0] - 40, centers["left_ankle"][1]))
    #user_img_pil.alpha_composite(shoe_resized, (centers["right_ankle"][0] - 40, centers["right_ankle"][1]))

    # Paste necklace (basic center below ears)
    #necklace_resized = necklace_img.resize((120, 70))
    #neck_x = (centers["left_ear"][0] + centers["right_ear"][0]) // 2 - 60
    #neck_y = (centers["left_ear"][1] + centers["right_ear"][1]) // 2 + 80
    #user_img_pil.alpha_composite(necklace_resized, (neck_x, neck_y))
    def crop_top_padding(img, alpha_tresh=10):
        np_img = np.array(img)
        alpha = np_img[:, :, 3]
        for row in range(alpha.shape[0]):
            if np.any(alpha[row] > alpha_tresh):
                return img.crop((0, row, img.width, img.height))
        return img
    
    necklace_img = crop_top_padding(necklace_img)
    
    chin_width = abs(right_ear[0] - left_ear[0])
    neck_center_x = (jaw_right[0] + jaw_left[0]) // 2
    collarbone_y = (jaw_right[1] + jaw_left[1]) // 2
    target_width = int(1.40 * chin_width)

    scale_factor = target_width / necklace_img.width
    resized_necklace = necklace_img.resize(
        (target_width, int(necklace_img.height * scale_factor)),
        resample=Image.LANCZOS
    )

    rotated_necklace = resized_necklace.rotate(2, expand=True)
    necklace_y = collarbone_y + int(0.45 * chin_width)

    paste_x = neck_center_x - rotated_necklace.width // 2
    paste_y = necklace_y + int(0.2 * chin_width)

    user_img_pil.alpha_composite(rotated_necklace, (paste_x, paste_y)) 

    # Save result
    result_filename = str(uuid.uuid4()) + ".png"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    user_img_pil.save(result_path)

    return jsonify({"result_image": "/" + result_path.replace("\\", "/")})

# Serve uploaded/result images
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
