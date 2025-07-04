import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import json
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
from model import model

def run_fashion_pipeline(user_img_path, output_folder):
    # Paths
    BASE_PATH = 'static/dataset'

    GARMENT_CATEGORIES = {
        'tops': os.path.join(BASE_PATH, 'Garments/Tops/'),
        'bottoms': os.path.join(BASE_PATH, 'Garments/Bottom/'),
        'saree': os.path.join(BASE_PATH, 'Garments/Saaree/'),
        'onepiece': os.path.join(BASE_PATH, 'Garments/Onepiece/')
    }

    ACCESSORY_CATEGORIES = {
        'purse': os.path.join(BASE_PATH, 'Accessories/Purse/'),
        'bracelet': os.path.join(BASE_PATH, 'Accessories/Bracelet/'),
        'necklace': os.path.join(BASE_PATH, 'Accessories/Neclace/'),
        'earrings': os.path.join(BASE_PATH, 'Accessories/Earings/'),
        'glasses': os.path.join(BASE_PATH, 'Accessories/Glasses')
    }

    IMG_HEIGHT, IMG_WIDTH = 224, 224
    
    def load_and_preprocess(img_path):
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def extract_features(model, img_path):
        try:
            preprocessed = load_and_preprocess(img_path)
            features = model.predict(preprocessed, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def get_all_image_paths(base_folder):
        return [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    all_features = {}
    category_map = {}

    for category, path in {**GARMENT_CATEGORIES, **ACCESSORY_CATEGORIES}.items():
        for img_path in get_all_image_paths(path):
            feat = extract_features(model, img_path)
            if feat is not None:
                all_features[img_path] = feat
                category_map[img_path] = category

    user_feat = extract_features(model, user_img_path)

    best_match = None
    max_sim = -1
    for path, feat in all_features.items():
        category = category_map[path]
        if category in GARMENT_CATEGORIES:
            sim = cosine_similarity([user_feat], [feat])[0][0]
            if sim > max_sim:
                max_sim = sim
                best_match = category

    final_accessory_paths = {}
    for acc_type, folder in ACCESSORY_CATEGORIES.items():
        best_path, best_score = None, -1
        for path in get_all_image_paths(folder):
            sim = cosine_similarity([user_feat], [all_features[path]])[0][0]
            if sim > best_score:
                best_path = path
                best_score = sim
        final_accessory_paths[acc_type] = best_path

    user_img_pil = Image.open(user_img_path).convert("RGB")
    user_img_cv = cv2.cvtColor(np.array(user_img_pil), cv2.COLOR_RGB2BGR)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise Exception("No pose landmarks detected")

        def get_landmark_xy(idx):
            lm = results.pose_landmarks.landmark[idx]
            return (int(lm.x * user_img_cv.shape[1]), int(lm.y * user_img_cv.shape[0]))

        left_ear = get_landmark_xy(mp_pose.PoseLandmark.LEFT_EAR.value)
        right_ear = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_EAR.value)
        left_eye = get_landmark_xy(mp_pose.PoseLandmark.LEFT_EYE.value)
        right_eye = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_EYE.value)
        left_shoulder = get_landmark_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_landmark_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
        right_index = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_INDEX.value)
        left_ankle = get_landmark_xy(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_landmark_xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        def offset_ear_to_earlobe(ear, eye, shoulder):
            dy = int(0.15 * abs(shoulder[1] - eye[1]))
            return (ear[0], ear[1] + dy)

        centers = {
            "left_earlobe": offset_ear_to_earlobe(left_ear, left_eye, left_shoulder),
            "right_earlobe": offset_ear_to_earlobe(right_ear, right_eye, right_shoulder),
            "necklace": ((left_shoulder[0] + right_shoulder[0]) // 2,
                         (left_shoulder[1] + right_shoulder[1]) // 2 + int(0.25 * abs(((left_hip[1] + right_hip[1]) // 2) - ((left_shoulder[1] + right_shoulder[1]) // 2)))),
            "hand": right_index,
            "left_foot": left_ankle,
            "right_foot": right_ankle
        }

    def split_accessory_pair(img, mode='horizontal'):
        w, h = img.size
        if mode == 'horizontal':
            left = img.crop((0, 0, w // 2, h))
            right = img.crop((w // 2, 0, w, h))
        else:
            top = img.crop((0, 0, w, h // 2))
            bottom = img.crop((0, h // 2, w, h))
            left, right = top, bottom
        return left, right

    def mirror_image(img):
        return ImageOps.mirror(img)

    def paste_accessory(base_img, accessory_img, top_center, scale=1.0):
        acc_resized = accessory_img.resize(
            (int(accessory_img.width * scale), int(accessory_img.height * scale)),
            resample=Image.LANCZOS
        )
        x, y = top_center
        x -= acc_resized.width // 2
        base_img.alpha_composite(acc_resized, (x, y))

    # Load user image as RGBA
    user_img_rgba = Image.open(user_img_path).convert("RGBA")

    # Load accessories
    earring_img = Image.open(final_accessory_paths["earrings"]).convert("RGBA")
    left_earring, right_earring = split_accessory_pair(earring_img)
    earring_scale = 0.22
    paste_accessory(user_img_rgba, left_earring, centers["left_earlobe"], earring_scale)
    paste_accessory(user_img_rgba, right_earring, centers["right_earlobe"], earring_scale)

    # Purse
    purse_img = Image.open(final_accessory_paths["purse"]).convert("RGBA")
    paste_accessory(user_img_rgba, purse_img, centers["hand"], scale=0.45)

    # Footwear
    footwear_img = Image.open(final_accessory_paths["footwear"]).convert("RGBA")
    if footwear_img.width / footwear_img.height > 1.5:
        right_shoe_img, left_shoe_img = split_accessory_pair(footwear_img)
    else:
        right_shoe_img = footwear_img
        left_shoe_img = mirror_image(footwear_img)
    paste_accessory(user_img_rgba, left_shoe_img, centers["left_foot"], scale=0.4)
    paste_accessory(user_img_rgba, right_shoe_img, centers["right_foot"], scale=0.4)

    # Necklace
    necklace_img = Image.open(final_accessory_paths["necklace"]).convert("RGBA")
    paste_accessory(user_img_rgba, necklace_img, centers["necklace"], scale=0.4)

    result_path = os.path.join(output_folder, os.path.splitext(os.path.basename(user_img_path))[0] + '_final_overlay.png')
    user_img_rgba.save(result_path)
    return os.path.basename(result_path)
