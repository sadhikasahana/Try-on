# 👗 Virtual Garment Try-On Web App

This is a Flask-based web application that allows users to try on virtual clothes (tops only) by uploading their own full-length image. It uses **MediaPipe** for pose detection, detects shoulder points, and overlays the selected garment image scaled and aligned to fit the user’s body.

---
## 🚀 Steps

- Select from a gallery of dresses with color options.
- Upload your own full-length photo.
- Automatic shoulder alignment using pose estimation.
- Resizes and places garment image for realistic virtual try-on.

---
# Recommendation System

This is a Flask-based web application that recommends and virtually applies fashion accessories to a user's uploaded image. It uses ResNet50 for image feature extraction, compares the user's features with a curated dataset, and recommends the top matching accessories across categories like earrings, necklace, purse, bracelet, and goggles. MediaPipe is used for pose detection to accurately place each accessory on the user’s body in a realistic and scaled manner.

---
## 🚀 Steps
- Takes a user image

- Extracts features using ResNet50

- Recommends accessories (necklace, earrings, purse, etc.) based on image similarity

 -Applies those accessories using MediaPipe pose landmarks

- Shows a virtual try-on output
