import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_features(img):
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

def compute_ssim(img1, img2):
    img1 = cv2.cvtColor(cv2.resize(img1,(224,224)), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.resize(img2,(224,224)), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score

def dominant_color_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1,3)
    return np.mean(pixels, axis=0)

# Load saved catalogue features
catalogue_features = np.load("features/catalogue_features.npy")
catalogue_names = np.load("features/catalogue_names.npy", allow_pickle=True)

# Load live image
live_img = cv2.imread("live/live_1.jpg")
live_feat = get_features(live_img)

# Find best match
sims = cosine_similarity(live_feat, catalogue_features)[0]
best_idx = np.argmax(sims)
best_score = sims[best_idx]
best_match_name = catalogue_names[best_idx]

print(f"Best Matched Design: {best_match_name}")
print(f"Similarity Score: {best_score:.3f}")

# Load best catalogue image
cat_img = cv2.imread("catalogue/" + best_match_name)

# Minute mismatch check
ssim_score = compute_ssim(live_img, cat_img)
print(f"SSIM Score: {ssim_score:.3f}")

# Stone colour difference
c1 = dominant_color_lab(cat_img)
c2 = dominant_color_lab(live_img)
color_diff = np.linalg.norm(c1 - c2)
print(f"Color Difference: {color_diff:.2f}")

# FINAL DECISION
if best_score < 0.85:
    print("❌ DESIGN MISMATCH")
elif ssim_score < 0.90:
    print("❌ MINUTE DESIGN MISMATCH")
elif color_diff > 10:
    print("❌ STONE COLOUR MISMATCH")
else:
    print("✅ MATCH OK")