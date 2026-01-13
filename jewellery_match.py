import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===========================
# LOAD BETTER FEATURE MODEL
# ===========================
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def get_features(img):
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    feat = model.predict(img)
    return feat / np.linalg.norm(feat)   # L2 normalize

# ===========================
# BACKGROUND REMOVAL
# ===========================
def remove_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# ===========================
# MASKED SSIM (MINUTE MISMATCH)
# ===========================
def compute_ssim_masked(img1, img2):
    img1 = cv2.cvtColor(cv2.resize(img1, (224,224)), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.resize(img2, (224,224)), cv2.COLOR_BGR2GRAY)

    _, mask1 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY_INV)
    _, mask2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY_INV)

    img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    img2 = cv2.bitwise_and(img2, img2, mask=mask2)

    score, _ = ssim(img1, img2, full=True)
    return score

# ===========================
# BETTER STONE COLOR CHECK
# ===========================
def color_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0,1], None, [50,60], [0,180, 0,256])
    cv2.normalize(hist, hist)
    return hist

# =================================================
# STEP 1: EXTRACT & SAVE CATALOGUE FEATURES
# Run this once, then comment it out
# =================================================

def build_catalogue_features():
    catalogue_dir = "catalogue/"
    feature_list = []
    name_list = []

    for img_name in os.listdir(catalogue_dir):
        img_path = os.path.join(catalogue_dir, img_name)
        img = cv2.imread(img_path)

        img = remove_background(img)
        feat = get_features(img)

        feature_list.append(feat)
        name_list.append(img_name)

    os.makedirs("features", exist_ok=True)
    np.save("features/catalogue_features.npy", np.vstack(feature_list))
    np.save("features/catalogue_names.npy", np.array(name_list))

    print("✅ Catalogue features saved successfully!")

# Uncomment only first time
build_catalogue_features()

# =================================================
# STEP 2: LIVE IMAGE MATCHING SYSTEM
# =================================================

# Load saved features
catalogue_features = np.load("features/catalogue_features.npy")
catalogue_names = np.load("features/catalogue_names.npy", allow_pickle=True)

# Load live image
live_img = cv2.imread("live/live_1.jpg")
live_img = remove_background(live_img)
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
cat_img = remove_background(cat_img)

# Minute mismatch check
ssim_score = compute_ssim_masked(live_img, cat_img)
print(f"SSIM Score: {ssim_score:.3f}")

# Stone colour check
h1 = color_histogram(cat_img)
h2 = color_histogram(live_img)
color_similarity = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
print(f"Color Similarity: {color_similarity:.3f}")

# ===========================
# FINAL DECISION LOGIC
# ===========================
if best_score < 0.75:
    print("❌ DESIGN MISMATCH")
elif ssim_score < 0.90:
    print("❌ MINUTE DESIGN MISMATCH")
elif color_similarity < 0.85:
    print("❌ STONE COLOUR MISMATCH")
else:
    print("✅ MATCH OK")
