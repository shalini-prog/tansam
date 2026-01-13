import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

catalogue_dir = "catalogue/"
feature_list = []
name_list = []

for img_name in os.listdir(catalogue_dir):
    img_path = os.path.join(catalogue_dir, img_name)
    feat = get_features(img_path)
    feature_list.append(feat)
    name_list.append(img_name)

np.save("features/catalogue_features.npy", np.vstack(feature_list))
np.save("features/catalogue_names.npy", np.array(name_list))

print("Catalogue features saved successfully!")