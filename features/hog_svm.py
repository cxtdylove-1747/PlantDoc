import os
from pathlib import Path
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def load_images_and_labels(root_dir: str, image_size: int = 128):
    root = Path(root_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X = []
    y = []
    for cls in classes:
        cls_dir = root / cls
        for img_path in tqdm(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")), desc=f"Loading {cls}"):
            img = imread(str(img_path))
            if img.ndim == 2:
                # gray to rgb
                img = np.stack([img]*3, axis=-1)
            img_resized = resize(img, (image_size, image_size), anti_aliasing=True)
            X.append(img_resized)
            y.append(class_to_idx[cls])
    X = np.array(X)
    y = np.array(y)
    return X, y, classes

def extract_hog_features(X: np.ndarray, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    feats = []
    for img in tqdm(X, desc="Extracting HOG"):
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        feature = hog(
            gray, orientations=9, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, block_norm="L2-Hys"
        )
        feats.append(feature)
    return np.array(feats)

def main():
    train_dir = "PlantDoc/TRAIN"
    test_dir = "PlantDoc/TEST"
    image_size = 128

    X_train, y_train, classes = load_images_and_labels(train_dir, image_size=image_size)
    X_test, y_test, _ = load_images_and_labels(test_dir, image_size=image_size)

    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train_hog, y_train)
    y_pred = clf.predict(X_test_hog)

    acc = accuracy_score(y_test, y_pred)
    print("HOG+SVM Test Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=classes))

if __name__ == "__main__":
    main()