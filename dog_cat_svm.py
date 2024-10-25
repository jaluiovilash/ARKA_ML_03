import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

IMG_SIZE = 64  # Size to resize images to (IMG_SIZE x IMG_SIZE)

# Directories
train_dir = "train"
test_dir = "test"

# Load and preprocess images
def load_images_from_folder(folder):
    images, labels = [], []
    for label, subfolder in enumerate(['cat', 'dog']):  # cat=0, dog=1
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)

print("Loading training data...")
X, y = load_images_from_folder(train_dir)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale and apply PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# PCA to reduce dimensionality
pca = PCA(n_components=100)  # Using 100 components for a small dataset
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

# Train the SVM
print("Training the SVM model...")
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate on the validation set
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred) * 100
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Confusion Matrix for validation set
plt.figure(figsize=(5, 5))
cm = confusion_matrix(y_val, y_val_pred)
ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"]).plot()
plt.title("Validation Confusion Matrix")
plt.show()

# Testing with new images
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    img = pca.transform(img)
    return img

# Test individual images
cat_test_path = 'test/cat/cat_test57.jpg'
dog_test_path = 'test/dog/dog_test80.jpg'

print("\nTesting individual images...")
for img_path, label in zip([cat_test_path, dog_test_path], ["Cat", "Dog"]):
    img_processed = load_and_preprocess_image(img_path)
    if img_processed is not None:
        prediction = svm_model.predict(img_processed)
        predicted_label = "Cat" if prediction[0] == 0 else "Dog"
        print(f"Image '{os.path.basename(img_path)}' is predicted as: {predicted_label} (Actual: {label})")

# Load and test on entire test dataset
print("\nLoading and testing on entire test dataset...")
X_test, y_test = load_images_from_folder(test_dir)
X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)

y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion Matrix for test set
plt.figure(figsize=(5, 5))
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"]).plot()
plt.title("Test Confusion Matrix")
plt.show()
