# import os
# import cv2  # OpenCV to read and preprocess images
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm

# # Set up directories
# data_dir = r"C:\Study\Web Dev\Internship\ARKA\TASK_3\train"  # Update with the actual path
# categories = ["cat", "dog"]  # Two categories (classes)

# # Image size for resizing (as SVMs don't support large image sizes well)
# img_size = 64

# # Function to load and preprocess images
# def load_images(data_dir, categories, img_size):
#     data = []
#     labels = []

#     for category in categories:
#         path = os.path.join(data_dir, category)
#         class_num = categories.index(category)  # Assigning labels as 0 for 'cat' and 1 for 'dog'

#         for img in tqdm(os.listdir(path)):  # Iterate through all images
#             try:
#                 img_path = os.path.join(path, img)
#                 img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
#                 resized_img = cv2.resize(img_array, (img_size, img_size))  # Resize to uniform size
#                 data.append(resized_img.flatten())  # Flatten image (convert 2D to 1D)
#                 labels.append(class_num)
#             except Exception as e:
#                 pass

#     return np.array(data), np.array(labels)

# # Load the dataset
# print("Loading images...")
# X, y = load_images(data_dir, categories, img_size)

# print("Images loaded!")

# # Encode labels (not necessary, but good practice if labels are not 0 and 1)
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# # Split dataset into training and testing sets
# print("Splitting dataset into train and test sets...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Data split done!")

# # Initialize and train the SVM model
# print("Training SVM model...")
# svm_model = SVC(kernel='linear')  # Linear kernel works well for this task
# svm_model.fit(X_train, y_train)

# print("SVM model trained!")

# # Make predictions on the test set
# print("Making predictions on test data...")
# y_pred = svm_model.predict(X_test)
# print("Predictions complete!")

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")


import os
import cv2  # OpenCV to read and preprocess images
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up directories
data_dir = r"C:\Study\Web Dev\Internship\ARKA\TASK_3\train"  # Update with the actual path
test_dir = r"C:\Study\Web Dev\Internship\ARKA\TASK_3\test"  # Path to new test images
categories = ["cat", "dog"]  # Two categories (classes)

# Image size for resizing
img_size = 64

# Function to load and preprocess images
def load_images(data_dir, categories, img_size):
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # Assigning labels as 0 for 'cat' and 1 for 'dog'

        for img in tqdm(os.listdir(path), desc=f"Loading {category} images"):  # Iterate through all images
            img_path = os.path.join(path, img)
            try:
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                resized_img = cv2.resize(img_array, (img_size, img_size))  # Resize to uniform size
                data.append(resized_img.flatten())  # Flatten image (convert 2D to 1D)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

# Function to visualize predictions
def visualize_predictions(images, true_labels, predicted_labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].reshape(img_size, img_size), cmap='gray')
        plt.title(f'True: {"Dog" if true_labels[i] == 1 else "Cat"}\nPred: {"Dog" if predicted_labels[i] == 1 else "Cat"}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load the dataset
print("Loading images...")
X, y = load_images(data_dir, categories, img_size)
print("Images loaded!")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split done!")

# Initialize and train the SVM model
print("Training SVM model...")
svm_model = SVC(kernel='linear')  # Linear kernel works well for this task
svm_model.fit(X_train, y_train)
print("SVM model trained!")

# Make predictions on the test set
print("Making predictions on test data...")
y_pred = svm_model.predict(X_test)
print("Predictions complete!")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Function to test new images
def test_new_images(test_dir, categories, img_size, model):
    test_data = []
    true_labels = []

    for category in categories:
        path = os.path.join(test_dir, category)
        class_num = categories.index(category)

        for img in tqdm(os.listdir(path), desc=f"Loading {category} test images"):
            img_path = os.path.join(path, img)
            try:
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img_array, (img_size, img_size))
                test_data.append(resized_img.flatten())
                true_labels.append(class_num)
            except Exception as e:
                print(f"Error loading test image {img_path}: {e}")

    X_new = np.array(test_data)
    y_true = np.array(true_labels)

    # Predict using the trained model
    y_pred_new = model.predict(X_new)

    # Print accuracy for new images
    accuracy = accuracy_score(y_true, y_pred_new)
    print(f"Test Accuracy on New Images: {accuracy * 100:.2f}%")

    # Visualize predictions
    visualize_predictions(X_new, y_true, y_pred_new)

# Test the model with new images
print("Testing with new images...")
test_new_images(test_dir, categories, img_size, svm_model)
