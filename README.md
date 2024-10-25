# Cat vs. Dog Image Classifier with SVM

## Project Overview
This project tackles the classic image classification problem of distinguishing cats from dogs using a Support Vector Machine (SVM) model. Leveraging the Kaggle dataset, we preprocess images, reduce dimensionality with Principal Component Analysis (PCA), and evaluate the model’s accuracy through training and testing stages.

## Folder Structure
```
├── train/
│   ├── cat/
│   │   ├── cat.0.jpg, cat.1.jpg, ...
│   └── dog/
│       ├── dog.0.jpg, dog.1.jpg, ...
├── test/
│   ├── cat/
│   │   ├── cat_test57.jpg, ...
│   └── dog/
│       ├── dog_test80.jpg, ...
└── dog_cat_svm.py
```

## Key Steps
1. Data Loading & Preprocessing: Loads grayscale images, resized for model compatibility.
2. Dimensionality Reduction with PCA: Optimizes model performance by reducing feature count while retaining significant variance.
3. SVM Training & Testing: Trains the model with the processed data, validated by separate testing.
4. Performance Visualization: Includes plots to show explained variance and overall accuracy.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Place your images in the `train` and `test` folders under `cat` and `dog`.
2. Run the program:
   ```bash
   python dog_cat_svm.py
   ```
3. Link for the dataset - https://www.kaggle.com/c/dogs-vs-cats/data

The program outputs validation and test accuracy and displays variance analysis graphs for enhanced interpretability.
