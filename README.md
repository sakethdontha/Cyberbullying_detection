# Cyberbullying Detection using Machine Learning

This project aims to detect cyberbullying in text data using machine learning models. The solution is trained on labeled datasets and focuses on identifying various types of cyberbullying (e.g., religion, age, ethnicity, gender, etc.).

---

## Repository Contents

- `cyberbullying-ml-project.ipynb`: Jupyter Notebook containing the full code for preprocessing, training, evaluation, and prediction.
- `README.md`: Project overview and instructions (this file).

---

## Objectives

- Build a model to classify whether a piece of text contains cyberbullying.
- Classify the type of bullying (e.g., religion-based, age-based).
- Evaluate performance using various metrics.
- Deploy a scalable solution to detect harmful language online.

---

## Technologies Used

- Python  
- Jupyter Notebook  
- Pandas, NumPy  
- Scikit-learn  
- Natural Language Toolkit (NLTK)  
- Matplotlib, Seaborn  

---

## Dataset

- Labeled dataset containing various cyberbullying types.
- Text data with corresponding class labels.
- Includes classes such as: `age`, `ethnicity`, `gender`, `not_cyberbullying`, `other_cyberbullying`, `religion`.

---

## Workflow Overview

1. **Data Preprocessing**:
   - Text cleaning (removing stopwords, special characters).
   - Label encoding and class balancing.

2. **Exploratory Data Analysis**:
   - Word cloud visualizations.
   - Class distribution analysis.

3. **Model Building**:
   - Train/Test split.
   - TF-IDF vectorization.
   - Logistic Regression and other models.

4. **Evaluation**:
   - Confusion Matrix.
   - Accuracy, Precision, Recall, F1 Score.

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/sakethdontha/Cyberbullying_detection.git
   cd Cyberbullying_detection
