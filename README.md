# DETEKSI-URL-PHISHING-MENGGUNAKAN-METODE-NATURAL-LANGUAGE-PROCESSING-DAN-ALGORITMA-RANDOM-FOREST

# Phishing URL Detection using Random Forest and NLP

This project implements a **machine learning model** to detect phishing URLs using a combination of **custom URL-based features** and **text-based TF-IDF features**. The model is trained using the **Random Forest Classifier** and evaluated with metrics such as **Accuracy, Precision, Recall, and F1-Score**.

---

## 📂 Project Structure
- `model_80-20.ipynb` → Jupyter Notebook containing preprocessing, feature extraction, model training, and evaluation.
- `new_data_urls.csv` → Dataset file used for training and testing (phishing vs legitimate URLs).

---

## 📊 Dataset
The dataset used in this project is available on Kaggle:

🔗 [Phishing and Legitimate URLs Dataset - Kaggle](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls)

### How to get the dataset:
1. Create or log in to your [Kaggle account](https://www.kaggle.com/).
2. Go to the dataset page: [Phishing and Legitimate URLs](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls).
3. Click **Download** to get the dataset in `.zip` format.
4. Extract the dataset and rename the CSV file into `new_data_urls.csv`.
5. Upload `new_data_urls.csv` into your Google Colab environment or local project directory.

---

## 🚀 Running the Project on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the following files into your Colab session:
   - `model_80-20.ipynb`
   - `new_data_urls.csv`
3. Make sure the CSV file and the notebook are in the same working directory.

### Steps to Run:
- Open `model_80-20.ipynb` in Google Colab.
- Run all cells sequentially:
  - **Data Loading** → Reads `new_data_urls.csv`.
  - **Preprocessing & Feature Extraction** → Extracts custom features (URL length, subdomains, digits, etc.) and TF-IDF features.
  - **Model Training** → Trains the Random Forest classifier with Stratified K-Fold Cross Validation (80:20 split).
  - **Evaluation** → Displays confusion matrix, accuracy, precision, recall, and F1-score.

Example code snippet to verify dataset load in Colab:
```python
import pandas as pd

# Load dataset
df = pd.read_csv("new_data_urls.csv")
print(df.head())
print("Dataset shape:", df.shape)
