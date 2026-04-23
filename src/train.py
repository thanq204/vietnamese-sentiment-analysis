print("🚀 START TRAINING...")

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle

from src.preprocess import clean_text

# ===== LOAD DATA =====
df = pd.read_csv("data/dulieu_huanluyen.csv", encoding="latin-1")

# ===== RENAME =====
df = df.rename(columns={
    "comments": "text",
    "flag": "label"
})

# ===== CLEAN TEXT =====
def clean_advanced(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # normalize slang
    slang = {
        "ko": "không",
        "k": "không",
        "hok": "không",
        "cx": "cũng",
        "vs": "với"
    }
    for k, v in slang.items():
        text = text.replace(k, v)

    # remove link
    text = re.sub(r"http\S+", "", text)

    # remove number
    text = re.sub(r"\d+", "", text)

    # remove special char
    text = re.sub(r"[^\w\s]", " ", text)

    # remove extra space
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["text"] = df["text"].apply(clean_advanced)

# ===== REMOVE BAD DATA =====
df = df[df["text"].str.strip() != ""]
df = df[df["text"].str.len() > 10]        # bỏ câu quá ngắn
df = df.drop_duplicates()                # bỏ trùng

# 👉 OPTIONAL: bỏ neutral để tăng mạnh accuracy
df = df[df["label"] != 2]

print("📊 Label distribution:")
print(df["label"].value_counts())

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ===== VECTORIZE =====
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

print("📦 Vectorizing...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== TRAIN =====
print("🤖 Training Logistic...")
lr_model = LogisticRegression(
    C=2.0,
    max_iter=200,
    class_weight="balanced"
)
lr_model.fit(X_train_vec, y_train)

print("🤖 Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

print("🤖 Training SVM...")
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

# ===== EVALUATE =====
print("\n📊 Logistic Regression:")
y_pred_lr = lr_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\n📊 Naive Bayes:")
y_pred_nb = nb_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))

print("\n📊 SVM:")
y_pred_svm = svm_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# ===== CHOOSE BEST MODEL =====
scores = {
    "lr": accuracy_score(y_test, y_pred_lr),
    "nb": accuracy_score(y_test, y_pred_nb),
    "svm": accuracy_score(y_test, y_pred_svm)
}

best_name = max(scores, key=scores.get)

if best_name == "lr":
    best_model = lr_model
elif best_name == "nb":
    best_model = nb_model
else:
    best_model = svm_model

print(f"\n🏆 Best model: {best_name} với accuracy = {scores[best_name]}")

# ===== SAVE =====
os.makedirs("model", exist_ok=True)

pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\n✅ Training xong, model đã lưu!")