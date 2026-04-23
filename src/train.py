print("🚀 START TRAINING...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import pickle

from src.preprocess import clean_text

# ===== LOAD DATA =====
df = pd.read_csv("data/dulieu_huanluyen.csv", encoding="latin-1")

# Đổi tên cột cho dễ dùng
df = df.rename(columns={
    "comments": "text",
    "flag": "label"
})

# ===== CLEAN DATA =====
df["text"] = df["text"].apply(clean_text)

# Xóa dòng rỗng
df = df[df["text"].str.strip() != ""]

# Check nhanh
print("Sample data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ===== VECTORIZE =====
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),   # unigram + bigram
    min_df=3,
    max_df=0.9
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===== TRAIN =====
model = LogisticRegression(
    C=2.0,
    max_iter=200,
    class_weight="balanced"
)
model.fit(X_train_vec, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)


# ===== EVALUATE =====
y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nNaive Bayes:")
print(classification_report(y_test, nb_model.predict(X_test_vec)))

# ===== SAVE =====
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\n✅ Training xong, model đã lưu!")