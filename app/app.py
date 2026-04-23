import streamlit as st
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict

st.title("Vietnamese Sentiment Analysis 😎")

text = st.text_input("Nhập câu:")

if st.button("Dự đoán"):
    if text.strip() == "":
        st.warning("Nhập gì đó đi bro 😭")
    else:
        result = predict(text)
        if result == 1:
            st.success("😊 Tích cực")
        else:
            st.error("😡 Tiêu cực")