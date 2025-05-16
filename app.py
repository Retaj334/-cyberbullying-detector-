
# مشروع كشف التنمر الإلكتروني - Streamlit
# إعداد: ريتاج

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# تحميل النموذج و vectorizer
model = joblib.load("model_tanammur.pkl")
vectorizer = joblib.load("vectorizer_tanammur.pkl")

st.title("كاشف التنمر الإلكتروني")
st.write("أدخلي العبارة، وراح نكشف إذا فيها تنمّر أو لا.")

user_input = st.text_input("أكتبي العبارة هنا")

if user_input:
    user_input = user_input.lower()
    vec = vectorizer.transform([user_input])
    pred = model.predict(vec)[0]
    result = "تنمّر" if pred == 1 else "كلام عادي"
    st.write(f"النتيجة: **{result}**")
