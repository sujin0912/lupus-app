import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 모델 불러오기
model = tf.keras.models.load_model("best_model.h5")

st.title("루푸스 얼굴 이미지 분류 AI")

uploaded_file = st.file_uploader("얼굴 이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='입력 이미지', use_column_width=True)

    # 전처리
    img = np.array(image.resize((128, 128))) / 255.0
    img = img.reshape(1, 128, 128, 3)

    prediction = model.predict(img)
    class_names = ['루푸스', '정상']
    result = class_names[np.argmax(prediction)]

    st.subheader(f" 예측 결과: **{result}**")
