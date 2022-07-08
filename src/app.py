import streamlit as st
from PIL import Image
from modules.model import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title('顔識別アプリケーション')
st.sidebar.write('チョコプラ松尾とパクセロイの顔を識別します。')

st.sidebar.write("")

img_source = st.sidebar.radio('画像のソースを選択してください。', ("画像をアップロード", "カメラで撮影"))

if img_source == '画像をアップロード':
    img_file = st.sidebar.file_uploader('画像を選択してください。', type=['png', 'jpg', 'jpeg'])
elif img_source == 'カメラで撮影':
    img_file = st.camera_input('カメラで撮影')

if img_file is not None:
    with st.spinner('判定中・・・'):
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(img_file)
            st.image(image, caption='Input', use_column_width=True)

        with col2:
            result_image = predict(image)
            st.image(result_image, caption='output', use_column_width=True)
