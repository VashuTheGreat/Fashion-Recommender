import streamlit as st
from temp import give_path_get_link

st.set_page_config(page_title="Image Recommendation", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #FFD700;
        margin-top: -30px;
        text-shadow: 2px 2px 6px #000;
    }
    .desc {
        text-align: center;
        font-size: 1.1em;
        color: #ccc;
        margin-bottom: 30px;
    }
    .scroll-container {
        display: flex;
        overflow-x: auto;
        padding: 20px;
        gap: 15px;
        scrollbar-width: thin;
    }
    .scroll-container img {
        border-radius: 12px;
        width: 130px;
        height: 130px;
        object-fit: cover;
        transition: transform 0.2s ease;
    }
    .scroll-container img:hover {
        transform: scale(1.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🖼️ Image Recommendation</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Upload an image to discover visually similar ones below</div>", unsafe_allow_html=True)

image_choice = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if image_choice:
    st.image(image_choice, use_column_width=True, caption="Uploaded Image")
    res,_ = give_path_get_link(image_choice)
    if res:
        st.markdown("<h3 style='color:#FFD700;margin-top:30px;'>Recommended Images</h3>", unsafe_allow_html=True)
        html = "<div class='scroll-container'>" + "".join([f"<img src='{i}'/>" for i in res]) + "</div>"
        st.markdown(html, unsafe_allow_html=True)
