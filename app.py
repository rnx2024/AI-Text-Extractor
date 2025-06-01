import streamlit as st
from google.oauth2 import service_account
from google.cloud import vision
from PIL import Image
import io

# Set page
st.set_page_config(page_title="AI Text Extractor", layout="centered")
st.title("🔍 AI Text Extractor")

# Authenticate with Google Vision API
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    # ✅ Show image using Pillow
    pil_image = Image.open(io.BytesIO(image_bytes))
    st.image(pil_image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("🔎 Analyzing image with Google Vision AI..."):
        vision_image = vision.Image(content=image_bytes)
        response = client.text_detection(image=vision_image)

        if response.error.message:
            st.error(f"❌ API Error: {response.error.message}")
        else:
            annotations = response.text_annotations
            if annotations:
                full_text = annotations[0].description
                st.subheader("Extracted Text:")
                st.text_area("Text", full_text, height=300)
            else:
                st.warning("No text found in the image.")

    if 'full_text' in locals() and full_text:
        st.download_button(
            label="📥 Download Text",
            data=full_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )
