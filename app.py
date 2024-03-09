import streamlit as st
from PIL import Image
import pytesseract
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = None

def initialize_summarizer():
    global summarizer
    summarizer = pipeline("summarization",model="t5-small")

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("Image Text Summarizer")

    # Check if summarizer is initialized
    if summarizer is None:
        initialize_summarizer()

    # Upload image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from the uploaded image
        text = extract_text_from_image(image)

        # Summarize the extracted text
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        # Display the summary
        st.subheader("Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
