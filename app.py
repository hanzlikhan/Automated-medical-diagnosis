import streamlit as st
from ibm_watson import VisualRecognitionV4
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from PIL import Image
import numpy as np
import cv2
import io
import base64

# Initialize IBM Watson Visual Recognition API
api_key = "YOUR_API_KEY"
service_url = "YOUR_SERVICE_URL"

authenticator = IAMAuthenticator(api_key)
visual_recognition = VisualRecognitionV4(
    version='2023-08-25',  # Make sure this version is correct for your use case
    authenticator=authenticator
)
visual_recognition.set_service_url(service_url)

# Function to preprocess image for model input
def preprocess_image(image: Image.Image) -> np.ndarray:
    image_np = np.array(image)
    resized_image = cv2.resize(image_np, (224, 224))  # Resize to model's input size
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return normalized_image

# Function to generate a downloadable link for the report
def generate_download_link(content, filename):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Report</a>'

# Streamlit app setup
st.title("Automated Medical Diagnosis Support System")
st.write("Upload a medical image to receive a diagnostic suggestion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for model prediction
    preprocessed_image = preprocess_image(image)
    
    # Diagnosis button
    if st.button("Analyze Image"):
        with st.spinner('Analyzing...'):
            # Convert image to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            try:
                # Example API call (adjust according to IBM Watson Visual Recognition API)
                response = visual_recognition.classify(
                    images_file=image_bytes,
                    classifier_ids=['default']  # Use your classifier ID
                ).get_result()

                # Extract diagnosis result
                diagnosis = response['images'][0]['classifiers'][0]['classes'][0]
                result = diagnosis['class']
                confidence = diagnosis['score']
                
                # Display diagnosis result
                st.write("Diagnosis:", result)
                st.write("Confidence Score:", f"{confidence:.2f}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        # Generate a report
        report_content = f"Diagnosis: {result}\nConfidence Score: {confidence:.2f}"
        download_link = generate_download_link(report_content, "diagnosis_report.txt")
        st.markdown(download_link, unsafe_allow_html=True)
else:
    st.info("Please upload an image to start the diagnosis.")

# Footer with instructions or additional information
st.markdown(
    """
    **Note:** This tool provides preliminary analysis and should not replace a professional medical diagnosis. 
    Ensure patient confidentiality and data security at all times.
    """
)
