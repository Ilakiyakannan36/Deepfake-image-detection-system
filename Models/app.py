import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="DeepGuard - Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ddd;
    }
    .real-result {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .fake-result {
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    .confidence-bar {
        height: 20px;
        background: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Feature extraction function
def extract_advanced_features(image_path):
    """Extract features for deepfake detection"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (128, 128))
        features = []
        
        # Color features
        for channel in range(3):
            features.append(np.mean(img[:, :, channel]))
            features.append(np.std(img[:, :, channel]))
        
        # Texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.max(gray))
        features.append(np.min(gray))
        
        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.sum(edges > 0) / edges.size)
        
        # Noise features
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.var(laplacian))
        features.append(np.median(gray))
        
        return np.array(features)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load model function
@st.cache_resource
def load_model(model_path='models/deepguard_advanced_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ DeepGuard - Deepfake Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        DeepGuard uses machine learning to detect AI-generated 
        and manipulated images. Upload an image to check if it's 
        authentic or potentially manipulated.
        """)
        
        st.header("How it Works")
        st.write("""
        1. Upload an image (JPG, PNG, JPEG)
        2. The system extracts forensic features
        3. Machine learning model analyzes the features
        4. Get instant results with confidence scores
        """)
        
        st.header("Model Info")
        st.write("""
        - Trained on 2,041 images
        - Random Forest classifier
        - 85%+ accuracy
        - Analyzes color and texture patterns
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image to analyze"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Load model
            model = load_model()
            
            if model:
                # Extract features
                features = extract_advanced_features(temp_path)
                
                if features is not None:
                    # Predict
                    prediction = model.predict(features.reshape(1, -1))[0]
                    probabilities = model.predict_proba(features.reshape(1, -1))[0]
                    
                    # Display results
                    with col2:
                        st.header("📊 Detection Results")
                        
                        # Result box
                        result_class = "real-result" if prediction == 0 else "fake-result"
                        result_text = "REAL ✅" if prediction == 0 else "FAKE ⚠️"
                        
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>Prediction: {result_text}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence scores
                        st.subheader("Confidence Scores")
                        
                        # Real probability
                        real_prob = probabilities[0]
                        st.write(f"**Real Probability: {real_prob:.3f}**")
                        st.progress(real_prob)
                        
                        # Fake probability  
                        fake_prob = probabilities[1]
                        st.write(f"**Fake Probability: {fake_prob:.3f}**")
                        st.progress(fake_prob)
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        if prediction == 0 and real_prob > 0.7:
                            st.success("✅ This image appears to be authentic with high confidence!")
                        elif prediction == 1 and fake_prob > 0.7:
                            st.error("⚠️ This image is likely manipulated or AI-generated!")
                        elif max(probabilities) > 0.6:
                            status = "REAL" if prediction == 0 else "FAKE"
                            st.warning(f"🤔 Likely {status} (moderate confidence)")
                        else:
                            st.info("❓ Uncertain prediction - low confidence")
                
                # Clean up temp file
                os.unlink(temp_path)
    
    # Demo section if no file uploaded
    if uploaded_file is None:
        with col2:
            st.header("🎯 Try It Out")
            st.info("""
            Upload an image to the left to analyze it. The system will:
            
            1. Extract forensic features from the image
            2. Analyze patterns using machine learning
            3. Provide a confidence score for authenticity
            
            **Example results you might see:**
            - ✅ REAL (85% confidence) - Likely authentic
            - ⚠️ FAKE (90% confidence) - Likely manipulated
            - 🤔 Uncertain (55% confidence) - Hard to tell
            """)

# Run the app
if __name__ == "__main__":
    main()