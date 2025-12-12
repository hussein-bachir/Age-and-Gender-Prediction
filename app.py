import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tempfile
import os
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
IMG_SIZE = 128

# Initialize session state
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

def load_trained_model(model_path='gender_age_model.h5'):
    """Load the trained Keras model"""
    try:
        model = load_model(model_path, compile=False)
        # Recompile with appropriate metrics
        model.compile(
            optimizer='adam',
            loss={'age_output': 'mae', 'gender_output': 'binary_crossentropy'},
            metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def detect_faces_simple(frame):
    """Detect faces in frame using Haar Cascade"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(30, 30)
        )
        
        return faces
            
    except Exception as e:
        st.warning(f"Face detection issue: {str(e)}")
        return []

def predict_face_simple(model, face_img):
    """Predict age and gender for a face"""
    try:
        # Resize and normalize
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        # Predict
        predictions = model.predict(face_batch, verbose=0)
        age_pred = float(predictions[0][0][0])
        gender_pred_prob = float(predictions[1][0][0])
        
        # Interpret results
        gender = "Female" if gender_pred_prob > 0.5 else "Male"
        confidence = gender_pred_prob if gender_pred_prob > 0.5 else 1 - gender_pred_prob
        
        return age_pred, gender, confidence
        
    except Exception as e:
        st.warning(f"Prediction error: {str(e)}")
        return 0, "Unknown", 0.0

def main():
    st.set_page_config(
        page_title="UTKFace Age & Gender Predictor",
        page_icon="🎭",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎭 Real-time Age & Gender Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image or use your webcam for real-time predictions</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading
        model_path = st.text_input("Model file path:", "gender_age_model.h5")
        
        st.markdown("---")
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Image Upload**: Upload a clear face photo
        2. **Webcam Mode**: Click 'Start Webcam' for real-time
        3. **For best results**:
           - Face the camera directly
           - Ensure good lighting
           - Remove sunglasses/hats
        
        **Gender labels**: 0=Male, 1=Female
        """)
    
    # Main content - tabs for different modes
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Real-time Webcam"])
    
    with tab1:
        # Original image upload functionality
        st.subheader("Upload Image for Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=["jpg", "jpeg", "png"],
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
        
        with col2:
            if uploaded_file is not None:
                if st.button("🔍 Predict from Upload", type="primary"):
                    with st.spinner("Processing..."):
                        model = load_trained_model(model_path)
                        
                        if model:
                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                # Preprocess
                                img = load_img(tmp_path, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE))
                                img_array = img_to_array(img) / 255.0
                                img_array = np.expand_dims(img_array, axis=0)
                                
                                # Predict
                                predictions = model.predict(img_array, verbose=0)
                                age_pred = float(predictions[0][0][0])
                                gender_pred_prob = float(predictions[1][0][0])
                                
                                predicted_gender = "Female" if gender_pred_prob > 0.5 else "Male"
                                gender_confidence = gender_pred_prob if gender_pred_prob > 0.5 else 1 - gender_pred_prob
                                
                                # Display results
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                col_a, col_g = st.columns(2)
                                with col_a:
                                    st.metric("Predicted Age", f"{age_pred:.1f} years")
                                with col_g:
                                    st.metric("Predicted Gender", predicted_gender, 
                                             delta=f"{gender_confidence:.1%} conf")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show confidence
                                st.progress(gender_confidence)
                                
                                # Clean up
                                os.unlink(tmp_path)
                                
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
    
    with tab2:
        st.subheader("Real-time Webcam Prediction")
        
        # Load model
        model = load_trained_model(model_path)
        
        if model is None:
            st.error("❌ Could not load model. Please check model file.")
            st.stop()
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Webcam", type="primary", key="start_btn"):
                st.session_state.webcam_active = True
        with col2:
            if st.button("⏹️ Stop Webcam", type="secondary", key="stop_btn"):
                st.session_state.webcam_active = False
        
        # Webcam feed
        if st.session_state.webcam_active:
            st.success("✅ Webcam active - Detecting faces...")
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check camera permissions.")
                st.session_state.webcam_active = False
            else:
                # Create placeholders
                frame_placeholder = st.empty()
                current_prediction = st.empty()
                
                try:
                    while st.session_state.webcam_active and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Cannot read from webcam")
                            break
                        
                        # Convert for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces
                        faces = detect_faces_simple(frame)
                        
                        # Process each face
                        for (x, y, w, h) in faces:
                            # Draw rectangle
                            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Predict for this face
                            face_roi = frame[y:y+h, x:x+w]
                            if face_roi.size > 0:
                                age, gender, confidence = predict_face_simple(model, face_roi)
                                
                                # Add label
                                label = f"{gender} {age:.1f}y"
                                if confidence > 0.5:
                                    label += f" ({confidence:.0%})"
                                
                                # Draw text with background
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
                                
                                # Draw background rectangle for text
                                cv2.rectangle(frame_rgb, 
                                            (x, y - text_size[1] - 10), 
                                            (x + text_size[0], y), 
                                            (0, 255, 0), 
                                            -1)
                                
                                # Draw text
                                cv2.putText(frame_rgb, label, (x, y - 5), 
                                          font, 0.7, (0, 0, 0), 2)
                                
                                # Show current prediction
                                with current_prediction.container():
                                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                    col_a, col_g = st.columns(2)
                                    with col_a:
                                        st.metric("Current Age", f"{age:.1f} years")
                                    with col_g:
                                        st.metric("Current Gender", gender, 
                                                 delta=f"{confidence:.1%} conf")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display frame
                        frame_placeholder.image(frame_rgb, channels="RGB", width=640)
                        
                        # If no faces detected
                        if len(faces) == 0:
                            with current_prediction.container():
                                st.info("👤 No faces detected. Please position your face in front of the camera.")
                        
                        # Control frame rate
                        time.sleep(0.03)
                        
                except Exception as e:
                    st.error(f"❌ Webcam error: {str(e)}")
                
                # Clean up
                cap.release()
                cv2.destroyAllWindows()
        
        else:
            # Instructions when webcam is off
            st.info("""
            ### 👆 Ready to Start?
            
            **Click 'Start Webcam' to begin real-time face detection**
            
            **How it works:**
            1. Click ▶️ **Start Webcam** button
            2. Allow camera access in your browser
            3. Position your face in the camera view
            4. The system will automatically detect your face
            5. Age and gender predictions will appear on screen
            6. Click ⏹️ **Stop Webcam** when finished
            
            **Tips for best results:**
            - 🎯 Face the camera directly
            - 💡 Ensure good lighting
            - 👓 Remove sunglasses/hats
            - 📏 Stay close enough for clear view
            """)

if __name__ == "__main__":
    main()