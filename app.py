import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
IMG_SIZE = 128

def load_trained_model(model_path='gender_age_model.h5'):
    """Load the trained Keras model"""
    try:
        model = load_model(model_path, compile=False)
        # Recompile with appropriate metrics if needed
        model.compile(
            optimizer='adam',
            loss={'age_output': 'mae', 'gender_output': 'binary_crossentropy'},
            metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
        )
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model file exists and is in the correct format.")
        return None

def preprocess_image(image_file, img_size=IMG_SIZE):
    """
    Preprocess image for model prediction
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            # Save uploaded file to temp
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and preprocess image
        img = load_img(tmp_path, color_mode='rgb', target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="UTKFace Age & Gender Predictor",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 UTKFace Age and Gender Prediction")
    st.markdown("""
    Upload a facial image to predict age and gender.
    *Model trained on UTKFace dataset*
    """)
    
    # Sidebar for model info
    with st.sidebar:
        st.header("⚙️ Model Information")
        st.markdown("""
        **Task**: Multi-task learning  
        **Output 1**: Age (regression)  
        **Output 2**: Gender (classification)  
        **Architecture**: Custom CNN  
        **Training Data**: UTKFace Dataset
        """)
        
        # Model loading option
        st.subheader("📁 Model Configuration")
        model_path = st.text_input("Path to model file:", "gender_age_model.h5")
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This app uses a deep learning model trained on the 
        [UTKFace dataset](https://susanqq.github.io/UTKFace/).
        
        **Gender labels**:  
        • 0 = Male  
        • 1 = Female
        """)
    
    # Main content - split into two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear facial image for best results",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                
                # Resize for display
                display_width = 400
                width_percent = (display_width / float(image.size[0]))
                display_height = int((float(image.size[1]) * float(width_percent)))
                image_display = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                st.image(image_display, caption="Uploaded Image", width=400)
                
                # Add image info
                st.caption(f"**Image details:** {image.size[0]} × {image.size[1]} pixels")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if uploaded_file is not None and st.button("🔍 Predict Age & Gender", type="primary"):
            with st.spinner("Processing image and making prediction..."):
                # Load model
                model = load_trained_model(model_path)
                
                if model:
                    # Preprocess image
                    processed_img, original_img = preprocess_image(uploaded_file)
                    
                    if processed_img is not None:
                        try:
                            # Make prediction
                            predictions = model.predict(processed_img, verbose=0)
                            age_pred = float(predictions[0][0][0])
                            gender_pred_prob = float(predictions[1][0][0])
                            
                            # Interpret predictions
                            # UTKFace: 0 = Male, 1 = Female
                            predicted_gender = "Female" if gender_pred_prob > 0.5 else "Male"
                            gender_confidence = gender_pred_prob if gender_pred_prob > 0.5 else 1 - gender_pred_prob
                            
                            # Display results in a nice format
                            st.success("✅ Prediction Complete!")
                            
                            # Create columns for metrics
                            col_a, col_g = st.columns(2)
                            
                            with col_a:
                                st.metric(
                                    label="**Predicted Age**",
                                    value=f"{age_pred:.1f}",
                                    help="Age in years"
                                )
                                
                            with col_g:
                                st.metric(
                                    label="**Predicted Gender**",
                                    value=predicted_gender,
                                    delta=f"{gender_confidence:.1%} confidence"
                                )
                            
                            # Add detailed information in expander
                            with st.expander("📋 View Detailed Results"):
                                st.write(f"**Age prediction:** {age_pred:.1f} years")
                                st.write(f"**Gender prediction:** {predicted_gender}")
                                st.write(f"**Gender confidence:** {gender_confidence:.2%}")
                                
                                # Confidence bar
                                st.write("**Gender confidence level:**")
                                st.progress(gender_confidence)
                                
                                # Age interpretation
                                st.write("**Age category:**")
                                if age_pred < 13:
                                    st.info("👶 Child (0-12 years)")
                                elif age_pred < 20:
                                    st.info("👦 Teenager (13-19 years)")
                                elif age_pred < 30:
                                    st.info("🧑 Young Adult (20-29 years)")
                                elif age_pred < 50:
                                    st.info("👨 Middle-aged (30-49 years)")
                                elif age_pred < 65:
                                    st.info("👨‍🦳 Mature Adult (50-64 years)")
                                else:
                                    st.info("👴 Senior (65+ years)")
                            
                            # Visualization
                            st.subheader("📈 Result Visualization")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            
                            # Original image
                            ax1.imshow(original_img)
                            ax1.set_title("Original Image")
                            ax1.axis('off')
                            
                            # Prediction info
                            ax2.axis('off')
                            info_text = f"Prediction Results:\n\n"
                            info_text += f"Age: {age_pred:.1f} years\n"
                            info_text += f"Gender: {predicted_gender}\n"
                            info_text += f"Confidence: {gender_confidence:.1%}"
                            ax2.text(0.1, 0.5, info_text, fontsize=12, 
                                    verticalalignment='center', 
                                    transform=ax2.transAxes)
                            ax2.set_title("Prediction Summary")
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            # Download results option
                            st.subheader("💾 Download Results")
                            result_text = f"""Age and Gender Prediction Results:
                            
Image: {uploaded_file.name}
Predicted Age: {age_pred:.1f} years
Predicted Gender: {predicted_gender}
Gender Confidence: {gender_confidence:.2%}
Age Category: {get_age_category(age_pred)}

Generated by UTKFace Age & Gender Predictor
"""
                            
                            st.download_button(
                                label="📥 Download Results as Text",
                                data=result_text,
                                file_name="prediction_results.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Prediction error: {str(e)}")
    
    # Add footer with instructions
    st.markdown("---")
    
    # Tips in expandable section
    with st.expander("💡 Tips for better predictions"):
        st.markdown("""
        **For optimal results:**
        1. 📸 Use clear, front-facing facial images
        2. 💡 Ensure good, even lighting
        3. 🎭 Avoid hats, sunglasses, or face coverings
        4. 😶 Use images with neutral expressions
        5. 🖼️ Crop image to focus on face
        
        **Limitations:**
        - Model trained on specific dataset (UTKFace)
        - May not perform optimally on images outside training distribution
        - Age prediction has inherent uncertainty
        """)
    
    # Add citation
    st.caption("""
    *Model trained on the [UTKFace dataset](https://susanqq.github.io/UTKFace/). 
    This is a demonstration project for educational purposes.*
    """)

def get_age_category(age):
    """Helper function to categorize age"""
    if age < 13:
        return "Child"
    elif age < 20:
        return "Teenager"
    elif age < 30:
        return "Young Adult"
    elif age < 50:
        return "Middle-aged"
    elif age < 65:
        return "Mature Adult"
    else:
        return "Senior"

if __name__ == "__main__":
    main()