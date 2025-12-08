# 🎭 Age and Gender Prediction

A deep learning-based application that predicts age and gender from facial images using the UTKFace dataset. The project features a multi-task CNN model and a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## ✨ Features

- **Multi-task Learning**: Simultaneous age regression and gender classification
- **Streamlit Web Interface**: User-friendly web application
- **Real-time Predictions**: Instant age and gender predictions
- **Confidence Scores**: Gender prediction with confidence percentage
- **Age Categorization**: Automatic age group classification
- **Responsive Design**: Works on various screen sizes
- **Download Results**: Export predictions as text file

## 🎥 Test

![alt text](image.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone [https://github.com/yourusername/utkface-age-gender-prediction.git](https://github.com/yourusername/utkface-age-gender-prediction.git)
cd utkface-age-gender-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
Download the pre-trained model from Releases and place it in the project root directory.

## 🚀 Usage

### Running the Streamlit Application
```bash
streamlit run app.py
```
The application will open in your default web browser at `http://localhost:8501`.

### Using the Application
1. **Upload Image**: Click "Browse files" to upload a facial image (JPG, PNG, JPEG)
2. **Click Predict**: Press the "Predict Age & Gender" button
3. **View Results**: See age prediction, gender classification with confidence
4. **Download Results**: Option to save predictions as text file



## 📁 Project Structure

```text
Age-and-Gender-Prediction/
│
├── app.py                    # Main Streamlit application
├── model_training.ipynb      # Jupyter notebook for model training
├── gender_age_model.h5       # Pre-trained model (download separately)
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🧠 Model Architecture

The model uses a multi-task learning approach with shared convolutional layers:

```text
Input Layer (128×128×3 RGB image)
                │
        ┌───────▼───────┐
        │ Conv2D (16)   │
        │ MaxPool2D     │
        ├───────────────┤
        │ Conv2D (32)   │
        │ MaxPool2D     │
        ├───────────────┤
        │ Conv2D (64)   │
        │ MaxPool2D     │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ Flatten       │
        │ Dense(64)     │
        │ Dropout(0.5)  │
        │ Dense(32)     │
        └───────┬───────┘
                │
┌───────────────▼───────────────┐
│ Age Head      │ Gender Head   │
│ Dense(1)      │ Dense(1)      │
│ (Linear)      │ (Sigmoid)     │
└───────────────┴───────────────┘
```

**Key Features:**
* **Shared Features:** Common CNN backbone for both tasks
* **Task-Specific Heads:** Separate branches for age and gender
* **Dropout Regularization:** Prevents overfitting
* **Batch Normalization:** Improves training stability

## 📊 Dataset

The model is trained on the UTKFace Dataset:

* **Source:** UTKFace Dataset
* **Images:** ~20,000 facial images
* **Annotations:** Age, Gender, Ethnicity
* **Age Range:** 0-116 years
* **Gender Labels:** 0 = Male, 1 = Female
* **Image Size:** 200×200 pixels (resized to 128×128)

**Dataset Distribution**
* **Age Distribution:** Uniform with peaks at young and middle ages
* **Gender Distribution:** Approximately balanced

## 📈 Results

### Model Performance

| Metric | Value | Description |
| :--- | :--- | :--- |
| Age MAE | ~9.3 years | Mean Absolute Error |
| Gender Accuracy | ~94% | Classification accuracy |
| Inference Time | < 0.5s | Per image prediction |
| Model Size | ~15 MB | .h5 file size |

## 🌐 Deployment

### Local Deployment
```bash
# Run locally
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8080
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t age-gender-predictor .
docker run -p 8501:8501 age-gender-predictor
```

### Cloud Deployment Options
* **Streamlit Cloud:** One-click deployment
* **Heroku:** Container deployment
* **AWS EC2:** Virtual machine hosting
* **Google Cloud Run:** Serverless container


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

* **UTKFace Dataset:** For providing the training data
* **TensorFlow/Keras:** Deep learning framework
* **Streamlit:** For the amazing web app framework
* **OpenCV:** Image processing library

## 📞 Contact

Your Name - [@hsnbshr98](https://github.com/hsnbshr98) - hussinbash98@gmail.com

Project Link: [https://github.com/hsnbshr98/Age-and-Gender-Prediction](https://github.com/hsnbshr98/Age-and-Gender-Prediction)

## ⭐ Support

If you find this project helpful, please give it a star! ⭐

---
Built with ❤️ using Python, TensorFlow, and Streamlit