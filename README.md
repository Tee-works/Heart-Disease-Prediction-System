# Heart Disease Prediction System

## Project Overview
Heart disease remains one of the leading causes of mortality worldwide, making early detection and prevention crucial for saving lives. This machine learning system predicts heart disease risk by analyzing patient medical data. Using parameters like blood pressure, cholesterol levels, and ECG results, our model achieves 96.5% accuracy in identifying potential heart disease cases. The project includes a Jupyter notebook showing the complete analysis and a Streamlit web application that makes predictions accessible through an easy-to-use interface.

## Features
- Data analysis and visualization of heart disease factors
- Machine learning model for risk prediction
- Interactive web interface for medical professionals
- Real-time prediction with detailed risk factor analysis
- Visual representation of prediction results

## Model Performance
Our model achieves exceptional performance metrics:
- Accuracy: 96.5%
- Class-wise Performance:
  - Healthy (Class 0):
    - Precision: 97%
    - Recall: 96%
    - F1-Score: 97%
  - Heart Disease (Class 1):
    - Precision: 96%
    - Recall: 97%
    - F1-Score: 96%

These metrics indicate that the model is highly reliable in identifying both healthy patients and those at risk of heart disease, with a well-balanced performance across both classes.

## Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Dataset Information
The dataset includes 303 patient records with the following features:
- Age: Patient's age in years
- Sex: Patient's gender (1 = male, 0 = female)
- Chest Pain Type: Types of chest pain
- Resting Blood Pressure: Blood pressure (mm Hg) at rest
- Serum Cholesterol: Cholesterol level in mg/dl
- Fasting Blood Sugar: Blood sugar > 120 mg/dl (1 = true, 0 = false)
- Resting ECG Results: Electrocardiogram results
- Maximum Heart Rate: Maximum heart rate achieved
- Exercise Induced Angina: Chest pain during exercise
- ST Depression: ST depression induced by exercise
- Slope: Slope of peak exercise ST segment
- Number of Major Vessels: Colored by fluoroscopy
- Thalassemia: Blood disorder type


## Usage
The project can be used in two ways:

1. Jupyter Notebook:
- Open `Heart_Disease_Prediction.ipynb` to see the complete analysis
- Includes data preprocessing, model development, and evaluation

2. Streamlit Application:
- Provides an interactive interface for medical professionals
- Enter patient data to receive instant risk predictions
- Displays detailed risk factor analysis and recommendations

## Technical Details
- Model Type: Random Forest Classifier
- Validation Method: Train-test split (80-20)
- Performance Evaluation: Precision, Recall, F1-Score for both classes
- Data Processing: Standard scaling and feature engineering

## Contributing
Contributions to improve the project are welcome. Please feel free to:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset from the UCI Machine Learning Repository
- Based on research in heart disease prediction using machine learning
