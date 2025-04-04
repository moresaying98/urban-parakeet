import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('ET.pkl')  # 加载训练好的ET模型

# GGT input
GGT = st.sidebar.number_input("GGT:", min_value=1, max_value=70, value=50)  # γ-谷氨酰转肽酶

# ALP input
ALP = st.sidebar.number_input("ALP:", min_value=30, max_value=650, value=50)  # 碱性磷酸酶

# Height input
Height = st.sidebar.number_input("Height:", min_value=130, max_value=210, value=150)  # 身高

# TG input
TG = st.sidebar.number_input("TG:", min_value=10, max_value=400, value=50)  # 甘油三脂

# WC input
WC = st.sidebar.number_input("WC:", min_value=50, max_value=170, value=80)  # 腰围

# Insulin input
Insulin = st.sidebar.number_input("Insulin:", min_value=0.5, max_value=250.0, value=10.0,)# 胰岛素

# GLU input
GLU = st.sidebar.number_input("GLU:", min_value=50, max_value=550, value=150)  # 血糖

# RBC input
RBC = st.sidebar.number_input("RBC:", min_value=3.5, max_value=7.5, value=5.5)  # 红细胞

# HDL input
HDL = st.sidebar.number_input("HDL:", min_value=20, max_value=120, value=50)  # 高密度脂蛋白

# WBC input
WBC = st.sidebar.number_input("WBC:", min_value=0, max_value=20, value=5)  # 白细胞

# Collecting all the feature values
if st.button("Make Prediction"):
    # Features array
    features = np.array([[GGT, ALP, Height, TG, WC, Insulin, GLU, RBC, HDL, WBC]])  # Reshape array for model input
    
    # Model prediction
    predicted_class = model.predict(features)[0]  # Predict the class
    predicted_proba = model.predict_proba(features)[0]  # Get probabilities

    # Display the prediction results
    st.write(f"**Predicted Class:** {'Sick' if predicted_class == 1 else 'Not Sick'}")  # Show predicted class
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # Show probabilities for each class

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # Based on predicted class, get the probability and convert to percentage

    if predicted_class == 1:  # If the prediction is 'Sick'
        advice = (
            f"According to our model, your risk of NAFLD disease is high. "
            f"The probability of you having NAFLD disease is {probability:.1f}%. "
            "Although this is just a probability estimate, it suggests that you might have a higher risk of NAFLD disease. "
            "I recommend that you contact a liver specialist for further examination and assessment, "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:  # If the prediction is 'Not Sick'
        advice = (
            f"According to our model, your risk of NAFLD disease is low. "
            f"The probability of you not having NAFLD disease is {probability:.1f}%. "
            "Nevertheless, maintaining a healthy lifestyle is still very important. "
            "I suggest that you have regular health check-ups to monitor your liver health, "
            "and seek medical attention if you experience any discomfort."
        )

    st.write(advice)  # Show advice

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # Probability for Class 0
        'Class_1': predicted_proba[1]   # Probability for Class 1
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # Set figure size

    # Create bar chart
    bars = plt.barh(['Not Sick', 'Sick'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # Horizontal bar chart

    # Add title and labels
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # Title
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # X-axis label
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # Y-axis label

    # Add probability text labels
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  
        plt.text(v + 0.005, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # Add text labels

    # Hide other axes (top, right)
    plt.gca().spines['top'].set_visible(False)  # Hide top spine
    plt.gca().spines['right'].set_visible(False)  # Hide right spine

    # Show the plot
    st.pyplot(plt)  # Display the chart
