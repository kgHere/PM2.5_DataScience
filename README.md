# 🌫️ PM2.5 Air Quality Prediction & Scaling Techniques

## 📌 Overview
This project focuses on predicting **PM2.5 air pollution levels** using the **Beijing Multi-Site Air Quality dataset**. Multiple forecasting models were implemented and compared, with **Random Forest** selected as the best-performing approach. The project also explores **scaling and dimensionality reduction techniques** to improve model training efficiency.

---

## 📊 Dataset
- Beijing Multi-Site Air Quality Dataset  
- Hourly air pollution and meteorological data  
- Target variable: PM2.5 concentration  

---

## 🎯 Objectives
- Predict PM2.5 concentration accurately  
- Compare different machine learning and time-series models  
- Identify an effective prediction model  
- Improve training efficiency using scaling techniques  

---

## 🧠 Models Used
- Random Forest  
- SARIMA  
- Naive Baseline  
- LSTM  
- Ridge Regression  

---

## ⚙️ Scaling Techniques
To improve training efficiency, the following techniques were applied:
- Johnson–Lindenstrauss (JL) Projection  
- SVD-based dimensionality reduction  
- Random subsampling  

---

## 📈 Evaluation
Models were evaluated using standard regression and classification metrics.  
PM2.5 values were also mapped to air quality categories for classification-based analysis.

---

## 🛠️ Technologies Used
- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib  

