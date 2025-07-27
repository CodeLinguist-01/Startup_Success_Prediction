# 🚀 Startup Growth Success Tracker & Dashboard

A Streamlit-based interactive web application to **predict startup success scores** using a trained ML model, and explore a rich **dashboard** to analyze startup data, funding, geography, and model insights.

---
## 🔗 Live Demo

Try the live app here:  
[Startup_Success_Prediction](https://startupsuccessprediction-kzxgtkznzxrujnzeagzcvw.streamlit.app/)

---
## Features

- **Success Score Predictor:**  
  Input your startup details and get a predicted success score (0-10) with actionable insights.

- **Interactive Dashboard:**  
  Explore startup data by country, industry, funding stage, and success categories.  
  Visualizations include pie charts, histograms, treemaps, geographic maps, and feature importance.

- **Model Insights:**  
  View feature importance from the trained model to understand what drives success predictions.

---

## Tech Stack

- Python 3.8+  
- Streamlit for frontend UI and app deployment  
- Scikit-learn for ML model  
- Plotly for interactive visualizations  
- Pandas, NumPy for data handling  
- Joblib & Pickle for model serialization

---

## Getting Started

### Prerequisites

- Python 3.8 or above  
- Recommended to use a virtual environment  

### Installation

1. Clone this repo:  
   ```bash
   git clone https://github.com/CodeLinguist-01/Startup_Success_Prediction.git
   cd Startup_Success_Prediction
   ```

## Files Needed

Make sure these files are present in your project folder:

- `model.pkl` — Trained ML model file  
- `scaler.pkl` — Scaler used for feature normalization  
- `input_columns.pkl` — Expected input features list  
- `startup_predictions-offline.csv` — Dataset with offline predictions  
- `Final-startup_success_predictions.csv` — Original startup data  
- `feature_importance.csv` — Feature importance values from the model  
- `app.py` — Main Streamlit app script  
- `requirements.txt` — Dependency list  

---

## Running the App

Run the Streamlit app using:

```bash
streamlit run app.py
```
## How to Use

### Predict Success Score
- Fill the startup details form on the home page.  
- Click **Predict Success Score** to see the predicted score with interpretation.

### Explore Dashboard
Use the sidebar to navigate between:

- **Overview:** Filter startups by country, industry, funding stage; view key metrics and charts.  
- **Profile & Geography:** Analyze geographic spread and funding by industry.  
- **Model Insights:** View feature importance and dataset.

---

## Project Structure

```bash
Startup_Growth_Prediction/
├── app.py                                 # Main Streamlit app  
├── model.pkl                              # Trained ML model  
├── scaler.pkl                             # Feature scaler  
├── input_columns.pkl                      # Expected input features list  
├── global_startup_success_dataset.csv     # Predictions dataset  
├── Final-startup_success.csv              # Original dataset  
├── feature_importance.csv                 # Feature importance info  
├── requirements.txt                       # Python dependencies  
```
---

## About the Model

The model predicts startup success scores based on multiple features including funding, employees, valuation, industry, and funding stage. It uses scaled inputs and logistic regression or XGBoost under the hood.

Feature importance data explains which features impact predictions most.

---

## Dependencies

Key Python libraries used:

- `streamlit >= 1.20.0`  
- `pandas >= 1.3.0`  
- `numpy >= 1.21.0`  
- `scikit-learn >= 1.0.0`  
- `plotly >= 5.0.0`  
- `streamlit-option-menu >= 0.3.5`  
- `joblib >= 1.1.0`  

---

## License

This project is licensed under the MIT License.

---

## Contact

Created by **CodeLinguist-01**  
[GitHub](https://github.com/CodeLinguist-01)

Feel free to open issues or contribute!


