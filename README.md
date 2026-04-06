<!-- 🌌 Header -->
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=220&section=header&text=Student%20Performance%20Prediction&fontSize=40&fontColor=ffffff&animation=fadeIn"/>
</p>

---

# 🎓 AI Student Performance Prediction System

An advanced **Machine Learning + Data Analytics project** that predicts student academic performance using behavioral, academic, and lifestyle features.

---

# 🚀 Key Highlights

- 🧠 Random Forest ML Model  
- 📊 Feature Engineering Pipeline  
- 📈 Model Evaluation (MAE, RMSE, R²)  
- 🖥️ CLI-based training & prediction system  
- 🌐 Interactive Streamlit Dashboard  
- 🤖 AI-generated insights & reports  
- 📄 PDF report generation  

---

# 📊 Model Performance

| Metric | Value |
|--------|------|
| MAE | Stored in `metrics.pkl` |
| RMSE | Stored in `metrics.pkl` |
| R² Score | Stored in `metrics.pkl` |

> Model uses **RandomForestRegressor (500 trees, depth=10)** for stable predictions :contentReference[oaicite:0]{index=0}

---

# 🧠 How It Works

1. Load student dataset  
2. Handle missing values (mean/mode) :contentReference[oaicite:1]{index=1}  
3. Apply feature engineering  
4. Create final score formula  
5. Train ML model  
6. Evaluate performance  
7. Save model & metrics  
8. Predict new student score  

---

# ⚙️ Run Project (IMPORTANT)

# 📥 Clone & Download

### 🔹 Clone Repository
```bash
git clone https://github.com/VASANI007/AI-Student-Performance-Prediction-Analytics-System-ML-Dashboard-
```

### 🔹 Train Model
```bash
python main.py --mode train
```

### 🔹 Predict Score
```bash
python main.py --mode predict
```

✔ CLI handled using argparse  
✔ Modes: `train` or `predict` :contentReference[oaicite:2]{index=2}  

---

# 🖥️ Run Dashboard

```bash
streamlit run app/app.py
```

---

# 📂 Project Structure

```
STUDENT-PERFORMANCE-PREDICTION/

├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── student_data.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── models/
│   ├── model.pkl
│   ├── columns.pkl
│   └── metrics.pkl
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_building.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluate_model.py
│
├── main.py
└── requirements.txt
```

---

# 📄 File Explanation (Core System 🚀)

### 🔹 src/train_model.py
- Full training pipeline  
- Creates **final_score formula**
- Trains RandomForest model  
- Saves:
  - model.pkl  
  - columns.pkl  
  - metrics.pkl :contentReference[oaicite:3]{index=3}  

---

### 🔹 src/predict.py
- Loads trained model  
- Applies preprocessing + feature engineering  
- Predicts student score :contentReference[oaicite:4]{index=4}  

---

### 🔹 src/feature_engineering.py
- Creates advanced features:
  - total_academic  
  - performance_index  
- Improves prediction accuracy :contentReference[oaicite:5]{index=5}  

---

### 🔹 src/data_preprocessing.py
- Handles missing values  
- Encodes categorical data  
- Cleans dataset :contentReference[oaicite:6]{index=6}  

---

### 🔹 src/evaluate_model.py
- Calculates:
  - MAE  
  - RMSE  
  - R² Score :contentReference[oaicite:7]{index=7}  

---

### 🔹 main.py
- CLI controller  
- Runs:
  - Training mode  
  - Prediction mode :contentReference[oaicite:8]{index=8}  

---

### 🔹 app/app.py
- Streamlit dashboard  
- Features:
  - 🎯 Prediction UI  
  - 📊 Data visualization  
  - 🤖 AI chatbot  
  - 📈 Clustering (KMeans, DBSCAN)  
  - 📄 PDF report generation :contentReference[oaicite:9]{index=9}  

---

# 📊 Dashboard Features

- 🎯 Student score prediction  
- 📊 Dataset visualization  
- 📈 Regression analysis  
- 🔥 Heatmaps  
- 🤖 AI insights generation  
- 📄 Downloadable PDF report  
- 📊 Clustering:
  - KMeans  
  - DBSCAN  
  - Hierarchical  

---

# 🧠 ML Logic

Final score is calculated using weighted formula:

- Previous Score → 50%  
- Internal Marks → 20%  
- Assignments → 20%  
- Attendance → 10%  

✔ Balanced academic evaluation system :contentReference[oaicite:10]{index=10}  

---

# 🔮 Prediction System

- Real-time input-based prediction  
- Feature-aligned inference  
- Handles categorical + numeric inputs  
- Returns rounded final score  

---

# 🚀 Future Improvements

- 🔥 Deep Learning models  
- 📱 Mobile UI  
- ☁️ Cloud deployment  
- 📡 Real-time student tracking  

---

# 👨‍💻 Author

Daksh Vasani  

Machine Learning Enthusiast  
Data Analytics Developer  

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/>
</p>
