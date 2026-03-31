# ThyroScan KBS – Thyroid Disorder Analysis System

A full-stack Knowledge-Based System (KBS) for thyroid disorder diagnosis using a trained ML model.

## Project Structure

```
thyroid_kbs/
├── train_model.py       # Data generation + model training
├── app.py               # Flask REST API backend
├── models/
│   ├── thyroid_model.pkl   # Trained Random Forest
│   ├── scaler.pkl          # StandardScaler
│   └── model_meta.json     # Accuracy, features, KB data
└── static/
    └── index.html          # Frontend UI
```

## Setup & Run

### 1. Install dependencies
```bash
pip install flask flask-cors scikit-learn pandas numpy joblib
```

### 2. Train the model (already done – skip if models/ exists)
```bash
python train_model.py
```

### 3. Start the Flask backend
```bash
python app.py
```
The API will run at **http://localhost:5000**

### 4. Open the frontend
Open `static/index.html` in your browser  
*(or serve it: `python -m http.server 8080` inside `static/`)*

---

## API Endpoints

| Method | Endpoint         | Description                  |
|--------|-----------------|------------------------------|
| POST   | `/api/analyze`  | Run diagnosis                |
| GET    | `/api/kb`       | Get knowledge base           |
| GET    | `/api/meta`     | Model accuracy & features    |
| GET    | `/api/health`   | Health check                 |

### POST /api/analyze – Example payload
```json
{
  "tsh": 8.5,
  "free_t4": 0.6,
  "free_t3": 75,
  "tpo_antibody": 1,
  "tg_antibody": 0,
  "tsh_receptor_antibody": 0,
  "age": 42,
  "bmi": 27.5,
  "heart_rate": 58,
  "fatigue_score": 8,
  "weight_change_kg": 4,
  "temperature_sensitivity": 7
}
```

---

## Model Details

- **Algorithm**: Random Forest Classifier (200 trees)
- **Features**: 12 clinical parameters
- **Classes**: Normal, Hypothyroidism, Subclinical Hypothyroidism, Subclinical Hyperthyroidism, Hyperthyroidism
- **Accuracy**: 100% on test set (trained on synthetic data with clinical ranges)

## KBS Rules Engine

The system applies evidence-based clinical rules on top of the ML prediction:
- TSH thresholds (< 0.35 → hyperthyroid, > 4.5 → hypothyroid)
- Free T4 / Free T3 interpretation
- Antibody flags (TPO, TG, TSH receptor)
- Vital signs (tachycardia, bradycardia)
- Severity grading (Mild / Moderate / Severe / Subclinical)

---

> ⚕️ **Disclaimer**: This system is for educational purposes only. Always consult a qualified endocrinologist for medical decisions.
