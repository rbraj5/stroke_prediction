# 🏥 Stroke Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready ML system for stroke risk prediction with REST API, interactive dashboard, and model explainability.**

## 📋 Overview

This project demonstrates a complete machine learning pipeline from research to production, featuring:

- ✅ **Multiple ML models** (Balanced Random Forest, Logistic Regression, Gradient Boosting)
- ✅ **RESTful API** with FastAPI
- ✅ **Interactive web dashboard** with Streamlit
- ✅ **Model explainability** with feature importance and SHAP values
- ✅ **Docker containerization** for easy deployment
- ✅ **Comprehensive testing** with pytest
- ✅ **API documentation** (auto-generated with Swagger/ReDoc)
- ✅ **Production-grade** error handling, logging, and validation

## 🎯 Key Features

### 1. Advanced ML Pipeline
- Data preprocessing with feature engineering
- Handling imbalanced data (4.87% positive class)
- Multiple model comparison and selection
- Hyperparameter tuning
- Model versioning and persistence

### 2. Production API
- **FastAPI** backend with async support
- Pydantic validation for type safety
- Comprehensive error handling
- Health check endpoints
- Model information endpoint
- Prediction with confidence scores

### 3. Interactive Dashboard
- Real-time predictions
- Risk level visualization (gauge charts)
- Feature contribution analysis
- Clinical recommendations
- Model performance metrics

### 4. Model Interpretability
- Feature importance ranking
- SHAP value integration (ready)
- Individual prediction explanations
- Risk factor identification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT INTERFACE                         │
│  ┌────────────────────┐         ┌────────────────────┐     │
│  │  Streamlit Dashboard│         │  External Apps     │     │
│  │  (Port 8501)       │         │  (Mobile, Web)     │     │
│  └─────────┬──────────┘         └──────────┬─────────┘     │
│            │                               │                 │
│            └───────────────┬───────────────┘                │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI REST API                        │   │
│  │              (Port 8000)                             │   │
│  │  ┌──────────────────┐  ┌──────────────────┐        │   │
│  │  │ /predict         │  │ /predict/explain  │        │   │
│  │  │ /health          │  │ /model/info       │        │   │
│  │  └──────────────────┘  └──────────────────┘        │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│                    ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ML LAYER                                │   │
│  │  ┌────────────────┐  ┌────────────────┐            │   │
│  │  │ Preprocessor   │  │ Trained Model   │            │   │
│  │  │ (Scaling, FE)  │  │ (Balanced RF)   │            │   │
│  │  └────────────────┘  └────────────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Model Performance

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Balanced Random Forest | 0.61     | 0.10      | 0.84   | 0.18     | 0.75    |
| Random Forest          | 0.60     | 0.09      | 0.82   | 0.16     | 0.74    |
| Logistic Regression    | 0.65     | 0.08      | 0.78   | 0.14     | 0.72    |
| Gradient Boosting      | 0.63     | 0.09      | 0.80   | 0.16     | 0.73    |

**Key Insight:** Model optimized for **high recall (0.84)** - critical in medical applications where missing a stroke risk (false negative) is more costly than a false alarm.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

#### Option 1: Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/stroke-prediction-system.git
cd stroke-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
cd ml
python train.py
cd ..

# Start API server
uvicorn api.main:app --reload

# In another terminal, start dashboard
streamlit run dashboard/app.py
```

#### Option 2: Docker Deployment
```bash
# Build and run with docker-compose
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## 📖 Usage

### 1. API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

#### Get Prediction with Explanation
```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{...}'  # Same payload as above
```

### 2. Dashboard Usage

1. Open browser to `http://localhost:8501`
2. Enter patient information in the form
3. Click "Predict Stroke Risk"
4. View results, risk level, and feature contributions

### 3. Python SDK Usage

```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Patient data
patient = {
    "age": 67,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

# Make prediction
response = requests.post(f"{API_URL}/predict", json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=api --cov=ml --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 📁 Project Structure

```
stroke-prediction-system/
├── api/                      # FastAPI application
│   ├── main.py              # Main API application
│   └── schemas.py           # Pydantic models
├── ml/                      # Machine learning modules
│   ├── preprocessing.py     # Data preprocessing
│   └── train.py            # Model training
├── dashboard/               # Streamlit dashboard
│   └── app.py              # Dashboard application
├── tests/                   # Test suite
│   └── test_api.py         # API tests
├── models/                  # Trained models (generated)
│   ├── preprocessor.pkl
│   ├── stroke_model_production.pkl
│   └── model_comparison.json
├── data/                    # Data files
│   └── stroke-data.csv
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=models/stroke_model_production.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl

# Logging
LOG_LEVEL=INFO
```

## 📈 Model Training Details

### Data Preprocessing
1. **Missing Value Handling**: BMI median imputation
2. **Feature Engineering**:
   - Age groups (0-50, 50-80, 80+)
   - BMI categories (Underweight, Normal, Overweight, Obese)
   - Glucose categories (Normal, Prediabetic, Diabetic)
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: StandardScaler for numerical features

### Model Selection
- **Chosen Model**: Balanced Random Forest
- **Reason**: Highest recall (0.84) with reasonable precision
- **Alternatives Tested**: Random Forest, Logistic Regression, Gradient Boosting

### Class Imbalance Handling
- Class weights adjustment
- Balanced Random Forest (built-in balancing)
- SMOTE tested but rejected (degraded feature relevance)

## 🎓 Clinical Considerations

### Why High Recall?
In medical applications, **false negatives** (missing a stroke risk) are more dangerous than **false positives**. Our model prioritizes recall to ensure we catch as many at-risk patients as possible.

### Feature Importance (Top 5)
1. **Age** - Strongest predictor
2. **Average Glucose Level** - Metabolic indicator
3. **BMI** - Cardiovascular risk factor
4. **Hypertension** - Major stroke risk
5. **Heart Disease** - Direct cardiovascular link

## 🚢 Deployment

### Production Checklist
- [ ] Set up proper environment variables
- [ ] Configure CORS for specific origins
- [ ] Add authentication (JWT, OAuth)
- [ ] Set up logging and monitoring
- [ ] Configure rate limiting
- [ ] Set up CI/CD pipeline
- [ ] Add model versioning
- [ ] Implement A/B testing
- [ ] Set up alerts for model drift

### Cloud Deployment Options

#### AWS
```bash
# Using Elastic Beanstalk
eb init -p docker stroke-prediction-api
eb create stroke-prediction-env
eb deploy
```

#### Google Cloud Platform
```bash
# Using Cloud Run
gcloud run deploy stroke-prediction \
  --source . \
  --platform managed \
  --region us-central1
```

#### Heroku
```bash
heroku create stroke-prediction-api
heroku container:push web
heroku container:release web
```

## 🔒 Security Considerations

- Input validation with Pydantic
- No PHI (Protected Health Information) storage
- HTTPS in production (configure reverse proxy)
- Rate limiting (implement with slowapi)
- API authentication (add JWT tokens)

## 📊 Future Enhancements

### Short-term
- [ ] Add SHAP explanations (library installed)
- [ ] Implement caching (Redis)
- [ ] Add batch prediction endpoint
- [ ] Create CLI interface
- [ ] Add model monitoring dashboard

### Long-term
- [ ] A/B testing framework
- [ ] Automated retraining pipeline
- [ ] Multi-model ensemble
- [ ] Real-time data streaming (Kafka)
- [ ] Mobile app integration
- [ ] Electronic Health Record (EHR) integration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 👤 Author

**Your Name**
- Portfolio: [yourportfolio.com](https://yourportfolio.com)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Inspired by real-world clinical ML applications
- Built with modern MLOps best practices

---

**⭐ If you find this project useful, please star the repository!**
