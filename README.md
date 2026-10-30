# 🍕 Food Delivery Time Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end Machine Learning web application designed to predict food delivery durations in minutes. The project combines exploratory data analysis, geospatial feature engineering (Haversine formula), multi-model benchmarking (Linear Regression, Decision Tree, Random Forest, XGBoost), hyperparameter tuning, and a production-ready **FastAPI** backend coupled with an interactive web UI.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [Dataset](#dataset)
  - [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
  - [Model Benchmarking & Results](#model-benchmarking--results)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
  - [Endpoints](#endpoints)
  - [Sample Request & Response](#sample-request--response)
- [Installation & Getting Started](#-installation--getting-started)
- [Deploying to Render](#-deploying-to-render)
- [Running Standalone Inference](#-running-standalone-inference)
- [Web Interface](#-web-interface)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Overview

Accurate Estimated Time of Arrival (ETA) is a critical component for food delivery platforms, directly impacting customer satisfaction, rider dispatching, and kitchen operations.

This project delivers:
- **Predictive Accuracy:** Uses a tuned **XGBoost Regressor** capable of estimating delivery times within an average margin of ~4 minutes ($R^2 \approx 0.82$).
- **Intelligent Preprocessing:** Computes great-circle geospatial distance via the Haversine formula from restaurant and drop-off coordinates, handles missing values, and normalizes feature distributions.
- **Full-Stack Serving:** Ships with a high-performance **FastAPI** service serving both REST endpoints and a frontend dashboard with real-time ETA animations.

---

## 🏛 System Architecture

```mermaid
flowchart LR
    subgraph Client ["Client Layer"]
        UI["Web Interface (HTML/CSS/JS)"]
        API_Client["API Client / cURL"]
    end

    subgraph Server ["FastAPI Backend (main.py)"]
        CORS["CORS Middleware"]
        Pydantic["Pydantic Validation (DeliveryOrder)"]
        FeatureBuilder["Feature Matrix Builder (build_features)"]
        Scaler["StandardScaler (standard_scaler.pkl)"]
        XGB["XGBoost Model (xgb_best.json)"]
    end

    UI -->|POST /predict| CORS
    API_Client -->|POST /predict| CORS
    CORS --> Pydantic
    Pydantic --> FeatureBuilder
    FeatureBuilder --> Scaler
    Scaler --> XGB
    XGB -->|{"predicted_delivery_time_min": ...}| Client
```

---

## 🔬 Machine Learning Workflow

### Dataset
The model was trained on a comprehensive food delivery dataset comprising **45,593 records** with 20 raw features:
- **Delivery Partner Info:** Partner ID, Age, Ratings.
- **Geographic Coordinates:** Restaurant latitude/longitude and delivery location latitude/longitude.
- **Temporal Metrics:** Order Date, Time Ordered, Time Picked up.
- **Contextual Conditions:** Weather conditions (Sunny, Cloudy, Fog, Sandstorms, Stormy, Windy), Traffic density (Low, Medium, High, Jam), Vehicle condition (0–3), Festival status (Yes/No), and City type (Urban, Semi-Urban, Metropolitan).
- **Order Details:** Type of order (Meal, Snack, Drinks, Buffet), Type of vehicle (Motorcycle, Scooter, Electric Scooter, Bicycle), and number of multiple deliveries.
- **Target Variable:** `Time_taken(min)` — actual duration in minutes.

### Feature Engineering & Preprocessing

1. **Geospatial Distance (Haversine Formula):**
   Calculates the great-circle distance between restaurant and delivery drop-off points:
   $$\text{distance} = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
2. **Temporal Feature Extraction:**
   - Parsed pickup and order timestamps to extract `order_hour`.
3. **Categorical Encoding:**
   - **Ordinal Encoding:** `Road_traffic_density` mapped ordered scale (`Low: 0`, `Medium: 1`, `High: 2`, `Jam: 3`).
   - **One-Hot Encoding with Baseline Drops:** Nominal variables (`Type_of_order`, `Type_of_vehicle`, `Weatherconditions`, `City`, `Festival`) one-hot encoded with reference categories (e.g., `Buffet`, `bicycle`, `Cloudy`, `Metropolitian`, `Festival_No`) dropped to prevent dummy variable trap and multicollinearity.
4. **Standardization:**
   - Continuous inputs scaled using scikit-learn's `StandardScaler`, persisted as `standard_scaler.pkl`.

### Model Benchmarking & Results

Four regression architectures were evaluated using **Root Mean Squared Error (RMSE)** and **Coefficient of Determination ($R^2$)**:

| Model | Test RMSE (min) | Test $R^2$ | Notes |
| :--- | :---: | :---: | :--- |
| **Linear Regression** (Baseline) | `6.21` | `0.57` | Underfits non-linear interactions |
| **Decision Tree Regressor** | `5.46` | `0.67` | Severe training set overfitting ($R^2=1.0$) |
| **Random Forest Regressor** | `4.03` | `0.82` | Strong ensemble performance |
| **XGBoost Regressor (Tuned)** 🏆 | **`3.99 - 4.40`** | **`0.78 - 0.82`** | **Best balance of accuracy & generalization** |

The tuned XGBoost model demonstrated the highest generalizability and is serialized as both `xgb_best.json` (for cross-platform deployment) and `xgb_best.pkl`.

---

## 📁 Project Structure

```text
Food_Delivery_Time_Prediction/
│
├── static/                              # Frontend static assets
│   ├── index.html                       # Responsive HTML5 user interface
│   ├── style.css                        # Modern CSS styling & glassmorphism
│   └── script.js                        # Client-side validation & fetch API
│
├── Food delivery.csv                    # Raw training dataset (45k+ records)
├── Food_Delivery_Time_Prediction2.ipynb # Complete EDA, modeling & evaluation notebook
├── main.py                              # FastAPI backend & serving pipeline
├── test.py                              # Standalone CLI inference verification script
├── requirements.txt                     # Core Python dependencies
├── standard_scaler.pkl                  # Serialized scikit-learn StandardScaler
├── xgb_best.json                        # Serialized XGBoost model (JSON format)
├── xgb_best.pkl                         # Serialized XGBoost model (Joblib pickle)
├── render.yaml                          # Render Blueprint infrastructure specification
├── .python-version                      # Pinned Python version (3.11.9) for Render
├── .gitignore                           # Git ignore rules for virtualenvs and caches
└── README.md                            # Comprehensive project documentation
```

---

## 🔌 API Reference

FastAPI exposes automatic, interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check status |
| `POST` | `/predict` | Predicts delivery time given order parameters |
| `GET` | `/` | Serves the web application user interface |

### Sample Request & Response

#### Request (`POST /predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Delivery_person_Age": 28,
       "Delivery_person_Ratings": 4.8,
       "distance_km": 6.5,
       "order_hour": 19,
       "Road_traffic_density": 2,
       "Vehicle_condition": 2,
       "multiple_deliveries": 0,
       "Type_of_order": "Meal",
       "Type_of_vehicle": "motorcycle",
       "Festival": "No",
       "City": "Urban",
       "Weatherconditions": "Sunny"
     }'
```

#### Response (`200 OK`)
```json
{
  "predicted_delivery_time_min": 24.32
}
```

---

## 🚀 Installation & Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository
```bash
git clone https://github.com/your-username/Food_Delivery_Time_Prediction.git
cd Food_Delivery_Time_Prediction
```

### 3. Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt uvicorn
```

### 5. Launch the Application
Run the Uvicorn ASGI server:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
- **Interactive Web App:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Docs:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## ☁️ Deploying to Render

This application is fully pre-configured for free cloud hosting on [Render](https://render.com).

### Step 1: Push Code to GitHub

Make sure all changes are committed and pushed to your remote GitHub repository:

```bash
git add .
git commit -m "Configure project for Render deployment"
git push origin main
```

### Step 2: Deploy on Render

Choose either of the two approaches below:

#### Option A: One-Click Blueprint Deployment (Recommended)
Because the project includes a `render.yaml` specification, Render can configure everything automatically:
1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. Click **New +** at the top right and select **Blueprint**.
3. Connect your GitHub account and select your `Food-Delivery-Time-Prediction` repository.
4. Render will detect `render.yaml` and parse the service settings.
5. Click **Apply**. Render will automatically provision the build and deploy your service!

#### Option B: Manual Web Service Setup
1. In your [Render Dashboard](https://dashboard.render.com/), click **New +** > **Web Service**.
2. Select **Build and deploy from a Git repository** and connect your repository.
3. Fill in the deployment details:
   - **Name:** `food-delivery-time-prediction`
   - **Region:** Pick a region near you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
4. Expand **Advanced**:
   - Add an Environment Variable:
     - **Key:** `PYTHON_VERSION`
     - **Value:** `3.11.9`
   - Set **Health Check Path:** `/health`
   - Set **Auto-Deploy:** `Yes`
5. Click **Create Web Service**.

> [!TIP]
> **Free Tier Tip:** Render free web services spin down after 15 minutes of inactivity. The first request after a period of dormancy may take 30–50 seconds to initialize ("cold start"). Subsequent requests will be near-instantaneous.

---

## 🧪 Running Standalone Inference

To test model inference directly via Python without launching the FastAPI server:

```bash
python test.py
```

Expected output:
```text
scaled: [ 1.3458  0.6472  0.0382 -0.3204 -0.3129 -1.2241  0.4439]
prediction: 26.54...
```

---

## 🎨 Web Interface

The frontend provides an intuitive delivery time estimation dashboard:
- **Input Sections:** Organizes parameters logically by **Trip** (distance, traffic, hour), **Rider & Vehicle** (age, rating, vehicle condition, vehicle type), and **Conditions** (weather, city tier, order type, festival).
- **Dynamic Feedback:** Computes predictions asynchronously and animates delivery route progress across a visual path.
- **Contextual Summary:** Classifies predicted ETA with human-friendly descriptions (e.g., *"Quick one. Expect it fast."*, *"A normal delivery time."*, *"Conditions are working against this order."*).

---

## 🔮 Future Enhancements

- [ ] **Live Mapping Integration:** Integrate OpenStreetMap / Leaflet to display actual routing directions and turn-by-turn distance instead of linear Haversine approximation.
- [ ] **Dockerization:** Add `Dockerfile` and `docker-compose.yml` for containerized multi-platform deployment.
- [ ] **Live Weather API:** Automatically fetch real-time weather conditions based on city coordinates.
- [ ] **Model Observability:** Implement Prometheus metrics and drift monitoring with Evidently AI.

---

## 📄 License

This project is licensed under the MIT License - feel free to adapt and use it for academic and commercial purposes.
