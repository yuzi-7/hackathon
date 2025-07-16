# Phone Performance Model API

This directory contains a complete machine learning model for predicting phone performance and categorizing devices into LOW, MID, and HIGH specs.

## 📁 Files Overview

### 🚀 **Core API Files**
- **`flask_api.py`** - Main Flask API server that serves the model predictions
- **`test_api.py`** - Comprehensive test suite for the API endpoints

### 🤖 **Model Files**
- **`real_dataset_model_rf.pkl`** - Trained Random Forest model (96.4% accuracy)
- **`real_dataset_model_scaler.pkl`** - Feature scaler for input normalization
- **`real_dataset_model_label_encoder.pkl`** - Label encoder for performance categories
- **`real_dataset_model_os_encoder.pkl`** - OS encoder for operating system features
- **`real_dataset_model_config.json`** - Model configuration and metadata

### 📊 **Data Files**
- **`categorized_phones.json`** - Complete phone database with categorization (980 phones)
- **`requirements.txt`** - Python dependencies

### 🔧 **Environment**
- **`.venv/`** - Python virtual environment with all dependencies

## 📱 Phone Categories

The model categorizes phones into three performance tiers:

- **🔥 HIGH SPECS** (490 phones - 50.0%): Score ≥ 75/100
- **📱 MID SPECS** (456 phones - 46.5%): Score ≥ 50/100 and < 75/100
- **⚡ LOW SPECS** (34 phones - 3.5%): Score < 50/100

## 🚀 How to Run

1. **Activate virtual environment:**
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start the API server:**
   ```bash
   python flask_api.py
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

## 🌐 API Endpoints

- **`GET /health`** - Health check
- **`POST /predict`** - Single device prediction
- **`POST /batch_predict`** - Batch device predictions
- **`GET /model_info`** - Model information

## 📈 Model Performance

- **Training Accuracy:** 99.1%
- **Testing Accuracy:** 96.4%
- **Model Type:** Random Forest Classifier
- **Features:** 11 device specifications
- **Dataset:** 980 real smartphone devices

## 🔧 Required Input Features

```json
{
  "num_cores": 8,
  "processor_speed": 2.8,
  "battery_capacity": 4000,
  "fast_charging_available": 1,
  "ram_capacity": 8,
  "internal_memory": 128,
  "screen_size": 6.1,
  "refresh_rate": 90,
  "os": "android",
  "resolution_height": 2400,
  "resolution_width": 1080
}
```

## 🎯 Example Response

```json
{
  "status": "success",
  "category": "HIGH",
  "confidence": 0.9621,
  "performance_score": 79.0,
  "probabilities": {
    "HIGH": 0.9621,
    "MID": 0.0379,
    "LOW": 0.0000
  }
}
```

## 🔗 Integration

This API is ready for integration with Android applications or any system that needs phone performance predictions.
