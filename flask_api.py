#!/usr/bin/env python3
"""
Flask API for Device Performance Model
Serves the trained model for Android backend integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app

class DevicePerformanceAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.os_encoder = None
        self.config = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Load model components
            self.model = joblib.load('real_dataset_model_rf.pkl')
            self.scaler = joblib.load('real_dataset_model_scaler.pkl')
            self.label_encoder = joblib.load('real_dataset_model_label_encoder.pkl')
            self.os_encoder = joblib.load('real_dataset_model_os_encoder.pkl')
            
            # Load configuration
            with open('real_dataset_model_config.json', 'r') as f:
                self.config = json.load(f)
                self.feature_names = self.config['feature_names']
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def calculate_performance_score(self, device_specs):
        """Calculate performance score for a device"""
        score = 0
        
        # CPU Performance (40% weight)
        cores = device_specs.get('num_cores', 4)
        speed = device_specs.get('processor_speed', 2.0)
        
        if cores and cores > 0:
            if cores >= 8:
                score += 20
            elif cores >= 6:
                score += 16
            elif cores >= 4:
                score += 12
            else:
                score += 8
        
        if speed and speed > 0:
            if speed >= 3.0:
                score += 20
            elif speed >= 2.5:
                score += 16
            elif speed >= 2.0:
                score += 12
            else:
                score += 8
        
        # RAM Performance (25% weight)
        ram = device_specs.get('ram_capacity', 4)
        if ram and ram > 0:
            if ram >= 16:
                score += 25
            elif ram >= 12:
                score += 22
            elif ram >= 8:
                score += 18
            elif ram >= 6:
                score += 14
            elif ram >= 4:
                score += 10
            else:
                score += 6
        
        # Storage Performance (15% weight)
        storage = device_specs.get('internal_memory', 64)
        if storage and storage > 0:
            if storage >= 512:
                score += 15
            elif storage >= 256:
                score += 12
            elif storage >= 128:
                score += 10
            elif storage >= 64:
                score += 8
            else:
                score += 4
        
        # Screen and Display (10% weight)
        screen_size = device_specs.get('screen_size', 6.0)
        refresh_rate = device_specs.get('refresh_rate', 60)
        
        if screen_size and screen_size > 0:
            if screen_size >= 6.5:
                score += 5
            elif screen_size >= 6.0:
                score += 4
            else:
                score += 2
        
        if refresh_rate and refresh_rate > 0:
            if refresh_rate >= 120:
                score += 5
            elif refresh_rate >= 90:
                score += 3
            else:
                score += 1
        
        # Battery Performance (10% weight)
        battery = device_specs.get('battery_capacity', 4000)
        fast_charging = device_specs.get('fast_charging_available', 0)
        
        if battery and battery > 0:
            if battery >= 5000:
                score += 6
            elif battery >= 4000:
                score += 4
            else:
                score += 2
        
        if fast_charging and fast_charging > 0:
            score += 4
        
        return min(100, max(0, score))
    
    def predict_performance(self, device_specs):
        """Predict device performance"""
        try:
            # Prepare input features
            features = []
            for feature in self.feature_names:
                if feature == 'os_encoded':
                    # Handle OS encoding
                    os_value = device_specs.get('os', 'android')
                    try:
                        encoded_os = self.os_encoder.transform([os_value])[0]
                    except:
                        encoded_os = 0  # Default to 0 if unknown OS
                    features.append(encoded_os)
                else:
                    features.append(device_specs.get(feature, 0))
            
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Calculate performance score
            performance_score = self.calculate_performance_score(device_specs)
            
            return {
                'status': 'success',
                'category': category,
                'confidence': float(confidence),
                'performance_score': float(performance_score),
                'probabilities': {
                    class_name: float(prob) for class_name, prob in 
                    zip(self.label_encoder.classes_, probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize the API
api = DevicePerformanceAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_version': api.config['model_version'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict device performance"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['num_cores', 'processor_speed', 'ram_capacity', 'internal_memory']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Make prediction
        result = api.predict_performance(data)
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict performance for multiple devices"""
    try:
        data = request.get_json()
        
        if not data or 'devices' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No devices data provided'
            }), 400
        
        devices = data['devices']
        results = []
        
        for i, device in enumerate(devices):
            result = api.predict_performance(device)
            result['device_index'] = i
            results.append(result)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_devices': len(devices),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_version': api.config['model_version'],
        'feature_names': api.config['feature_names'],
        'label_classes': api.config['label_classes'],
        'os_classes': api.config['os_classes'],
        'features_count': api.config['features_count'],
        'trained_on': api.config['trained_on']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
