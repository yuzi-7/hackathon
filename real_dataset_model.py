#!/usr/bin/env python3
"""
Enhanced Device Performance Model using Real Smartphone Dataset
This version uses the actual smartphone dataset from CSV file
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os
import datetime

class RealDatasetPerformanceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.os_encoder = LabelEncoder()
        self.feature_names = [
            'num_cores', 'processor_speed', 'battery_capacity', 
            'fast_charging_available', 'ram_capacity', 'internal_memory', 
            'screen_size', 'refresh_rate', 'os_encoded',
            'resolution_height', 'resolution_width'
        ]
        
    def load_and_preprocess_data(self, csv_path='C:/Users/user/Downloads/smartphones.csv'):
        """Load and preprocess the real smartphone dataset"""
        print("📁 Loading smartphone dataset...")
        
        # Load the dataset
        data = pd.read_csv(csv_path)
        
        # Remove unwanted columns
        columns_to_remove = ['price', 'avg_rating', '5G_or_not', 'primary_camera_rear', 
                           'primary_camera_front', 'num_rear_cameras']
        data = data.drop(columns_to_remove, axis=1, errors='ignore')
        
        print(f"📊 Dataset loaded: {len(data)} devices")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Encode categorical variables
        data = self.encode_categorical_features(data)
        
        # Create performance categories
        data = self.create_performance_categories(data)
        
        # Clean and prepare final dataset
        data = self.prepare_final_dataset(data)
        
        return data
    
    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        print("🔧 Handling missing values...")
        
        # Fill missing processor_speed with median
        if 'processor_speed' in data.columns:
            data['processor_speed'].fillna(data['processor_speed'].median(), inplace=True)
        
        # Fill missing battery_capacity with median
        if 'battery_capacity' in data.columns:
            data['battery_capacity'].fillna(data['battery_capacity'].median(), inplace=True)
        
        # Fill missing fast_charging with 0 (no fast charging)
        if 'fast_charging' in data.columns:
            data['fast_charging'].fillna(0, inplace=True)
        
        # Fill missing fast_charging_available with 0
        if 'fast_charging_available' in data.columns:
            data['fast_charging_available'].fillna(0, inplace=True)
        
        # Fill missing os with 'android'
        if 'os' in data.columns:
            data['os'].fillna('android', inplace=True)
        
        # Fill missing extended_memory_available with 0
        if 'extended_memory_available' in data.columns:
            data['extended_memory_available'].fillna(0, inplace=True)
        
        return data
    
    def encode_categorical_features(self, data):
        """Encode categorical features"""
        print("🔄 Encoding categorical features...")
        
        # Encode OS
        if 'os' in data.columns:
            # Replace empty strings with 'android'
            data['os'] = data['os'].replace('', 'android')
            data['os'] = data['os'].replace('""', 'android')
            
            # Encode OS
            data['os_encoded'] = self.os_encoder.fit_transform(data['os'])
        
        return data
    
    def create_performance_categories(self, data):
        """Create performance categories based on device specifications"""
        print("⚡ Creating performance categories...")
        
        performance_scores = []
        
        for idx, row in data.iterrows():
            score = self.calculate_performance_score(row)
            performance_scores.append(score)
        
        data['performance_score'] = performance_scores
        
        # Create categories based on performance score
        def categorize_performance(score):
            if score >= 75:
                return 'HIGH'
            elif score >= 50:
                return 'MID'
            else:
                return 'LOW'
        
        data['performance_category'] = data['performance_score'].apply(categorize_performance)
        
        # Print category distribution
        category_counts = data['performance_category'].value_counts()
        print("\n📊 Performance Category Distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {category}: {count} devices ({percentage:.1f}%)")
        
        return data
    
    def calculate_performance_score(self, device_row):
        """Calculate performance score for a device"""
        score = 0
        
        # CPU Performance (40% weight)
        cores = device_row.get('num_cores', 4)
        speed = device_row.get('processor_speed', 2.0)
        
        if pd.notna(cores) and cores > 0:
            if cores >= 8:
                score += 20
            elif cores >= 6:
                score += 16
            elif cores >= 4:
                score += 12
            else:
                score += 8
        
        if pd.notna(speed) and speed > 0:
            if speed >= 3.0:
                score += 20
            elif speed >= 2.5:
                score += 16
            elif speed >= 2.0:
                score += 12
            else:
                score += 8
        
        # RAM Performance (25% weight)
        ram = device_row.get('ram_capacity', 4)
        if pd.notna(ram) and ram > 0:
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
        storage = device_row.get('internal_memory', 64)
        if pd.notna(storage) and storage > 0:
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
        screen_size = device_row.get('screen_size', 6.0)
        refresh_rate = device_row.get('refresh_rate', 60)
        
        if pd.notna(screen_size) and screen_size > 0:
            if screen_size >= 6.5:
                score += 5
            elif screen_size >= 6.0:
                score += 4
            else:
                score += 2
        
        if pd.notna(refresh_rate) and refresh_rate > 0:
            if refresh_rate >= 120:
                score += 5
            elif refresh_rate >= 90:
                score += 3
            else:
                score += 1
        
        # Battery Performance (10% weight)
        battery = device_row.get('battery_capacity', 4000)
        fast_charging = device_row.get('fast_charging_available', 0)
        
        if pd.notna(battery) and battery > 0:
            if battery >= 5000:
                score += 6
            elif battery >= 4000:
                score += 4
            else:
                score += 2
        
        if pd.notna(fast_charging) and fast_charging > 0:
            score += 4
        
        return min(100, max(0, score))
    
    def prepare_final_dataset(self, data):
        """Prepare final dataset for training"""
        print("🔧 Preparing final dataset...")
        
        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in data.columns:
                print(f"⚠️  Missing feature: {feature}")
                if feature == 'os_encoded':
                    data[feature] = 0
                else:
                    data[feature] = data[feature.replace('_encoded', '')].median() if feature.replace('_encoded', '') in data.columns else 0
        
        # Remove rows with missing target
        data = data.dropna(subset=['performance_category'])
        
        # Remove rows with too many missing features
        data = data.dropna(subset=self.feature_names, how='all')
        
        # Fill remaining missing values with median/mode
        for feature in self.feature_names:
            if data[feature].dtype in ['int64', 'float64']:
                data[feature].fillna(data[feature].median(), inplace=True)
            else:
                data[feature].fillna(data[feature].mode().iloc[0] if not data[feature].mode().empty else 0, inplace=True)
        
        print(f"✅ Final dataset: {len(data)} devices")
        return data
    
    def train_model(self, data):
        """Train the performance prediction model"""
        print("🚀 Training Random Forest model...")
        
        # Prepare features and target
        X = data[self.feature_names]
        y = data['performance_category']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        # Get predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\n📈 MODEL PERFORMANCE RESULTS")
        print("=" * 40)
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Testing Accuracy: {test_accuracy:.3f}")
        print(f"Dataset Size: {len(data)} devices")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return train_accuracy, test_accuracy
    
    def predict_device_performance(self, device_specs):
        """Predict performance for a device"""
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
        performance_score = self.calculate_performance_score(pd.Series(device_specs))
        
        return {
            'category': category,
            'confidence': confidence,
            'performance_score': performance_score,
            'probabilities': {
                class_name: prob for class_name, prob in 
                zip(self.label_encoder.classes_, probabilities)
            }
        }
    
    def save_model(self, model_path='real_dataset_model'):
        """Save the trained model"""
        print(f"💾 Saving model to {model_path}...")
        
        # Save model components
        joblib.dump(self.model, f'{model_path}_rf.pkl')
        joblib.dump(self.scaler, f'{model_path}_scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_path}_label_encoder.pkl')
        joblib.dump(self.os_encoder, f'{model_path}_os_encoder.pkl')
        
        # Save configuration
        config = {
            'feature_names': self.feature_names,
            'label_classes': self.label_encoder.classes_.tolist(),
            'os_classes': self.os_encoder.classes_.tolist(),
            'model_version': '3.0',
            'features_count': len(self.feature_names),
            'trained_on': datetime.datetime.now().isoformat()
        }
        
        with open(f'{model_path}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved successfully!")
    
    def predict_all_devices(self, data):
        """Predict performance for all devices"""
        print("\n🧪 TESTING ALL DEVICES")
        print("=" * 50)
        
        results = []
        total_devices = len(data)
        
        for idx, row in data.iterrows():
            device_name = f"{row.get('brand_name', 'Unknown')} {row.get('model', 'Device')}"
            
            # Predict performance
            prediction = self.predict_device_performance(row.to_dict())
            
            # Create result
            result = {
                'device_name': device_name,
                'actual_category': row['performance_category'],
                'predicted_category': prediction['category'],
                'confidence': round(prediction['confidence'], 3),
                'performance_score': round(prediction['performance_score'], 1),
                'probabilities': {k: round(v, 3) for k, v in prediction['probabilities'].items()}
            }
            
            results.append(result)
            
            # Print progress every 100 devices
            if (idx + 1) % 100 == 0 or (idx + 1) == total_devices:
                print(f"Processed {idx + 1}/{total_devices} devices...")
        
        print(f"\n✅ Completed predictions for all {total_devices} devices!")
        return results

def save_results_to_json(results, filename='real_dataset_results.json'):
    """Save results to JSON file"""
    output_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_version': '3.0',
        'dataset_source': 'Real smartphone dataset',
        'results': results
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main function to run the real dataset model"""
    print("🚀 Enhanced Device Performance Model - Real Dataset")
    print("=" * 60)
    
    # Initialize model
    model = RealDatasetPerformanceModel()
    
    try:
        # Load and preprocess data
        data = model.load_and_preprocess_data()
        
        # Train model
        train_acc, test_acc = model.train_model(data)
        
        # Save model
        model.save_model()
        
        # Test on all devices
        results = model.predict_all_devices(data)
        
        # Save results
        save_results_to_json(results)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final Model Performance: {test_acc:.3f} accuracy")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
