#!/usr/bin/env python3
"""
Interactive Phone Categorization System
Takes user input for phone specifications and categorizes them into LOW, MID, or HIGH tiers
"""

import joblib
import numpy as np
import json
from datetime import datetime
import sys
import warnings

# Suppress sklearn version warnings for better user experience
warnings.filterwarnings('ignore', category=UserWarning)

class PhoneCategorizer:
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
            print("🔄 Loading categorization model...")
            
            # Load model components
            self.model = joblib.load('real_dataset_model_rf.pkl')
            self.scaler = joblib.load('real_dataset_model_scaler.pkl')
            self.label_encoder = joblib.load('real_dataset_model_label_encoder.pkl')
            self.os_encoder = joblib.load('real_dataset_model_os_encoder.pkl')
            
            # Load configuration
            with open('real_dataset_model_config.json', 'r') as f:
                self.config = json.load(f)
                self.feature_names = self.config['feature_names']
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def calculate_performance_score(self, device_specs):
        """Calculate performance score for a device"""
        score = 0
        
        # CPU Performance (50% weight)
        cores = device_specs.get('num_cores', 4)
        speed = device_specs.get('processor_speed', 2.0)
        
        if cores and cores > 0:
            if cores >= 8:
                score += 25
            elif cores >= 6:
                score += 20
            elif cores >= 4:
                score += 15
            else:
                score += 10
        
        if speed and speed > 0:
            if speed >= 3.0:
                score += 25
            elif speed >= 2.5:
                score += 20
            elif speed >= 2.0:
                score += 15
            else:
                score += 10
        
        # RAM Performance (30% weight)
        ram = device_specs.get('ram_capacity', 4)
        if ram and ram > 0:
            if ram >= 16:
                score += 30
            elif ram >= 12:
                score += 25
            elif ram >= 8:
                score += 20
            elif ram >= 6:
                score += 15
            elif ram >= 4:
                score += 10
            else:
                score += 5
        
        # Storage Performance (20% weight)
        storage = device_specs.get('internal_memory', 64)
        if storage and storage > 0:
            if storage >= 512:
                score += 20
            elif storage >= 256:
                score += 16
            elif storage >= 128:
                score += 12
            elif storage >= 64:
                score += 8
            else:
                score += 4
        
        return min(100, max(0, score))
    
    def predict_phone_category(self, device_specs):
        """Predict phone category"""
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
                'category': category,
                'confidence': float(confidence),
                'performance_score': float(performance_score),
                'probabilities': {
                    class_name: float(prob) for class_name, prob in 
                    zip(self.label_encoder.classes_, probabilities)
                }
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_user_input(self):
        """Get phone specifications from user input"""
        print("\n📱 PHONE CATEGORIZATION SYSTEM")
        print("=" * 50)
        print("Enter your phone specifications:")
        print("-" * 50)
        
        device_specs = {}
        
        # Get phone name
        device_specs['name'] = input("📱 Phone Name/Model: ").strip()
        
        # CPU specifications
        print("\n🔧 CPU SPECIFICATIONS:")
        while True:
            try:
                cores = int(input("   Number of CPU cores (e.g., 4, 6, 8): "))
                if cores > 0:
                    device_specs['num_cores'] = cores
                    break
                else:
                    print("   Please enter a positive number.")
            except ValueError:
                print("   Please enter a valid number.")
        
        while True:
            try:
                speed = float(input("   CPU speed in GHz (e.g., 2.4, 3.0): "))
                if speed > 0:
                    device_specs['processor_speed'] = speed
                    break
                else:
                    print("   Please enter a positive number.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Memory specifications
        print("\n💾 MEMORY SPECIFICATIONS:")
        while True:
            try:
                ram = int(input("   RAM capacity in GB (e.g., 4, 8, 16): "))
                if ram > 0:
                    device_specs['ram_capacity'] = ram
                    break
                else:
                    print("   Please enter a positive number.")
            except ValueError:
                print("   Please enter a valid number.")
        
        while True:
            try:
                storage = int(input("   Internal storage in GB (e.g., 64, 128, 256): "))
                if storage > 0:
                    device_specs['internal_memory'] = storage
                    break
                else:
                    print("   Please enter a positive number.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Battery specifications
        print("\n🔋 BATTERY SPECIFICATIONS:")
        while True:
            try:
                battery_capacity = int(input("   Battery capacity in mAh (e.g., 3000, 4000): "))
                if battery_capacity > 0:
                    device_specs['battery_capacity'] = battery_capacity
                    break
                else:
                    print("   Please enter a positive number.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # OS specification
        print("\n🖥️ OPERATING SYSTEM:")
        device_specs['os'] = input("   Operating System (android/ios/other): ").strip().lower()
        
        return device_specs

def main():
    """Main function to run the categorization system"""
    try:
        # Initialize categorizer
        categorizer = PhoneCategorizer()
        
        # Get user input
        user_input = categorizer.get_user_input()
        
        # Predict category
        result = categorizer.predict_phone_category(user_input)
        
        # Display result
        if result:
            print("\n" + "=" * 60)
            print("🌟 PHONE CATEGORIZATION RESULT")
            print("=" * 60)
            
            # Phone name
            print(f"📱 Phone: {user_input.get('name', 'Unknown')}")
            print()
            
            # Category with emoji
            category_emoji = {
                'HIGH': '🔥',
                'MID': '📱', 
                'LOW': '⚡'
            }
            emoji = category_emoji.get(result['category'], '📱')
            print(f"🎯 Category: {emoji} {result['category']} SPECS")
            print(f"🎲 Confidence: {result['confidence']:.1%}")
            print(f"📊 Performance Score: {result['performance_score']:.1f}/100")
            
            # Performance bar
            score = result['performance_score']
            bar_length = 20
            filled_length = int(bar_length * score // 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"📈 Performance: [{bar}] {score:.1f}%")
            
            print("\n📋 Detailed Probabilities:")
            print("-" * 30)
            for category, probability in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                emoji = category_emoji.get(category, '📱')
                bar_length = 15
                filled_length = int(bar_length * probability)
                prob_bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"   {emoji} {category:4s}: [{prob_bar}] {probability:.1%}")
            
            # Category explanation
            print("\n💡 Category Explanation:")
            print("-" * 30)
            if result['category'] == 'HIGH':
                print("🔥 HIGH SPECS: Flagship device with premium performance")
                print("   Perfect for gaming, heavy multitasking, and professional use")
            elif result['category'] == 'MID':
                print("📱 MID SPECS: Balanced performance for everyday use")
                print("   Great for daily tasks, moderate gaming, and productivity")
            else:
                print("⚡ LOW SPECS: Entry-level device with basic performance")
                print("   Suitable for basic tasks, calls, and light app usage")
            
            print("\n" + "=" * 60)
            
        else:
            print("\n❌ Unable to predict the category.")
            print("Please check your input and try again.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using the Phone Categorization System!")
        print("Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again.")

if __name__ == "__main__":
    main()
