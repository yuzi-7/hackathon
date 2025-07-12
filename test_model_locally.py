#!/usr/bin/env python3
"""
Test script to run the Enhanced Device Performance Model locally
This simulates how the model would work on an Android device
"""

import numpy as np
import joblib
import json
import platform
import psutil
import os

class LocalDeviceAnalyzer:
    def __init__(self, model_path='enhanced_device_performance_model'):
        """Initialize the analyzer with saved model files"""
        print("🔄 Loading Enhanced Device Performance Model...")
        
        # Load the trained model components
        try:
            self.model = joblib.load(f'{model_path}_rf.pkl')
            self.scaler = joblib.load(f'{model_path}_scaler.pkl')
            self.label_encoder = joblib.load(f'{model_path}_encoder.pkl')
            
            # Load configuration
            with open(f'{model_path}_config.json', 'r') as f:
                self.config = json.load(f)
            
            self.feature_names = self.config['feature_names']
            print("✅ Model loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found. Please run 'enhanced_device_performance_model.py' first.")
            raise e
    
    def get_current_device_specs(self):
        """Get specifications of the current device (simulating Android device detection)"""
        print("\n🔍 Analyzing current device specifications...")
        
        # Get system information
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Convert to Android-like specifications
        specs = {
            'device_model': platform.node(),
            'cpu_cores': cpu_count,
            'cpu_frequency_mhz': cpu_freq.max if cpu_freq else 2000,  # MHz
            'ram_gb': memory.total / (1024**3),  # Convert bytes to GB
            'storage_gb': disk.total / (1024**3),  # Convert bytes to GB
            'android_version': 13,  # Simulated Android version
            'gpu_score': self.estimate_gpu_score(cpu_count, memory.total / (1024**3)),
            'architecture_score': 1,  # Assume modern 64-bit architecture
            'thermal_score': 0.8,  # Estimated thermal management
            'battery_capacity': 4000,  # Simulated battery capacity
            'screen_resolution_score': 2  # Simulated screen resolution (FHD)
        }
        
        return specs
    
    def estimate_gpu_score(self, cpu_cores, ram_gb):
        """Estimate GPU score based on CPU and RAM (simplified)"""
        base_score = 300
        if cpu_cores >= 8 and ram_gb >= 16:
            return 800  # High-end equivalent
        elif cpu_cores >= 4 and ram_gb >= 8:
            return 600  # Mid-high equivalent
        elif cpu_cores >= 4 and ram_gb >= 4:
            return 400  # Mid-range equivalent
        else:
            return base_score  # Basic equivalent
    
    def analyze_device(self):
        """Analyze the current device performance"""
        # Get device specifications
        specs = self.get_current_device_specs()
        
        # Prepare input features for the model
        features = np.array([
            specs['cpu_cores'],
            specs['cpu_frequency_mhz'],
            specs['ram_gb'],
            specs['storage_gb'],
            specs['android_version'],
            specs['gpu_score'],
            specs['architecture_score'],
            specs['thermal_score'],
            specs['battery_capacity'],
            specs['screen_resolution_score']
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Calculate additional metrics
        performance_score = self.calculate_performance_score(specs)
        concurrent_tasks = self.calculate_concurrent_tasks(specs)
        gaming_suitability = self.assess_gaming_suitability(specs)
        multitasking_efficiency = self.assess_multitasking_efficiency(specs)
        thermal_risk = self.assess_thermal_risk(specs)
        
        return {
            'specs': specs,
            'category': category,
            'confidence': confidence,
            'probabilities': {
                'HIGH': probabilities[np.where(self.label_encoder.classes_ == 'HIGH')[0][0]],
                'MID': probabilities[np.where(self.label_encoder.classes_ == 'MID')[0][0]],
                'LOW': probabilities[np.where(self.label_encoder.classes_ == 'LOW')[0][0]]
            },
            'performance_score': performance_score,
            'concurrent_tasks': concurrent_tasks,
            'gaming_suitability': gaming_suitability,
            'multitasking_efficiency': multitasking_efficiency,
            'thermal_risk': thermal_risk
        }
    
    def calculate_performance_score(self, specs):
        """Calculate performance score (same logic as the model)"""
        score = 0
        
        # CPU Performance (35% weight)
        cpu_score = 0
        cores = specs['cpu_cores']
        freq = specs['cpu_frequency_mhz']
        
        if cores >= 12: cpu_score += 20
        elif cores >= 8: cpu_score += 18
        elif cores >= 6: cpu_score += 15
        elif cores >= 4: cpu_score += 12
        else: cpu_score += 8
        
        if freq >= 3000: cpu_score += 15
        elif freq >= 2500: cpu_score += 13
        elif freq >= 2000: cpu_score += 11
        elif freq >= 1500: cpu_score += 9
        else: cpu_score += 6
        
        # RAM Performance (25% weight)
        ram_score = 0
        ram = specs['ram_gb']
        
        if ram >= 16: ram_score += 25
        elif ram >= 12: ram_score += 23
        elif ram >= 8: ram_score += 20
        elif ram >= 6: ram_score += 16
        elif ram >= 4: ram_score += 12
        elif ram >= 2: ram_score += 8
        else: ram_score += 4
        
        # Storage Performance (10% weight)
        storage_score = 0
        storage = specs['storage_gb']
        
        if storage >= 512: storage_score += 10
        elif storage >= 256: storage_score += 9
        elif storage >= 128: storage_score += 8
        elif storage >= 64: storage_score += 6
        elif storage >= 32: storage_score += 4
        else: storage_score += 2
        
        # GPU Performance (15% weight)
        gpu_score = min(15, specs['gpu_score'] / 80)
        
        # Android Version (5% weight)
        android_score = 0
        android_ver = specs['android_version']
        
        if android_ver >= 13: android_score += 5
        elif android_ver >= 11: android_score += 4
        elif android_ver >= 9: android_score += 3
        elif android_ver >= 8: android_score += 2
        else: android_score += 1
        
        # Calculate total
        total_score = (cpu_score + ram_score + storage_score + gpu_score + 
                      android_score + specs['architecture_score'] * 3 + 
                      specs['thermal_score'] * 4 + 
                      min(2, specs['battery_capacity'] / 3000) - 
                      specs['screen_resolution_score'] * 0.5)
        
        return min(100, max(0, total_score))
    
    def calculate_concurrent_tasks(self, specs):
        """Calculate estimated concurrent tasks"""
        base_tasks = 2
        base_tasks += specs['cpu_cores'] * 0.8
        ram_factor = min(1.0, specs['ram_gb'] / 8.0)
        base_tasks += ram_factor * 6
        freq_factor = min(1.0, specs['cpu_frequency_mhz'] / 2500)
        base_tasks += freq_factor * 3
        base_tasks *= specs['thermal_score']
        return int(min(base_tasks, 20))
    
    def assess_gaming_suitability(self, specs):
        """Assess gaming performance"""
        gaming_score = (specs['gpu_score'] * 0.5 + specs['ram_gb'] * 50 + 
                       specs['cpu_cores'] * specs['cpu_frequency_mhz'] / 1000 * 0.3) / 3
        
        if gaming_score > 400: return "EXCELLENT"
        elif gaming_score > 250: return "GOOD"
        elif gaming_score > 150: return "MODERATE"
        else: return "POOR"
    
    def assess_multitasking_efficiency(self, specs):
        """Assess multitasking efficiency"""
        ram_efficiency = min(1.0, specs['ram_gb'] / 12.0)
        cpu_efficiency = min(1.0, specs['cpu_cores'] / 8.0)
        android_efficiency = min(1.0, specs['android_version'] / 14.0)
        
        efficiency_score = (ram_efficiency * 0.5 + cpu_efficiency * 0.3 + android_efficiency * 0.2)
        
        if efficiency_score > 0.8: return "EXCELLENT"
        elif efficiency_score > 0.6: return "GOOD"
        elif efficiency_score > 0.4: return "MODERATE"
        else: return "POOR"
    
    def assess_thermal_risk(self, specs):
        """Assess thermal throttling risk"""
        cpu_intensity = (specs['cpu_cores'] * specs['cpu_frequency_mhz']) / 20000
        risk_score = 1 - (specs['thermal_score'] * 0.7 + (1 - min(cpu_intensity, 1.0)) * 0.3)
        
        if risk_score > 0.7: return "HIGH"
        elif risk_score > 0.4: return "MEDIUM"
        else: return "LOW"

def display_results(result):
    """Display the analysis results in a formatted way"""
    print("\n" + "="*60)
    print("🔍 DEVICE PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    specs = result['specs']
    
    # Device Information
    print(f"\n📱 DEVICE INFORMATION")
    print("-" * 30)
    print(f"Device: {specs['device_model']}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Hardware Specifications
    print(f"\n⚙️ HARDWARE SPECIFICATIONS")
    print("-" * 30)
    print(f"CPU Cores: {specs['cpu_cores']}")
    print(f"CPU Frequency: {specs['cpu_frequency_mhz']:.0f} MHz")
    print(f"RAM: {specs['ram_gb']:.1f} GB")
    print(f"Storage: {specs['storage_gb']:.0f} GB")
    print(f"GPU Score: {specs['gpu_score']:.0f}")
    
    # Performance Results
    print(f"\n🏆 PERFORMANCE RESULTS")
    print("-" * 30)
    category_emoji = {"HIGH": "🔥", "MID": "⚡", "LOW": "📱"}
    print(f"Category: {category_emoji.get(result['category'], '❓')} {result['category']} PERFORMANCE")
    print(f"Score: {result['performance_score']:.1f}/100")
    print(f"Confidence: {result['confidence']:.1f}%")
    
    # Capability Assessment
    print(f"\n📊 CAPABILITY ASSESSMENT")
    print("-" * 30)
    print(f"Concurrent Tasks: {result['concurrent_tasks']} apps")
    
    performance_emoji = {"EXCELLENT": "🌟", "GOOD": "✅", "MODERATE": "⚠️", "POOR": "❌"}
    thermal_emoji = {"LOW": "❄️", "MEDIUM": "🌡️", "HIGH": "🔥"}
    
    print(f"Gaming: {performance_emoji.get(result['gaming_suitability'], '❓')} {result['gaming_suitability']}")
    print(f"Multitasking: {performance_emoji.get(result['multitasking_efficiency'], '❓')} {result['multitasking_efficiency']}")
    print(f"Thermal Risk: {thermal_emoji.get(result['thermal_risk'], '❓')} {result['thermal_risk']}")
    
    # Prediction Confidence
    print(f"\n🎯 PREDICTION CONFIDENCE")
    print("-" * 30)
    for category in ['HIGH', 'MID', 'LOW']:
        probability = result['probabilities'][category]
        print(f"{category}: {probability:.1%}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if result['category'] == 'HIGH':
        print("• Perfect for demanding applications and heavy multitasking")
        print("• Excellent for gaming, video editing, and professional work")
        print("• Can handle multiple resource-intensive tasks simultaneously")
    elif result['category'] == 'MID':
        print("• Great for everyday tasks and moderate multitasking")
        print("• Good for social media, productivity, and light gaming")
        print("• Close unused applications for optimal performance")
    else:
        print("• Best for essential applications and basic tasks")
        print("• Use lightweight alternatives when possible")
        print("• Regular maintenance and cleanup recommended")
    
    if result['thermal_risk'] == 'HIGH':
        print("• Monitor temperature during intensive tasks")
        print("• Consider cooling solutions for sustained workloads")

def main():
    """Main function to run the device analysis"""
    print("🚀 Enhanced Device Performance Analyzer")
    print("="*60)
    
    try:
        # Initialize the analyzer
        analyzer = LocalDeviceAnalyzer()
        
        # Analyze the current device
        result = analyzer.analyze_device()
        
        # Display results
        display_results(result)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Model files location: {os.getcwd()}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Make sure you've run 'enhanced_device_performance_model.py' first to generate the model files.")

if __name__ == "__main__":
    main()
