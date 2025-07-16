#!/usr/bin/env python3
"""
Test script for Device Performance Model API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_single_prediction():
    """Test single device prediction"""
    print("\nTesting single prediction...")
    
    # Sample device specs
    device_specs = {
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
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=device_specs,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Prediction: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200 and result.get('status') == 'success':
            print(f"✅ Device Performance: {result['category']}")
            print(f"✅ Confidence: {result['confidence']:.2%}")
            print(f"✅ Performance Score: {result['performance_score']}")
            return True
        else:
            print(f"❌ Prediction failed: {result}")
            return False
            
    except Exception as e:
        print(f"Single prediction failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print("\nTesting batch prediction...")
    
    devices = [
        {
            "num_cores": 8,
            "processor_speed": 3.0,
            "battery_capacity": 5000,
            "fast_charging_available": 1,
            "ram_capacity": 12,
            "internal_memory": 256,
            "screen_size": 6.7,
            "refresh_rate": 120,
            "os": "android",
            "resolution_height": 2400,
            "resolution_width": 1080
        },
        {
            "num_cores": 4,
            "processor_speed": 2.0,
            "battery_capacity": 3000,
            "fast_charging_available": 0,
            "ram_capacity": 4,
            "internal_memory": 64,
            "screen_size": 5.5,
            "refresh_rate": 60,
            "os": "android",
            "resolution_height": 1920,
            "resolution_width": 1080
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json={"devices": devices},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200 and result.get('status') == 'success':
            print(f"✅ Batch prediction successful")
            print(f"✅ Total devices: {result['total_devices']}")
            
            for i, device_result in enumerate(result['results']):
                if device_result.get('status') == 'success':
                    print(f"Device {i+1}: {device_result['category']} "
                          f"(Score: {device_result['performance_score']}, "
                          f"Confidence: {device_result['confidence']:.2%})")
                else:
                    print(f"Device {i+1}: Error - {device_result.get('message', 'Unknown error')}")
            
            return True
        else:
            print(f"❌ Batch prediction failed: {result}")
            return False
            
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Model Info: {json.dumps(result, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print("\nTesting error handling...")
    
    # Test with missing required fields
    incomplete_specs = {
        "num_cores": 8,
        # Missing other required fields
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_specs,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 400 and result.get('status') == 'error':
            print(f"✅ Error handling working: {result['message']}")
            return True
        else:
            print(f"❌ Error handling failed: {result}")
            return False
            
    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False

def run_performance_test():
    """Run performance test with multiple requests"""
    print("\nRunning performance test...")
    
    device_specs = {
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
    
    num_requests = 10
    successful_requests = 0
    total_time = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=device_specs,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                successful_requests += 1
                total_time += (end_time - start_time)
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if successful_requests > 0:
        avg_response_time = total_time / successful_requests
        print(f"✅ Performance test results:")
        print(f"   - Successful requests: {successful_requests}/{num_requests}")
        print(f"   - Average response time: {avg_response_time:.3f} seconds")
        print(f"   - Success rate: {successful_requests/num_requests:.2%}")
        return True
    else:
        print("❌ Performance test failed - no successful requests")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Error Handling", test_error_handling),
        ("Performance Test", run_performance_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API is ready for integration.")
    else:
        print("⚠️  Some tests failed. Please check the API setup.")

if __name__ == "__main__":
    main()
