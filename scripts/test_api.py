import requests
import json

def test_api():
    url = "http://localhost:8000/predict"
    
    # Sample data (from dataset or made up)
    data = {
        "Gender": "male",
        "Age": 25,
        "Height": 180.0,
        "Weight": 75.0,
        "Duration": 30.0,
        "Heart_Rate": 100.0,
        "Body_Temp": 40.0
    }
    
    print(f"Sending request to {url} with data:")
    print(json.dumps(data, indent=2))
    
    try:
        response = requests.post(url, json=data)
        print(f"\nResponse Code: {response.status_code}")
        print("Response Body:")
        print(response.json())
        
        if response.status_code == 200:
            print("\nAPI Test Passed!")
        else:
            print("\nAPI Test Failed!")
            
    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Is the API server running?")

if __name__ == "__main__":
    test_api()
