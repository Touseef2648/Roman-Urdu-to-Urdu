import requests

# Define the URL for the API endpoint
url = "http://localhost:8000/process"

# Define the input data
data = {
    "text": "kya ?"
}

# Send a POST request to the API endpoint
response = requests.post(url, json=data)

# Check the response status code
if response.status_code == 200:
    # Extract the response data
    output_data = response.json()

    # Access the output text
    output_text = output_data["Urdu"]

    # Print the output text
    print("Output Text:", output_text)
else:
    print("Error:", response.status_code)
