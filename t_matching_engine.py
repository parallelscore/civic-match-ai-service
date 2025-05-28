import json

import requests

# Load the test data
with open('test_voter_submission.json', 'r') as f:
    test_data = json.load(f)

# Send the request
response = requests.post('http://localhost:8000/api/v1/matching_engine', json=test_data)
# response = requests.post('https://elected-ai.parallelscore.com/api/v1/matching_engine', json=test_data)

# Print the response
print(f"Status code: {response.status_code}")
print(json.dumps(response.json(), indent=2))
