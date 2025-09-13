import os
from dotenv import load_dotenv

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Looking for .env file in: {script_dir}")

# Construct the path to the .env file
dotenv_path = r"C:\Users\subhi\OneDrive\Desktop\ML Project\.env"
print(f"Full path to .env file: {dotenv_path}")

# Check if the .env file exists
if os.path.exists(dotenv_path):
    print(".env file found")
else:
    print(".env file not found")

# Load the .env file
load_dotenv(dotenv_path)

# Try to get the API key
api_key = os.getenv('COHERE_API_KEY')
if api_key:
    print(f"API key found: {api_key[:4]}...{api_key[-4:]}")
else:
    print("No API key found in .env file")

# Print all environment variables (be careful with this in production!)
print("\nAll environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value[:4]}...{value[-4:] if len(value) > 8 else value}")
