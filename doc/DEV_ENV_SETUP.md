# Notes and Instructions for Developers of this Project

## Python virtual environment maintenance commands
```
# Create python virtual environment
python3 -m venv testing-env
```
```
# Activate python virtual environment on UNIX
source testing-env/bin/activate
```
```
# Deactivate python virtual environment
deactivate
```
```
# Install required dependencies in the testing-env (activate first)
pip install -r requirements.txt
pip install .
```
```
# List current python packages in this virtual environment
pip list
```
```
# Save the current versions of all packages in the current environment
pip freeze > requirements.txt
```
## AWS credential setup
```
# Add your AWS credentials to your environment variables
export <YOUR_AWS_ACCESS_KEY_ID>
export <YOUR_AWS_SECRET_ACCESS_KEY>
export <YOUR_AWS_SESSION_TOKEN>
```

## Python Scripts for this project
```
# Run AI solution that parses a Corporate Actions PDF and returns structured data output
python3 src/parse_corporate_action.py
```
```
# Run AI solution that summarize a html page
python3 src/websummarization.py
```
