# Note and Instructions for Developers of this Project

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
