# Create venv (recommended Python 3.12)
py -3.12 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# If activation blocked
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run project
python ca.py