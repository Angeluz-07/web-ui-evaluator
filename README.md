# Web UI evaluator

## Set new environment in Windows
- `mkdir .venv`
- `python -m venv .venv`
- `.venv\Scripts\activate.bat`
- `pip install -r requirements.txt`

## Example training
- `python train_and_output.py`

## Example webapp usage in Windows
- `set FLASK_APP=web_app/app.py`
- `flask run`
- Open `http://127.0.0.1:5000/` in a browser