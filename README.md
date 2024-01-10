# Web UI evaluator

Project for the course of Artificial Intelligence, initially developed in January, 2020
## Description
Web application to classify web user interfaces into three classes:
- appealing contrast
- minimalism
- visual load

The app uses a keras model trained to classify web UIs screenshots. The webapp allows to upload an image and show the probability to belong to one of the three classes.
## Examples
### Results
![one](https://github.com/Angeluz-07/web-ui-evaluator/blob/master/example_results.jpg)
### Usage
![two](https://github.com/Angeluz-07/web-ui-evaluator/blob/master/example_usage.jpg)

## Set new environment
### on Windows
```
mkdir .venv
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
set FLASK_APP=web_app/app.py
```

### on Linux
```
mkdir .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source .env
``````

## Run web app
```
flask run
```
then open `http://127.0.0.1:5000/` in a browser.
