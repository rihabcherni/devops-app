# app/__init__.py

from flask import Flask

# Create a Flask application instance 
app = Flask(__name__)

# Import the main application module
from SVM_service.app import app  


if __name__ == '__main__':
    app.run()
