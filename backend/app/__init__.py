from flask import Flask

app = Flask(__name__)

# Load the model
from .models import load_model
model = load_model()
