import os
import tensorflow as tf
import numpy as np
import pickle

# import Flask dependencies
from flask import Flask, request, render_template

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# saver = tf.train.Saver()

# Define Flask app
app = Flask(__name__, static_url_path='/static')


# Define apps home page
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
	app.run(port=5000, debug=True)