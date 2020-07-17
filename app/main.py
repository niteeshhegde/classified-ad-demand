import constants

from flask import Flask, request, jsonify, make_response, render_template
import re
import pickle
import numpy as np
import pandas as pd
import json
import requests
from keras.preprocessing import text, sequence

stopWords = open("russian").read().splitlines()
tokenizer_desc = pickle.load(open('tokenizer_desc.pickle', 'rb'))
tokenizer_title = pickle.load(open('tokenizer_title.pickle', 'rb'))
sc = pickle.load(open('scaler.pickle', 'rb'))

MODEL_URI = 'http://35.199.176.26:8501/v1/models/my_saved_model:predict'

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


def clean_text(words, stop_words):
    words = ''.join([i for i in words if not i.isdigit()])
    words = ' '.join([re.sub(r'[^\w\s]', ' ', word) for word in words.split()])
    words = ' '.join([word.lower().strip() for word in words.split() if
                            word.lower().strip() not in (stop_words) and len(word) >= 3])
    return words


def tokenize_text(texts, token, len=50):
    texts = token.texts_to_sequences(texts)
    texts = sequence.pad_sequences(texts, maxlen=len)
    return texts


def process_request(req):
    x = np.zeros(82)

    region_en = "region_en_" + req.get("region_en")
    x[constants.columns.index(region_en)] = 1

    category_name_en = "category_name_en_" + req.get("category_name_en")
    x[constants.columns.index(category_name_en)] = 1

    user_type = "user_type_" + req.get("user_type")
    x[constants.columns.index(user_type)] = 1

    x_desc = clean_text(req.get("description"), stopWords)
    x_desc = tokenize_text([x_desc], tokenizer_desc, 50)

    x_title = clean_text(req.get("title"), stopWords)
    x_title = tokenize_text([x_title], tokenizer_title, 20)

    x_weekend = 1 if (req.get("weekday") in ["Saturday", "Sunday"]) else 0
    x_price = req.get("price")
    x_description_len = len(req.get("description").split())
    x_title_len = len(req.get("title").split())
    x_param_1_len = len(req.get("param_1").split())
    x_param_2_len = len(req.get("param_2").split())
    x_param_3_len = len(req.get("param_3").split())

    df = {
        'price': [x_price],
        'description_len': [x_description_len],
        'title_len': [x_title_len],
        'param_1_len': [x_param_1_len],
        'param_2_len': [x_param_2_len],
        'param_3_len': [x_param_3_len]
    }

    df = pd.DataFrame(df)

    sc.n_features_in_ = 6
    df = sc.transform(df)

    x[0] = x_weekend
    x[1] = df[0][0]
    x[2] = df[0][1]
    x[3] = df[0][2]
    x[4] = df[0][3]
    x[5] = df[0][4]
    x[6] = df[0][5]

    x_data = np.asarray(x)

    return x_title, x_desc, x_data


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def calc_class():
    if(request.content_type == 'application/x-www-form-urlencoded'):
        req = request.values
    elif(request.content_type == 'application/json'):
        req = request.json
    x_title, x_desc, x_data = process_request(req)

    data = json.dumps({
        'instances': [
            {
                "nlp_input_desc": x_desc[0].tolist(),
                "nlp_input_title": x_title[0].tolist(),
                "meta_input": x_data.tolist()
            }
        ]
    })

    response = requests.post(MODEL_URI, data=data.encode('utf-8'))
    result = json.loads(response.text)
    prediction = np.squeeze(result['predictions'][0])
    result = prediction.tolist()
    ad_classes = ["Poor", "Okay", "Good"]
    prediction_class = ad_classes[result.index(max(result))]
    return jsonify({
        "categories": {
            "Poor": result[0],
            "Okay": result[1],
            "Good": result[2]
        },
        "prediction": prediction_class
    })


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
