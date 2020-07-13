# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
import constants

from flask import Flask, request, jsonify, make_response
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import json
import requests
from keras.preprocessing import text, sequence
import nltk


# nltk.download("stopwords")
# stopWords = stopwords.words('russian')
sw_file = open("russian")
file_contents = sw_file.read()
stopWords = file_contents.splitlines()
MODEL_URI='http://35.199.176.26:8501/v1/models/my_saved_model:predict'

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


def clean_text(text):
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join([re.sub(r'[^\w\s]', ' ', word) for word in text.split()])
    text = ' '.join([word.lower().strip() for word in text.split() if
                            word.lower().strip() not in (stopWords) and len(word) >= 3])
    return text


def tokenize_text(texts, token, len=50):
    texts = token.texts_to_sequences(texts)
    texts = sequence.pad_sequences(texts, maxlen=len)
    return texts


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello User!'


@app.route('/api/predict', methods=['POST'])
def calc_class():
    content = request.data
    req = request.json
    errors = []
    with open('tokenizer_desc.pickle', 'rb') as handle:
        tokenizer_desc = pickle.load(handle)
    with open('tokenizer_title.pickle', 'rb') as handle:
        tokenizer_title = pickle.load(handle)
    sc = pickle.load(open('scaler.pickle', 'rb'))
    x = np.zeros(82)
    X_test_desc = req.get("description")
    region_en = "region_en_" + req.get("region_en")
    category_name_en = "category_name_en_" + req.get("category_name_en")
    user_type = "user_type_" + req.get("user_type")
    x[constants.columns.index(region_en)] = 1
    x[constants.columns.index(category_name_en)] = 1
    x[constants.columns.index(user_type)] = 1
    print(len(req.get("description")))
    X_test_desc = clean_text(X_test_desc)
    X_test_title = clean_text(req.get("title"))
    X_test_desc = tokenize_text([X_test_desc], tokenizer_desc, 50)
    X_test_title = tokenize_text([X_test_title], tokenizer_title, 20)
    x_weekend = 1 if(req.get("weekday") in ["Saturday", "Sunday"]) else 0
    x_price = req.get("price")
    x_description_len = len(req.get("description").split())
    x_title_len = len(req.get("title").split())
    x_param_1_len = len(req.get("param_1").split())
    x_param_2_len = len(req.get("param_2").split())
    x_param_3_len = len(req.get("param_3").split())
    df = {'price': [x_price],
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
    X_test = np.asarray(x)
    data = json.dumps({
        'instances': [
            {
                "nlp_input_desc": X_test_desc[0].tolist(),
                "nlp_input_title": X_test_title[0].tolist(),
                "meta_input": X_test.tolist()
            }
        ]
    })
    response = requests.post(MODEL_URI, data=data.encode('utf-8'))
    result = json.loads(response.text)
    prediction = np.squeeze(result['predictions'][0])
#    predictions = model.predict([X_test_desc, X_test_title, X_test])
    result = prediction.tolist().index(max(prediction.tolist()))
    return jsonify({"Category": result})


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
