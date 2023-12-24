from flask import Flask, request

import json
import utils
import joblib
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

estimator = joblib.load('estimators/ml_bids_estimator.json')

@app.route('/simple_model', methods=['POST'])
def simpel_model_predict():
    project = request.get_json()['project']
    data = utils.parse_project_bids(project)
    
    df = pd.DataFrame(columns=utils.data_columns, data=data)
    data = utils.prepare_df(df) # pass data with necessary columns

    response = estimator.predict_proba(data)[:,1].tolist()
    return json.dumps( {"pred": response} )


tf_pipeline = joblib.load('models/pipeline.json')
model = tf.keras.models.load_model('models/ml_projects_custom_embeddings_model.h5',
                                   custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/custom_embeddings_model', methods=['POST'])
def custom_embed_predict():
    project = request.get_json()['project']
    data = utils.parse_project_bids(project)
    
    df = pd.DataFrame(columns=utils.data_columns, data=data)
    data = utils.prepare_df(df)

    x_2 = tf_pipeline.transform(data).astype('float32')
    data = (df['project_description'].values, df['bid_description'].values,
            x_2)

    response = model.predict(utils.encode_words(data), verbose=0).tolist()
    return json.dumps( {"pred": response} )
