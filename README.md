# proposals-analyze

Thie project is designed to
* Create model to estimate the bid probability to be awarded
* Analyze which properties and how much influence on the bid to be awarded
* Generate bids (texts) which will have the most chance to be awarded

Also all files with level, mean the level of techniques used there, than higher than more advanced technique used to classify bids.

## Install
```
git clone https://github.com/hanzo-hasashi-x/proposals-analyze.git
cd proposals-analyze
pip install -r requirements.txt
```

## Rewiew

### Downloading bids for training (from freelancer.com)
* Generate the personal access token on this [page](https://accounts.freelancer.com/settings/develop)
* Set up config.json
	* Jobs are the ids of the project skills, which can be found on the [search page](https://www.freelancer.com/search/), by entering the necessary skills and then taking ids from url
	* projects_num - the number of project to parse (recomended 2000)
	* dataset_name - the name of the dataset file
* Add access token
	* Either by opening parse_data.py and setting it at variable access_token
	* Or by creating `.env`, with variable 
	`FREELANCER_ACCESS_TOKEN={your_access_token}`
* Run parsing with 
```
cd parser
python parse_data.py
``` 
which can take 10 minutes for 1000 projects

### Training algorithms (level_1.ipynb)
The trained algorithms are LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier, GaussianNB. More deeply trained will be SVC (with RandomSearch) and RandomForest (with GridSearch). The final estimator will the soft voting classifier of all trained models.

#### Training:

* Adjust the dataset_name variable, only if the dataset_name isn't that mentioned in config.json
* Change the name of estimator_name, estimator will be saved with the appropriate name in folder `estimators`
* Run all cells

### Training deep networks (level_2.ipynb)
There are two models which are different one from other by the embeddings and the consequential adjusting. To reach the best analyze of text data, the classification models are trained firstly only on text data, and then added into model which analyze other data.

* Pretrained embeddings: DNN model which analyze project and bid descriptions by using pretrained embeddings of 128 dimensions in output
* Custom embeddings: RNN model, which trained custom embeddings, with 2 GRU layers

#### Training:

* Adjust the variables such as dataset_name, models names and pipeline name
* Models will be saved with the appropriate names in folder `models`
* Run all cells

### Description of level_3.ipynb
* 1st model: Pretrained BERTs models are used for creating embeddings for project and bid descriptions with simple DNN for other data
* 2nd model: Custom transformer trained on `project_description` as decoder inputs and `bid_description` as input for encoder + simple DNN for other data

### Generating bids (bids_generation.ipynb)
For generation of bids ChatGPT was used with sending different prompts to reach the max probability. The models from previous trainings used as the tool to predict how much successful the bid will be. For each try it generates 3 tries, in each try ChatGPT looks at description and generates price, time of delivery and bid description, all other parameters are taken from the best bid.

To generate by yourself you will need to have api token of OpenAI API, to receive the working api token you will need
* Either to create a new account https://platform.openai.com/
* Or if you have already possibly to pay at min $5

After receiving api token
* Either by opening bids_generation.ipynb and setting it at variable api_token
* Or by creating `.env`, with variable 
`bids_generation.ipynb={your_api_token}`

### Using pretrained models
To use the model with pretained embeddings, it will be necessary to download it
```
python download_models.py
```

The passed data needs to have data columns, that's why it needs to be of DataFrame type

#### Using vote_classifier
```
import joblib
import utils

estimator = joblib.load('estimators/ml_bids_estimator.json')

data = utils.prepare_df(data) # pass data with necessary columns
pred = estimator.predict_proba(data)
```

#### Using neural networks

Load preprocess pipeline
```
import joblib
import tensorflow as tf
import tensorflow_hub as hub

preprocess = joblib.load('models/pipeline.json')
```

Using pretrained-vector embeddings
```
model = tf.keras.models.load_model('models/ml_projects_vector_embeddings_model.h5',
                                   custom_objects={"KerasLayer": hub.KerasLayer})

x_2 = preprocess.transform(data).astype('float32')
data = ( project_descriptions, bid_descriptions, x_2)

pred = model.predict(data)
```

Using custom embeddings
```
model = tf.keras.models.load_model('models/ml_projects_custom_embeddings_model.h5',
                                   custom_objects={"KerasLayer": hub.KerasLayer})

x_2 = preprocess.transform(data).astype('float32')
data = ( project_descriptions, bid_descriptions, x_2)

data = utils.encode_words(data)
pred = model.predict(data)
```

### API usage
```
import requests
import json

project = ... # requested from freelancer API and bids added to it

url = 'http://113.30.188.137/custom_embeddings_model' # or 'simple_model'
r = requests.post(url, data=json.dumps({"project": projects}),
				  headers={"Content-Type": "application/json"})

data = r.json()['pred']
```
