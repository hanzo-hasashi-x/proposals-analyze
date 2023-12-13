import pandas as pd
import numpy as np

import tensorflow as tf

import json

def parse_project_bids(project):
    base = [project['id'], project['title'], project['description'], project['budget']['minimum'],
            project['budget']['maximum'], project['submitdate']]
    
    project_bids = []
        
    for bid in project['bids']:
        row = [*base]
        
        reputation = bid['reputation']['entire_history']['category_ratings']
        
        row += [bid['score'], bid['highlighted'], bid['sealed'], bid['description'], bid['submitdate'], 
                bid['amount'], bid['period'], reputation['quality'], reputation['communication'], 
                reputation['professionalism'], bid['award_status'] == 'awarded']
        
        project_bids.append(row)
    
    return project_bids


data_columns = ['project_id', 'project_title', 'project_description', 'project_min_amount',
            'project_max_amount', 'project_time_submitted', 'bid_score', 'bid_highlighted', 'bid_sealed',
            'bid_description', 'bid_time_submitted', 'bid_amount', 'bid_period', 'bid_quality', 
            'bid_communication', 'bid_professionalism', 'bid_award_status']
def load_bids_df(dataset_name):
    with open(f'data/{dataset_name}.json', 'r') as f:
        projects = json.load(f)

    data = []
    for project in projects:
        project_bids = parse_project_bids(project)
            
        if [bid for bid in project_bids if bid[-1]]: # checking if there is awarded bid, if not skip
            data+=project_bids

    return pd.DataFrame(columns=data_columns, data=data)

def num_transform(df):
    df['project/bid_amount_proportion'] = df['bid_amount'] / df['project_min_amount']
    df['bid_delay'] = (df['bid_time_submitted'] - df['project_time_submitted']) / (3600*24*7) 
    
    df = df.drop(['bid_amount', 'project_min_amount', 'project_max_amount', 
                  'project_time_submitted', 'bid_time_submitted'], axis=1)
    return df

def get_proportion(df, column_name):
    proportions = []

    projects_titles_words = df[column_name].apply(lambda x: x.split()).apply(set)
    bid_descrs_words = df['bid_description'].apply(lambda x: x.split()).apply(set)

    for project_title_set, bid_descr_set in zip(projects_titles_words, bid_descrs_words):
        words_checked = [word in bid_descr_set for word in project_title_set]
        proportions.append(sum(words_checked) / len(words_checked))

    return proportions

greetings = ['good afternoon', 'good morning', 'how are you', 'how do you do', 'dear sir',
             'good night', 'hi', 'hello', 'pleased to meet you']

def transform_text_columns(df):
    
    df['bid_description'] = df['bid_description'].fillna('')

    df['same_words_title/bid_proportion'] = get_proportion(df, 'project_title')
    df['same_words_descr/bid_proportion'] = get_proportion(df, 'project_description')

    df['bid_descr_len'] = df['bid_description'].apply(lambda x: len(x.split()))
    
    proj_descr_len = df['project_description'].apply(lambda x: len(x.split()))
    df['proj/bid_lens_proportion'] = df['bid_descr_len'] / proj_descr_len

    df['bid_greetings'] = df['bid_description'].apply(lambda x: x.lower()).apply(
        lambda x: any([x.find(greeting) != -1  for greeting in greetings]))
    
    df = df.drop(['project_title', 'project_description', 'bid_description'], axis=1)
    
    return df

def prepare_df(df):

    df = df.copy()

    df = num_transform(df)
    df = transform_text_columns(df)

    df = df.drop(['bid_highlighted'], axis=1)
    df[df['bid_score'] > 50.] = np.mean(df['bid_score'])

    return df

with open('models/vocabulary.json', 'r') as f:
    truncated_vocabulary = json.load(f)

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

def preprocess_batch(X_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)

    return X_batch.to_tensor(default_value=b"<pad>")

def encode_words(X_batch):
    X_1 = table.lookup(preprocess_batch(X_batch[0]))
    X_2 = table.lookup(preprocess_batch(X_batch[1]))

    return (X_1,X_2, *X_batch[2:])