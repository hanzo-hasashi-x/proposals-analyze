import pandas as pd
import numpy as np

import json

def parse_user(user_info, role='freelancer'):
    location = user_info['location']
    location_data = [location['country']['name'], location['city']]

    status = user_info['status']
    verification = [status['deposit_made'], status['payment_verified'], 
                    status['identity_verified'], status['phone_verified'], status['email_verified']]

    reputation = None
    if role == 'employer':
        if user_info['employer_reputation']:
            reputation = user_info['employer_reputation']['entire_history']
    
            reputation = [reputation['overall'], reputation['reviews'], reputation['complete']]

        else: reputation = [0,0,0]

    if role == 'freelancer':
        reputation = user_info['reputation']['entire_history'] # if freelancer
        categories = reputation['category_ratings']

        reputation = [reputation['reviews'], reputation['complete'], categories['quality'], 
                 categories['communication'], categories['professionalism'], categories['hire_again'], categories['expertise']]

    user_data = [*location_data, *verification, *reputation]

    if role == 'freelancer':
        user_data.append(user_info['profile_description'])
        jobs_ids = [job['id'] for job in user_info['jobs']] if user_info['jobs'] else []
        user_data.append(jobs_ids)

    return user_data

def parse_project_bids(project,users):

    owner_id = str(project['owner_id'])
    employer_data = parse_user( users[owner_id], role='employer' ) 

    skills = [job['id'] for job in project['jobs']]

    base = [project['id'], project['title'], project['description'], project['budget']['minimum'],
            project['budget']['maximum'], project['submitdate'], *employer_data, skills]
    
    project_bids = []
        
    for bid in project['bids']:
        row = [*base]
        
        user = users[ str(bid['bidder_id']) ]
        user_data = parse_user(user)
        
        row += [bid['score'], bid['highlighted'], bid['sealed'], bid['description'], bid['submitdate'], 
                bid['amount'], bid['period'], *user_data, bid['award_status'] == 'awarded']
        
        project_bids.append(row)
    
    return project_bids

def load_projects(dataset_name):
    with open(f'data/{dataset_name}_bids_data.json', 'r') as f:
        projects = json.load(f)

    return projects

def load_users(dataset_name):
    with open(f'data/{dataset_name}_users_data.json', 'r') as f:
        projects = json.load(f)

    return projects

project_cols = ['proj_id','title','description','min_budg','max_budg','submitdate', 'country', 'city', 'deposit_made', 
                'payment_verif','identity_verif', 'phone_verif','email_verif', 'overall','reviews_num', 'complete_num','skills']
bid_cols = ['bid_score', 'bid_highlighted', 'bid_sealed', 'bid_text', 'bid_sumbitdate','bid_amount', 'bid_period',  
            'user_country','user_city','user_deposit_made', 'user_payment_verif', 'user_identity_verif', 'user_phone_verif',
            'user_email_verif', 'user_reviews_num', 'user_complete_num', 'user_quality', 'user_communication', 
            'user_professionalism','user_hire_again', 'user_expertise', 'user_profile', 'user_skills', 'awarded']

cols = project_cols+bid_cols
def load_bids_df(projects,users):

    data = []
    for project in projects:
        project_bids = parse_project_bids(project,users)
            
        if [bid for bid in project_bids if bid[-1]]: # checking if there is awarded bid, if not skip
            data+=project_bids

    return pd.DataFrame(columns=cols, data=data)

def num_transform(df):
    df['proj_min/bid_amount_prop'] = df['bid_amount'] / df['min_budg']
    df['proj_max/bid_amount_prop'] = df['bid_amount'] / df['max_budg']

    df['bid_delay'] = (df['bid_sumbitdate'] - df['submitdate']) / (3600*24*7) 

    skills_covered_prop = []
    for skills,user_skills in df[['skills', 'user_skills']].values:
        covered_skills = [skill in user_skills for skill in skills]
        skills_covered_prop.append( sum(covered_skills) / len(covered_skills) )

    df['skills_prop'] = skills_covered_prop

    df['reviews_num'] = df['reviews_num'].fillna(0.)
    df['complete_num'] = df['complete_num'].fillna(0.)
    
    return df

def get_proportion(df, column_name):
    proportions = []

    projects_titles_words = df[column_name].apply(lambda x: x.split()).apply(set)
    bid_descrs_words = df['bid_text'].apply(lambda x: x.split()).apply(set)

    for title_set, bid_descr_set in zip(projects_titles_words, bid_descrs_words):
        words_checked = [word in bid_descr_set for word in title_set]
        proportions.append(sum(words_checked) / len(words_checked))

    return proportions

greetings = ['good afternoon', 'good morning', 'how are you', 'how do you do', 'dear sir',
             'good night', 'hi', 'hello', 'pleased to meet you']

def text_transform(df):
    
    df['bid_text'] = df['bid_text'].fillna('')

    df['same_words_title/bid_proportion'] = get_proportion(df, 'title')
    df['same_words_descr/bid_proportion'] = get_proportion(df, 'description')

    df['bid_descr_len'] = df['bid_text'].apply(lambda x: len(x.split()))
    
    proj_descr_len = df['description'].apply(lambda x: len(x.split()))
    df['proj/bid_lens_proportion'] = df['bid_descr_len'] / proj_descr_len

    df['bid_greetings'] = df['bid_text'].apply(lambda x: x.lower()).apply(
        lambda x: any([x.find(greeting) != -1  for greeting in greetings]))
    
    return df

def cat_transform(df):
    df['country_corespond'] = df['country'] == df['user_country']

    cat_cols = ['payment_verif', 'identity_verif', 'phone_verif', 'email_verif',
                'user_payment_verif', 'user_identity_verif', 'user_phone_verif',
                'user_email_verif', 'deposit_made', 'bid_highlighted', 'bid_sealed',
                'country_corespond']

    df[cat_cols] = df[cat_cols].fillna(False).astype(bool)

    return df

def prepare_df(df):

    df = df.copy()

    df = num_transform(df)
    df = cat_transform(df)
    df = text_transform(df)

    df[df['bid_score'] > 50.] = np.mean(df['bid_score'].loc[ df['bid_score'] < 50 ])

    df = df.drop(['bid_amount', 'country', 'city', 'user_country', 'user_city', 
                  'submitdate', 'bid_sumbitdate', 'user_skills', 'skills', 'user_deposit_made', 
                  'title', 'description', 'bid_text', 'user_profile'], axis=1)
    return df
