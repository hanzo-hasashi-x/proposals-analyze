import requests

from dotenv import load_dotenv
import tqdm
import json
import os

load_dotenv()

access_token = '' or os.getenv('FREELANCER_ACCESS_TOKEN') 
with open('config.json', 'r') as f:
    config = json.load(f)

    jobs = config['jobs']

headers = {'freelancer-oauth-v1': access_token}

with open(f'../data/{config["dataset_name"]}_bids_data.json', 'r') as f:
    projects = json.load(f)

proj_owners_ids, bids_owners_ids = [],[]
for project in projects:
    proj_owners_ids.append(project['owner_id'])

    for bid in project['bids']:
        bids_owners_ids.append(bid['bidder_id'])

ids = set(proj_owners_ids + bids_owners_ids)
print('Unique ids len:', len(ids))

url = "https://www.freelancer.com/api/users/0.1/users"
profiles_url = "https://www.freelancer.com/api/users/0.1/users"

params = {"profile_description": True, "jobs": True, "membership_details": True, "portfolio_details": True,
          "reputation": True, "employer_reputation": True}

users_data = {}
for i in tqdm.tqdm(range(0,len(ids),100)):
    params["users[]"] = list(ids)[i:i+100]
    r = requests.get(url, headers=headers, params=params)

    try:
        users_data.update(r.json()['result']['users'])
    except:
        print('API error')
        break

with open(f'../data/{config["dataset_name"]}_users_data.json', 'w') as f:
    json.dump(users_data, f)
