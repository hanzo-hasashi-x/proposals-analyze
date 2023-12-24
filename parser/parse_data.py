from datetime import datetime
import requests

from dotenv import load_dotenv
import json
import tqdm
import os

load_dotenv()

BIDS_MIN_NUM = 3

access_token = '' or os.getenv('FREELANCER_ACCESS_TOKEN') 
with open('config.json', 'r') as f:
    config = json.load(f)

    jobs = config['jobs']

url = "https://www.freelancer.com/api/projects/0.1/projects/all"

headers = {'freelancer-oauth-v1': access_token}

offset = 0

projects = []
for i in tqdm.tqdm(range(0,config["projects_num"],100)):

    from_time, to_time = datetime(2020, 1, 1).timestamp(), datetime(2023,1,1).timestamp()
    data = {"jobs[]": jobs, "project_types[]": ['fixed'], "bid_award_statuses[]": ["awarded"],
            "project_statuses[]": ["closed"], "limit": 100, "bid_complete_statuses[]": ["complete"],
            "full_description": True, "offset": offset, "languages[]": ["en"], "job_details": True}
    
    r = requests.request("GET", url, headers=headers, params=data)
    
    try:
        data = r.json()['result']
    except:
        print(f'Parsing stopped on {offset} project')
        break
    
    bids_url_template = 'https://www.freelancer.com/api/projects/0.1/projects/{}/bids/'
    for project in data['projects']:
    
        if project['bid_stats']['bid_count'] > BIDS_MIN_NUM:
            bids_url = bids_url_template.format(project['id'])
        
            r = requests.request("GET", bids_url, headers=headers, params={"reputation": True})

            try:
                project['bids'] = r.json()['result']['bids']
            except: break
    
            projects.append(project)

    offset += 100

with open(f'../data/{config["dataset_name"]}_bids_data.json', 'w') as f:
    json.dump(projects, f)

print(r.status_code)
