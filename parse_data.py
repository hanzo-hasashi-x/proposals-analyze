from datetime import datetime
import requests
import json

with open('config.json', 'r') as f:
    config = json.load(f)

    access_token = config['access_token']
    jobs = config['jobs']

url = "https://www.freelancer.com/api/projects/0.1/projects/all"

headers = {'freelancer-oauth-v1': access_token}

offset = 0

projects = []
for i in range(0,1500,100):

    from_time, to_time = datetime(2020, 1, 1).timestamp(), datetime(2023,1,1).timestamp()
    data = {"jobs[]": jobs, "project_types[]": ['fixed'],# "from_time": int(from_time), "to_time": int(to_time),
            "bid_award_statuses[]": ["awarded"], "project_statuses[]": ["closed"], "limit": 100, "bid_complete_statuses[]": ["complete"],
            "full_description": True, "offset": offset}#, "job_details": True}
    
    r = requests.request("GET", url, headers=headers, params=data)
    print(r.url)
    
    data =  r.json()['result']
    
    print(len(data['projects']))
    
    bids_url_template = 'https://www.freelancer.com/api/projects/0.1/projects/{}/bids/'
    for project in data['projects']:
    
        if project['bid_stats']['bid_count'] > 3:
            bids_url = bids_url_template.format(project['id'])
        
            r = requests.request("GET", bids_url, headers=headers, params={"limit": 20,
                                                                           "user_responsiveness": True, "user_recommendations": True, 
                                                                           "user_reputation": True})
            project['bids'] = r.json()['result']['bids']
    
            projects.append(project)

    offset += 100

with open('data/machine_learning.json', 'w') as f:
    json.dump(projects, f)

print(r.status_code)
