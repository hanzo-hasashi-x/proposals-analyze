import requests 
import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--model_name', default='pretrained')

args = parser.parse_args()

name,url = '',''
if args.model_name == 'pretrained':
    url = 'https://drive.google.com/file/d/1-YyIbzKflnsCsLBTVPCAlm5pkYN6g6f_/view?usp=sharing'
    name = 'ml_projects_vector_embeddings_model.h5'

output = 'models/'+name
gdown.download(url, output, quiet=False,fuzzy=True)
