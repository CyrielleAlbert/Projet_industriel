from make_fragments import run

import json 

config_path = './../../config/classe.json'

with open(config_path,'r') as file:
    config = json.load(file)
    
run(config)