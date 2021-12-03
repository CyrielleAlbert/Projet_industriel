from make_fragments import run as make_frag 
from register_fragments import run as register_frag
from refine_registration import run as refine_frag
from integrate_scene import run as integrate_scene
import json 

config_path = './../../config/dataset.json'

with open(config_path,'r') as file:
    config = json.load(file)
    
make_frag(config)
register_frag(config)
refine_frag(config)
integrate_scene(config)