import json
from dotmap import DotMap

def read_config(path_to_json_conf):
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
    
    #config = DotMap(config_dict)
    
    return config_dict