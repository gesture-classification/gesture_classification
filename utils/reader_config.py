import json

def read_config(path_to_json_conf):
    with open(path_to_json_conf, 'r') as config_file:
        config = json.load(config_file)
    return config