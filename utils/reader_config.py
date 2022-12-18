import json
from dotmap import DotMap

def read_config(path_to_json_conf:str)->dict:
    """Функция загрузки конфига в память

    Args:
        path_to_json_conf (_str_): путь к файлу конфигурации

    Returns:
        config (dict): словарь с параметрами кинфигурации
    """    
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = DotMap(config_dict)
    
    return config