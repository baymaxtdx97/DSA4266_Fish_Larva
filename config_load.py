import yaml
YML_STR_LOAD = "../config_yml"

def read_yaml_file():
    ''' Load yaml file

    '''
    yml_file = open(YML_STR_LOAD)
    config = yaml.load(yml_file, Loader = yaml.FullLoader)
    return config

def save_yaml_file(config):
    '''Save yaml file
    '''
    with open(YML_STR_LOAD, 'w') as yam:
        yaml.dump(config, yam)
    return config