"""
Config processing module. Inspired from https://github.com/fizyr/tf-retinanet/blob/master/tf_retinanet/utils/config.py
"""
import yaml


def parse_yaml(path):
    """ Parse a YAML config file to a dictionary.
    """
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise (exc)
