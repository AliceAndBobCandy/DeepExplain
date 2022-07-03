'''config loader'''
import json

class Struct:
    """Helper class to parse dict to object"""
    def __init__(self, entries):
        self.__dict__.update(entries)

class ConfigLoader:
    def __init__(self, config_path):
        self.args = self.parse_config(config_path)

    def parse_config(self, config_path):
        try:
            with open(config_path) as config_parser:
                config = json.loads(json.dumps(json.load(config_parser)), object_hook=Struct) # json.load读取文件，然后使用json.dumps转成字符串，再使用json.loads读取字符串并存入Struct结构中
            return config
        except FileNotFoundError:
            print("No config found")
            return None