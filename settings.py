from yacs.config import CfgNode as CN
import os 
import json


CUR_DIR = os.path.dirname(__file__)
SETTINGS_JSON_PATH = os.path.join(CUR_DIR, 'SETTINGS.json')
with open(SETTINGS_JSON_PATH, 'r') as f:
    settings = json.load(f)

# copy cfgs from SETTINGS.json
SETTINGS = CN()
SETTINGS.__JSON_PATH__ = SETTINGS_JSON_PATH
for k, v in settings.items():
    SETTINGS[k] = v

SETTINGS.freeze()

print('Using global configuration (SETTINGS.json):')
print('-' * 80)
print(SETTINGS)
print('-' * 80)
print('\n\n\n')