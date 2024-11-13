import os
import shutil


cambridge_scenes = {
    'GreatCourt': 'https://www.repository.cam.ac.uk/bitstreams/bceda7ac-e51c-40ca-b9d7-d758d40cdcf4/download',
    'OldHospital': 'https://www.repository.cam.ac.uk/bitstreams/ae577bfb-bdce-488c-8ce6-3765eabe420e/download',
    'KingsCollege': 'https://www.repository.cam.ac.uk/bitstreams/1cd2b04b-ada9-4841-8023-8207f1f3519b/download',
    #'StMarysChurch': 'https://www.repository.cam.ac.uk/bitstreams/2559ba20-c4d1-4295-b77f-183f580dbc56/download',
    #'ShopFacade': 'https://www.repository.cam.ac.uk/bitstreams/4e5c67dd-9497-4a1d-add4-fd0e00bcb8cb/download',
    #'Street': 'https://www.repository.cam.ac.uk/bitstreams/d39a7732-591f-470a-9f5e-2a9892118abd/download'
}

_7scenes = {
    'Chess': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip',
    'Fire': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip',
    'Heads': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip',
    'Office': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip',
    'Pumpkin': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip',
    'Kitchen': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip',
    'Stairs': 'http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip'
}

C_DATA_DIR = '/data1/heungchan/CambridgeLandmarks/'
_7_DATA_DIR = '/data1/heungchan/7Scenes/'

wget_cmd = 'wget {url} -O {out_name}'

for cs, cs_url in cambridge_scenes.items():
    print(f'Downloading dataset for {cs}...')
    # urllib.request.urlretrieve(cs_url, f'{cs}.zip')
    os.system(wget_cmd.format(url=cs_url, out_name=f'{cs}.zip'))
    unzip_cmd = f'unzip {cs}.zip'
    os.system(unzip_cmd)
    shutil.move(cs, C_DATA_DIR)
    os.remove(f'{cs}.zip') # remove zip file
    
for scene, scene_url in _7scenes.items():
    print(f'Downloading dataset for {scene}...')
    # urllib.request.urlretrieve(scene_url, f'{scene}.zip')
    os.system(wget_cmd.format(url=scene_url, out_name=f'{scene}.zip'))
    unzip_cmd = f'unzip {scene}.zip'
    os.system(unzip_cmd)
    shutil.move(scene, _7_DATA_DIR)
    os.remove(f'{scene}.zip') # remove zip file