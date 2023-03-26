'''
Demo Notebook
This quick notebook will show how to
- load config
- create the dataset
- make a model
- load pretrained weights
This notebook should be run directly from the scripts/ directory.
'''

# Config
from hydra import core, initialize, compose
from omegaconf import OmegaConf


# CHANGE ME
DATASET_DIR = '/home/goroyeh/nuScene_dataset/media/datasets/nuscenes'
LABELS_DIR = '/home/goroyeh/nuScene_dataset/media/datasets/cvt_labels_nuscenes'


core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks

initialize(config_path='../config')

# Add additional command line overrides
cfg = compose(
    config_name='config',
    overrides=[
        'experiment.save_dir=../logs/',                 # required for Hydra in notebooks
        # '+experiment=cvt_nuscenes_vehicle',
        '+experiment= cvt_nuscenes_vehicle', #gkt_nuscenes_vehicle_kernel_7x1_goro',
        f'data.dataset_dir={DATASET_DIR}',
        f'data.labels_dir={LABELS_DIR}',
        'data.version=v1.0-trainval',
        'loader.batch_size=1',
    ]
)

# resolve config references
OmegaConf.resolve(cfg)

print(list(cfg.keys()))

# Data Setup
import torch
import numpy as np
from cross_view_transformer.common import setup_experiment, load_backbone

# Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
SPLIT = 'val_qualitative_000'
SUBSAMPLE = 5

model, data, viz = setup_experiment(cfg)
print(f'done setup_experiment')
dataset = data.get_split(SPLIT, loader=False)
dataset = torch.utils.data.ConcatDataset(dataset)
dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
print(len(dataset)) # 49


# Network Setup
from pathlib import Path

# Download a pretrained model (13 Mb)
# TODO: Change this model from cvt to gkt
download_model = False # True if we want to download a pretrained model from the web
MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
# CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/logs/cvt_nuscenes_vehicles_50k.ckpt'
# MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/outputs/uuid_test/checkpoints/model_test.ckpt'
# wget $MODEL_URL -O $CHECKPOINT_PATH

import aiofiles
import aiohttp
import asyncio

async def async_http_download(src_url, dest_file, chunk_size=65536):
    async with aiofiles.open(dest_file, 'wb') as fd:
        async with aiohttp.ClientSession() as session:
            async with session.get(src_url) as resp:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    await fd.write(chunk)

if download_model:
    SRC_URL = MODEL_URL
    DEST_FILE = CHECKPOINT_PATH
    asyncio.run(async_http_download(SRC_URL, DEST_FILE))

print(f'CWD: {Path.cwd()}') # /home/goroyeh/GKT/segmentation
print(f'CHECKPOINT_PATH.exists()? {Path(CHECKPOINT_PATH).exists()}')

if Path(CHECKPOINT_PATH).exists():
    network = load_backbone(CHECKPOINT_PATH)
else:
    network = model.backbone

    print(f'{CHECKPOINT_PATH} not found. Using randomly initialized weights.')

# Run the model
import time
import imageio
import ipywidgets as widgets

GIF_PATH = './predictions.gif'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)
network.eval()

images = list()
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = network(batch)
        visualization = np.vstack(viz(batch=batch, pred=pred))
        images.append(visualization)

# import module
import codecs
f = open('gkt_display.html', 'w')


# Save a gif
duration = [1 for _ in images[:-1]] + [5 for _ in images[-1:]]
imageio.mimsave(GIF_PATH, images, duration=duration)
print(f'Succefully saved predictions.gif')
html = f'''
<div align="center">
<img src="{GIF_PATH}?modified={time.time()}" width="80%">
</div>
'''

# writing the code into the file
f.write(html)
# close the file
f.close()
print(f'Succefully saved gkt_display.html')
# viewing html files
# below code creates a 
# codecs.StreamReaderWriter object
file = codecs.open("gkt_display.html", 'r', "utf-8")
# using .read method to view the html 
# code from our object
# print(file.read())

## Viewing the html on web
# import module
# import webbrowser
# open html file
# webbrowser.open('gkt_display.html') 

# On ipynb iPython Notebook
# display(widgets.HTML(html))