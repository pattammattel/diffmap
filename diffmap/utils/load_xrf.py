import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import h5py
import pandas as pd
import datetime
import warnings
import pickle
import os
import sys

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions


def _load_scan(scan_id, fill_events=False):
    '''Load scan from databroker by scan id'''

    #if scan_id > 0 and scan_id in data_cache:
    #    df = data_cache[scan_id]
    #else:
    #    hdr = db[scan_id]
    #    scan_id = hdr['start']['scan_id']
    #    if scan_id not in data_cache:
    #        data_cache[scan_id] = db.get_table(hdr, fill=fill_events)
    #    df = data_cache[scan_id]
    hdr = db[scan_id]
    scan_id = hdr['start']['scan_id']
    df = db.get_table(hdr,fill=fill_events)

    return scan_id, df

def get_flyscan_dimensions(hdr):
    
    if 'dimensions' in hdr.start:
        return hdr.start['dimensions']
    else:
        return hdr.start['shape']

def get_all_scalar_data(hdr):

    keys = list(hdr.table().keys())
    scalar_keys = [k for k in keys if k.startswith('sclr1') ]
    print(f"{scalar_keys = }")
    scan_dim = get_flyscan_dimensions(hdr)
    scalar_stack_list = []

    for sclr in sorted(scalar_keys):
        
        scalar = np.array(list(hdr.data(sclr))).squeeze()
        sclr_img = scalar.reshape(scan_dim)
        scalar_stack_list.append(sclr_img)

    # Stack all the 2D images along a new axis (axis=0).
    scalar_stack = np.stack(scalar_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)

    return  scalar_stack, sorted(scalar_keys),

def get_all_xrf_roi_data(hdr):


    channels = [1, 2, 3]
    keys = list(hdr.table().keys())
    roi_keys = [k for k in keys if k.startswith('Det')]
    det1_keys = [k for k in keys if k.startswith('Det1')]
    elem_list = [k.replace("Det1_", "") for k in det1_keys]

    print(f"{elem_list = }")

    scan_dim = get_flyscan_dimensions(hdr)
    xrf_stack_list = []

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(h.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        xrf_img = spectrum.reshape(scan_dim)
        xrf_stack_list.append(xrf_img)

    # Stack all the 2D images along a new axis (axis=0).
    xrf_stack = np.stack(xrf_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)
    return xrf_stack, sorted(elem_list)