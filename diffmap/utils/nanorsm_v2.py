############ Created by Hanfei Yan for nanodiffraction analysis at HXN ############
############ Edited by Hanfei Yan on July 06, 2023 #################################

import numpy as np
from pystackreg import StackReg
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tifffile as tf
import h5py
import pandas as pd
import datetime
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
#from matplotlib.widgets import Slider, Button
#from ipywidgets import interact, interactive, fixed, interact_manual
#import ipywidgets as widgets
#from scipy.interpolate import interpn
import csv
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import pandas as pd
import warnings
import glob
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys

sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

det_params = {'merlin1':55, "merlin2":55, "eiger2_images":75}

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
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        xrf_img = spectrum.reshape(scan_dim)
        xrf_stack_list.append(xrf_img)

    # Stack all the 2D images along a new axis (axis=0).
    xrf_stack = np.stack(xrf_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)
    return xrf_stack, sorted(elem_list)

def get_sid_list(str_list, interval):
    num_elem = np.size(str_list)
    for i in range(num_elem):
        str_elem = str_list[i].split('-')
        if i == 0:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            sid_list = np.reshape(tmp,(-1,))
        else:
            if np.size(str_elem) == 1:
                tmp = int(str_elem[0])
            else:
                tmp = np.arange(int(str_elem[0]),int(str_elem[1])+1,interval)
            tmp = np.reshape(tmp,(-1,))
            sid_list = np.concatenate((sid_list,tmp))
    return sid_list

def get_scan_details(sid = -1):
    param_dict = {"scan_id":int(sid)}
    h = db[int(sid)]
    df = db.get_table(h,stream_name = "baseline")
    start_doc = h.start
    mots = start_doc['motors']

    # Create a datetime object from the Unix time.
    datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
    formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
    param_dict["time"] = formatted_time
    param_dict["motors"] = start_doc["motors"]
    if "detectors" in start_doc.keys():
        param_dict["detectors"] = start_doc["detectors"]
        param_dict["scan_start1"] = start_doc["scan_start1"]
        param_dict["num1"] = start_doc["num1"]
        param_dict["scan_end1"] = start_doc["scan_end1"]
        
        if len(mots)==2:

            param_dict["scan_start2"] = start_doc["scan_start2"]
            param_dict["scan_end2"] = start_doc["scan_end2"]
            param_dict["num2"] = start_doc["num2"]
        param_dict["exposure_time"] = start_doc["exposure_time"]

    elif "scan" in start_doc.keys():
        param_dict["scan"] = start_doc["scan"]


    
    param_dict["zp_theta"] = np.round(df.zpsth.iloc[0],3)
    param_dict["mll_theta"] = np.round(df.dsth.iloc[0],3)
    param_dict["energy"] = np.round(df.energy.iloc[0],3)
    return param_dict

def export_scan_details(sid_list, wd):

    for sid in tqdm(sid_list):
        export_scan_metadata(sid, wd)

def get_scan_metadata(sid):
    
    output = db.get_table(db[int(sid)],stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(sid = int(sid))])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    return output



def export_scan_metadata(sid, wd):
    output = db.get_table(db[int(sid)],stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(sid = int(sid))])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    sid_ = df_dictionary['scan_id']
    save_as = os.path.join(wd,f"{sid_}_metadata.csv")
    output.to_csv(save_as,index=False)
    print(f"{save_as = }")
    

def load_ims(file_list):
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = tf.imread(file_name)
        im_row, im_col = np.shape(im)
        if i == 0:
            im_stack = np.reshape(im,(1,im_row,im_col))
        else:
            #im_stack_num = i 
            im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
            row = np.maximum(im_row,im_stack_row)
            col = np.maximum(im_col,im_stack_col)
            if im_row < im_stack_row:
                r_s = np.round((im_stack_row-im_row)/2)
            else:
                r_s = 0
            if im_col < im_stack_col:
                c_s = np.round((im_stack_col-im_col)/2)
            else:
                c_s = 0
            im_stack_tmp = np.zeros((im_stack_num+1,row,col))
            im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
            im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
            im_stack = im_stack_tmp
    return im_stack

def load_txts(file_list):
    # stacking is along the first axis
    num_ims = np.size(file_list)
    for i in tqdm(range(num_ims),desc="Progress"):
        file_name = file_list[i]
        im = np.loadtxt(file_name)
        im_row, im_col = np.shape(im)
        if i == 0:
            im_stack = np.reshape(im,(1,im_row,im_col))
        else:
            #im_stack_num = i 
            im_stack_num, im_stack_row,im_stack_col = np.shape(im_stack)
            row = np.maximum(im_row,im_stack_row)
            col = np.maximum(im_col,im_stack_col)
            if im_row < im_stack_row:
                r_s = np.round((im_stack_row-im_row)/2)
            else:
                r_s = 0
            if im_col < im_stack_col:
                c_s = np.round((im_stack_col-im_col)/2)
            else:
                c_s = 0
            im_stack_tmp = np.zeros((im_stack_num+1,row,col))
            im_stack_tmp[0:im_stack_num,0:im_stack_row,0:im_stack_col] = im_stack
            
            im_stack_tmp[im_stack_num,r_s:im_row+r_s,c_s:im_col+c_s] = im
            im_stack = im_stack_tmp
    return im_stack

def create_file_list(data_path, prefix, postfix, sid_list):
    num = np.size(sid_list)
    file_list = []
    for sid in sid_list:
        tmp = ''.join([data_path, prefix,'{}'.format(sid),postfix])
        file_list.append(tmp)
    return file_list

def align_im_stack(im_stack, norm_intensity = False,reference = "previous"):
    # default stacking axis is zero
    #im_stack = np.moveaxis(im_stack,2,0)
    if norm_intensity:
        mean = np.mean(im_stack)
        std = np.std(im_stack)
        im_stack = (im_stack - mean) / std
    sr = StackReg(StackReg.TRANSLATION)
    #sr = StackReg(StackReg.SCALED_ROTATION)
    #sr = StackReg(StackReg.RIGID_BODY)
    tmats = sr.register_stack(im_stack, reference=reference)
    out = sr.transform_stack(im_stack)
    a = tmats[:,0,2]
    b = tmats[:,1,2]
    trans_matrix = np.column_stack([-b,-a])
    return out, trans_matrix

def load_h5_data(file_list, roi, mask):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'
    
    num_scans = np.size(file_list)
    det = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        file_name = file_list[i]
        f = h5py.File(file_name,'r')       
        if mask is None:
            data = f[det]
        else:
            data = f[det]*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
        if i == 0:
            raw_size = np.shape(f[det])
            print("Total scan points: {}; raw image row: {}; raw image col: {}".format(raw_size[0],raw_size[1],raw_size[2]))
            data_size = np.shape(data)
            print("Total scan points: {}; data image row: {}; data image col: {}".format(data_size[0],data_size[1],data_size[2]))
            diff_data = np.zeros(np.append(num_scans,np.shape(data)),dtype=data_type)
        sz = diff_data.shape    
        diff_data[i] = np.resize(data,(sz[1],sz[2],sz[3])) # in case there are lost frames
    if  num_scans == 1: # assume it is a rocking curve scan
        diff_data = np.swapaxes(diff_data,0,1) # move angle to the first axis
        print("Assume it is a rocking curve scan; number of angles = {}".format(diff_data.shape[0]))
    return diff_data  


def return_diff_array(sid, det="eiger2_image", mon="sclr1_ch4", threshold=None):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    data_type = 'float32'
    data_name = '/entry/instrument/detector/data'

    #skip 1d

    hdr = db[int(sid)]
    start_doc = hdr["start"]
    if not start_doc["plan_type"] in ("FlyPlan1D",):

        file_name = get_path(sid,det)
        print(file_name)
        num_subscan = len(file_name)
        
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = np.asarray(f[data_name],dtype=data_type)
            #data = np.asarray(f[data_name])
            print(data.shape)
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
                print(data.shape)

        raw_size = np.shape(data)
        if threshold is not None:
            data[data<threshold[0]] = 0
            data[data>threshold[1]] = 0
        
        if mon is not None:
            mon_data = db[sid].table()[mon]
            ln = np.size(mon_data)
            mon_array = np.zeros(ln,dtype=data_type)
            for n in range(1,ln):
                mon_array[n] = mon_data[n] 
            avg = np.mean(mon_array[mon_array != 0])
            mon_array[mon_array == 0] = avg
                
            #misssing frame issue

            if len(mon_array) != data.shape[0]:
                if len(mon_array) > data.shape[0]:
                    last_data_point = data[-1]  # Last data point along the first dimension
                    last_data_point = last_data_point[np.newaxis, :,:]  
                    data = np.concatenate((data, last_data_point), axis=0)
                else:
                    last_mon_array_element = mon_array[-1]
                    mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            

            data = data/mon_array[:,np.newaxis,np.newaxis]
            

    return data

def export_diff_data_as_tiff(first_sid,last_sid, det="eiger2_image", mon="sclr1_ch4", roi=None, mask=None, threshold=None, wd = '.', norm_with_ic = True):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    sid_list = np.arange(first_sid,last_sid+1)
    
    data_type = 'float32'
  
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        print(f"{sid = }")

        #skip 1d

        hdr = db[int(sid)]
        start_doc = hdr["start"]
        scan_table = hdr.table()
        if not start_doc["plan_type"] in ("FlyPlan1D",):

            file_name = get_path(sid,det)
            print(file_name)
            num_subscan = len(file_name)
            
            if num_subscan == 1:
                f = h5py.File(file_name[0],'r') 
                data = np.asarray(f[data_name],dtype=data_type)
                #data = np.asarray(f[data_name])
                print(data.shape)
            else:
                sorted_files = sort_files_by_creation_time(file_name)
                ind = 0
                for name in sorted_files:
                    f = h5py.File(name,'r')
                    if ind == 0:
                        data = np.asarray(f[data_name],dtype=data_type)
                    else:   
                        data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                    ind = ind + 1
                    print(data.shape)
                #data = list(db[sid].data(det))
                #data = np.asarray(np.squeeze(data),dtype=data_type)
            raw_size = np.shape(data)
            if threshold is not None:
                data[data<threshold[0]] = 0
                data[data>threshold[1]] = 0
            if mon is not None:
                mon_data = db[sid].table()[mon]
                ln = np.size(mon_data)
                mon_array = np.zeros(ln,dtype=data_type)
                for n in range(1,ln):
                    mon_array[n] = mon_data[n] 
                avg = np.mean(mon_array[mon_array != 0])
                mon_array[mon_array == 0] = avg
                
            #misssing frame issue

            if len(mon_array) != data.shape[0]:
                if len(mon_array) > data.shape[0]:
                    last_data_point = data[-1]  # Last data point along the first dimension
                    last_data_point = last_data_point[np.newaxis, :,:]  
                    data = np.concatenate((data, last_data_point), axis=0)
                else:
                    last_mon_array_element = mon_array[-1]
                    mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            
            if norm_with_ic:
                data = data/mon_array[:,np.newaxis,np.newaxis]
            
            if mask is not None:     
                #sz = data.shape
                data = data*mask
            if roi is None:
                data = np.flip(data[:,:,:],axis = 1)
            else:
                data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
                
            print(f"data size = {data.size/1_073_741_824 :.2f} GB")
            save_folder =  os.path.join(wd,f"{sid}_diff_data")   

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                

            saved_as = os.path.join(save_folder,f"{sid}_diff_{det}.tiff")
            tf.imwrite(saved_as, data, dtype = np.float32)
            export_scan_metadata(sid,save_folder)
            scan_table.to_csv(os.path.join(save_folder,f"{sid}_scan_table.csv"))
            print(f"{saved_as =}")
            
def export_diff_data_as_h5(sid_list, 
                           det="eiger2_image", 
                           wd = '.', 
                           mon = 'sclr1_ch4',
                           compression = 'gzip'):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    
    data_type = 'float32'
    if isinstance(sid_list, (int, float)):
        sid_list = [sid_list]

  
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        print(f"{sid = }")

        #skip 1d

        hdr = db[int(sid)]
        start_doc = hdr["start"]
        sid = start_doc["scan_id"]
        
        if 'num1' and 'num2' in start_doc:
            dim1,dim2 = start_doc['num1'],start_doc['num2']
        elif 'shape' in start_doc:
            dim1,dim2 = start_doc.shape
        try:
            xy_scan_positions = list(np.array(df[mots[0]]),np.array(df[mots[1]]))
        except:
            xy_scan_positions = list(get_scan_positions(hdr))

        scan_table = get_scan_metadata(int(sid))
        if not start_doc["plan_type"] in ("FlyPlan1D",):

            file_name = get_path(sid,det)
            num_subscan = len(file_name)
            
            if num_subscan == 1:
                f = h5py.File(file_name[0],'r') 
                data = np.asarray(f[data_name],dtype=data_type)
            else:
                sorted_files = sort_files_by_creation_time(file_name)
                ind = 0
                for name in sorted_files:
                    f = h5py.File(name,'r')
                    if ind == 0:
                        data = np.asarray(f[data_name],dtype=data_type)
                    else:   
                        data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                    ind = ind + 1
                #data = list(db[sid].data(det))
                #data = np.asarray(np.squeeze(data),dtype=data_type)
            _, roi1,roi2 = np.shape(data)
            if mon != None:
                mon_array = np.array(list(hdr.data(str(mon)))).squeeze()
            data = np.flip(data[:,:,:],axis = 1)

                
            print(f"data size = {data.size/1_073_741_824 :.2f} GB")

            #save_folder =  os.path.join(wd,f"{sid}_diff_data")   

            #if not os.path.exists(save_folder):
                #os.makedirs(save_folder)

            if wd:
                save_folder = wd
                
            saved_as = os.path.join(save_folder,f"scan_{sid}_{det}")

            f.close()

            
            with h5py.File(saved_as+'.h5','w') as f:

                # Create (or get) the group for the detector raw data
                det_group = f.require_group(f"/diff_data/{det}/")
                det_group.create_dataset(
                    "raw_data",
                    data=data.reshape(dim1, dim2, roi1, roi2),
                    compression=compression
                )
                if mon != None:
                    det_group.create_dataset(
                        "Io",
                        data=mon_array.reshape(dim1, dim2)
                    )
                
                # Create (or get) the group for scan data
                scan_group = f.require_group("scan/")
                scan_group.create_dataset(
                    "scan_positions",
                    data=xy_scan_positions
                )

                scan_table.to_csv(saved_as+'_meta_data.csv')

                xrf_group = f.require_group('/xrf_roi_data/')
                xrf_stack, elem_list  = get_all_xrf_roi_data(hdr)
                scalar_stack, scalar_keys  = get_all_scalar_data(hdr)
                xrf_group.create_dataset('xrf_roi_array', data = xrf_stack)
                xrf_group.create_dataset('xrf_elem_names', data = elem_list)

                scalar_group = f.require_group('/scalar_data/')
                scalar_group.create_dataset('scalar_array', data = scalar_stack)
                scalar_group.create_dataset('scalar_array_names', data = scalar_keys)

            f.close()
            print(f"{saved_as =}")



def unpack_diff_h5(filename, det):
    """
    Unpack raw_data, Io, and scan_positions from an HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file.
        det (str): Detector name used in the group path.

    Returns:
        tuple: (raw_data, Io, scan_positions)
    """
    with h5py.File(filename, "r") as f:
        # Read data from the detector group
        det_group = f[f"/diff_data/{det}/"]
        raw_data = det_group["raw_data"][()]
        Io = det_group["Io"][()]
        
        # Read scan positions
        scan_positions = f["/diff_data/scan/scan_positions"][()]
    
    return raw_data, Io, scan_positions

def export_diff_h5_log_file(logfile, diff_detector = 'merlin1',compression = None):

    df = pd.read_csv(logfile)
    sid_list = df['scan_id'].to_numpy(dtype = 'int')
    angles = df['angle'].to_numpy()
    print(sid_list)

    dir_ = os.path.abspath(os.path.dirname(logfile))
    folder_name = os.path.basename(logfile).split('.')[0]
    save_folder =  os.path.join(dir_,folder_name+"_diff_data")
    data_path = save_folder
    os.makedirs(save_folder, exist_ok = True)

    print(f"h5 files will be saved to {save_folder}")
    
    export_diff_data_as_h5(sid_list, 
                           det=diff_detector,
                           wd = save_folder, 
                           compression = compression)
    
    print(f"All scans from {logfile} is exported to {save_folder}")



def export_single_diff_data(param_dict):
    
    '''
    load diffraction data of a single scan through databroker
    roi[row_start,col_start,row_size,col_size]
    mask has to be the same size of the image data, which corresponds to the last two axes
    
    param_dict = {wd:'.', 
                 "sid":-1, 
                 "det":"merlin1", 
                 "mon":"sclr1_ch4", 
                 "roi":None, 
                 "mask":None, 
                 "threshold":None}
    '''

    det=param_dict["det"]
    mon=param_dict["mon"]
    roi=param_dict["roi"]
    mask=param_dict["mask"]
    threshold=param_dict["threshold"]
    wd = param_dict["wd"]


    data_type = 'float32'
    data_name = '/entry/instrument/detector/data'
    sid = param_dict["sid"]
    start_doc = db[int(sid)].start
    sid = start_doc["scan_id"]
    param_dict["sid"] = sid
    file_name = get_path(sid,det)
    num_subscan = len(file_name)
    scan_table = db[sid].table()

    #print(f"Loading{sid} please wait...")
        

    hdr = db[int(sid)]
    start_doc = hdr["start"]
    sid = start_doc["scan_id"]
    
    if 'num1' and 'num2' in start_doc:
        dim1,dim2 = start_doc['num1'],start_doc['num2']
    elif 'shape' in start_doc:
        dim1,dim2 = start_doc.shape
    try:
        xy_scan_positions = list(np.array(df[mots[0]]),np.array(df[mots[1]]))
    except:
        xy_scan_positions = list(get_scan_positions(hdr))

    scan_table = get_scan_metadata(int(sid))
    if not start_doc["plan_type"] in ("FlyPlan1D",):

        file_name = get_path(sid,det)
        num_subscan = len(file_name)
        
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = np.asarray(f[data_name],dtype=data_type)
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
            #data = list(db[sid].data(det))
            #data = np.asarray(np.squeeze(data),dtype=data_type)
        _, roi1,roi2 = np.shape(data)

        if threshold is not None:
            data[data<threshold[0]] = 0
            data[data>threshold[1]] = 0

        norm_with = mon

        if norm_with is not None:
            #mon_array = np.stack(hdr.table(fill=True)[norm_with])
            mon_array = np.array(list(hdr.data(str(norm_with)))).squeeze()
            norm_data = data/mon_array[:,np.newaxis,np.newaxis]
            print(f"data normalized with {norm_with} ")

        
        if mask is not None:     
            #sz = data.shape
            data = data*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
            
        print(f"data size = {data.size/1_073_741_824 :.2f} GB")

        #save_folder =  os.path.join(wd,f"{sid}_diff_data")   

        #if not os.path.exists(save_folder):
            #os.makedirs(save_folder)

        if wd:
            save_folder = wd
            
        saved_as = os.path.join(save_folder,f"scan_{sid}_{det}")

        f.close()

    print(f"data reshaped to {data.shape}")

    save_folder =  os.path.join(wd,f"{sid}_diff_data")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_as = os.path.join(save_folder,f"{sid}_diff_{det}.tiff")
    if mon is not None:
        tf.imwrite(saved_as, data.reshape(dim1,dim2,roi1,roi2), dtype = np.float32)

    else:
        tf.imwrite(saved_as, data.reshape(dim1,dim2,roi1,roi2).astype('uint16'), imagej=True)

    export_scan_metadata(sid,save_folder)
    scan_table.to_csv(os.path.join(save_folder,f"{sid}_scan_table.csv"))
    print(f"{saved_as =}")

def export_single_diff_data_old_scan(param_dict):
    
    '''
    load diffraction data of a single scan through databroker
    roi[row_start,col_start,row_size,col_size]
    mask has to be the same size of the image data, which corresponds to the last two axes
    
    param_dict = {wd:'.', 
                 "sid":-1, 
                 "det":"merlin1", 
                 "mon":"sclr1_ch4", 
                 "roi":None, 
                 "mask":None, 
                 "threshold":None}
    '''

    det=param_dict["det"]
    mon=param_dict["mon"]
    roi=param_dict["roi"]
    mask=param_dict["mask"]
    threshold=param_dict["threshold"]
    wd = param_dict["wd"]


    data_type = 'float32'
    data_name = '/entry/instrument/detector/data'
    sid = param_dict["sid"]
    start_doc = db[int(sid)].start
    sid = start_doc["scan_id"]
    param_dict["sid"] = sid
    file_name = get_path(sid,det)
    num_subscan = len(file_name)
    scan_table = db[sid].table()

    #print(f"Loading{sid} please wait...")
        
    if num_subscan == 1:
        f = h5py.File(file_name[0],'r') 
        data = np.asarray(f[data_name],dtype=data_type)
    else:
        sorted_files = sort_files_by_creation_time(file_name)
        ind = 0
        for name in sorted_files:
            f = h5py.File(name,'r')
            if ind == 0:
                data = np.asarray(f[data_name],dtype=data_type)
            else:   
                data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
            ind = ind + 1
        #data = list(db[sid].data(det))
        #data = np.asarray(np.squeeze(data),dtype=data_type)
    raw_size = np.shape(data)
    if threshold is not None:
        data[data<threshold[0]] = 0
        data[data>threshold[1]] = 0
    if mon is not None:
        mon_data = scan_table[mon]
        ln = np.size(mon_data)
        mon_array = np.zeros(ln,dtype=data_type)
        for n in range(1,ln):
            mon_array[n] = mon_data[n] 
        avg = np.mean(mon_array[mon_array != 0])
        mon_array[mon_array == 0] = avg
        
    #misssing frame issue

        if len(mon_array) != data.shape[0]:
            if len(mon_array) > data.shape[0]:
                last_data_point = data[-1]  # Last data point along the first dimension
                last_data_point = last_data_point[np.newaxis, :,:]  
                data = np.concatenate((data, last_data_point), axis=0)
            else:
                last_mon_array_element = mon_array[-1]
                mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            
        
        data = data/mon_array[:,np.newaxis,np.newaxis]
    
    if mask is not None:     
        #sz = data.shape
        data = data*mask
    if roi is None:
        data = np.flip(data[:,:,:],axis = 1)
    else:
        data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
        
    print(f"data size = {data.size/1_073_741_824 :.2f} GB")
    data_shape = data.shape
    print(f"{data_shape = } ")
    data_shape = data.shape
    dim1 = start_doc['num1']
    dim2 = start_doc['num2']

    data = data.reshape(dim2,dim1,data_shape[-2],data_shape[-1])
    data=np.flip(data, axis =2)

    print(f"data reshaped to {data.shape}")

    save_folder =  os.path.join(wd,f"{sid}_diff_data")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_as = os.path.join(save_folder,f"{sid}_diff_{det}.tiff")
    if mon is not None:
        tf.imwrite(saved_as, data, dtype = np.float32)

    else:
        tf.imwrite(saved_as, data.astype('uint16'), imagej=True)

    export_scan_metadata(sid,save_folder)
    scan_table.to_csv(os.path.join(save_folder,f"{sid}_scan_table.csv"))
    print(f"{saved_as =}")

def load_h5_data_db(sid_list, det, mon=None, roi=None, mask=None, threshold=None, save_intermediate_data = False, wd = '.', norm_with_ic = True):
    # load diffraction data of a list of scans through databroker, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'
  
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        print(f"{sid = }")
        file_name = get_path(sid,det)
        num_subscan = len(file_name)
         
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = np.asarray(f[data_name],dtype=data_type)
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
            #data = list(db[sid].data(det))
            #data = np.asarray(np.squeeze(data),dtype=data_type)
        raw_size = np.shape(data)
        if threshold is not None:
            data[data<threshold[0]] = 0
            data[data>threshold[1]] = 0
        if mon is not None:
            mon_data = db[sid].table()[mon]
            ln = np.size(mon_data)
            mon_array = np.zeros(ln,dtype=data_type)
            for n in range(1,ln):
                mon_array[n] = mon_data[n] 
            avg = np.mean(mon_array[mon_array != 0])
            mon_array[mon_array == 0] = avg
            
        #misssing frame issue

        if len(mon_array) != data.shape[0]:
            if len(mon_array) > data.shape[0]:
                last_data_point = data[-1]  # Last data point along the first dimension
                last_data_point = last_data_point[np.newaxis, :,:]  
                data = np.concatenate((data, last_data_point), axis=0)
            else:
                last_mon_array_element = mon_array[-1]
                mon_array = np.concatenate((mon_array, last_mon_array_element), axis=0)            
        if norm_with_ic:
            data = data/mon_array[:,np.newaxis,np.newaxis]
        
        if mask is not None:     
            #sz = data.shape
            data = data*mask
        if roi is None:
            data = np.flip(data[:,:,:],axis = 1)
        else:
            data = np.flip(data[:,roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]],axis = 1)
            
        print(f"{data.size = }")
            
        if save_intermediate_data:
            saved_as = os.path.join(wd,f"{sid}_diff_processed.tiff")
            tf.imwrite(saved_as, data, dtype = np.float32)
            print(f"{saved_as =}")
        
        if i == 0:
            print(f"Total scan points: {raw_size[0]}; raw image row: {raw_size[1]}; raw image col: {raw_size[2]}")
            data_size = np.shape(data)
            print(f"Total scan points: {data_size[0]}; data image row: {data_size[1]}; data image col: {data_size[2]}")
            diff_data = np.zeros(np.append(num_scans,np.shape(data)),dtype=data_type)
            #TODO maybe save intermediate data , sometime one file fails
        sz = diff_data.shape    
        diff_data[i] = np.resize(data,(sz[1],sz[2],sz[3])) # in case there are lost frames
    if  num_scans == 1: # assume it is a rocking curve scan
        diff_data = np.swapaxes(diff_data,0,1) # move angle to the first axis
        print(f"Assume it is a rocking curve scan; number of angles = {diff_data.shape[0]}")
    return diff_data  






def load_xrf_rois(sid, wd, norm = None):

    h = db[int(sid)]
    df = h.table()
    start_doc  = h['start']
    sid = start_doc ["scan_id"]
    list_of_rois = [roi for roi in df.columns if roi.startswith("Det")]
    if x is None:
        x = start_doc['motor1']
        #x = hdr['motors'][0]
    x_data = np.asarray(df[x])

    if y is None:
        y = start_doc['motor2']
        #y = hdr['motors'][1]
    y_data = np.asarray(df[y])

    if norm is not None:
        monitor = np.asarray(df[norm], dtype=np.float32)
        monitor = np.where(monitor == 0, np.nanmin(monitor),monitor) #patch for dropping first data point
        spectrum = spectrum/(monitor)

    nx, ny = start_doc["shape"]

    pass

def diff_data_from_local(sid_list, wd):
    num_scans = len(sid_list)
    first_file = os.path.join(wd, f"{sid_list[0]}_diff_processed.tiff")
    first_data = tf.imread(first_file)
    data_shape = np.shape(first_data)
    
    diff_data = np.zeros((num_scans,) + data_shape, dtype=np.float32)
    
    for i, sid in tqdm(enumerate(sid_list), desc="Stacking"):
        filename = os.path.join(wd, f"{sid}_{det}.tiff")
        data = tf.imread(filename)
        diff_data[i] = data
        
    return diff_data

def sum_all_h5_data_db(sid_list, det):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'    
    num_scans = np.size(sid_list)
    data_name = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        sid = int(sid_list[i])
        file_name = get_path(sid,det)
        num_subscan = len(file_name)
         
        if num_subscan == 1:
            f = h5py.File(file_name[0],'r') 
            data = f[data_name]
        else:
            sorted_files = sort_files_by_creation_time(file_name)
            ind = 0
            for name in sorted_files:
                f = h5py.File(name,'r')
                if ind == 0:
                    data = np.asarray(f[data_name],dtype=data_type)
                else:   
                    data = np.concatenate([data,np.asarray(f[data_name],dtype=data_type)],0)
                ind = ind + 1
            #data = list(db[sid].data(det))
            #data = np.squeeze(data)
        
        sum_data = np.sum(data,0)
        if i == 0:
            sum_all_data = sum_data
        else:
            sum_all_data = sum_all_data + sum_data
    return sum_all_data

def sum_all_h5_data(file_list):
    # load a list of scans, with data being stacked at the first axis
    # roi[row_start,col_start,row_size,col_size]
    # mask has to be the same size of the image data, which corresponds to the last two axes
    data_type = 'float32'    
    num_scans = np.size(file_list)
    det = '/entry/instrument/detector/data'
    for i in tqdm(range(num_scans),desc="Progress"):
        file_name = file_list[i]
        f = h5py.File(file_name,'r')       
        
        data = f[det]
        sum_data = np.sum(data,0)
        if i == 0:
            sum_all_data = sum_data
        else:
            sum_all_data = sum_all_data + sum_data
    return sum_all_data

def load_scaler_data(file_list,scaler_list):
    # scaler_date[scans,scan_points,scalers]
    data_type = 'float32'
    num_scans = np.size(file_list)
    num_scalers = np.size(scaler_list)
    #scaler_data = np.array((num_scans,num_scalers))
    for i in tqdm(range(num_scans),desc="Progress"):
        file_name = file_list[i]
        with open(file_name,'r') as f:
            data = list(csv.reader(f,delimiter='\t'))
        num_scan_points = len(data)-1
        if i == 0:
            scaler_data = np.zeros((num_scans,num_scan_points,num_scalers),dtype=data_type)
        for j in range(num_scalers):
            scaler_name = scaler_list[j]
            ind = np.argwhere(np.array(data[0])==scaler_name)[0][0]
            for k in range(num_scan_points):
                scaler_data[i,k,j] = data[k+1][ind]
    return scaler_data
        
def interp_sub_pix(data, shift_matrix):
    # shift a series of image data based on shift_matrix 
    # 3D: regular stack of images; stacking axis is the first
    # 4D: angular-line scans with 2D diffraction data, [angle,pos,im_row,im_col]
    # 5D: angular-grid scans with 2D diffraction data, [angle,pos_y,pos_x,im_row,im_col]
    sz = np.shape(data)
    sz_len = np.size(sz)
    shift_data = np.zeros(np.shape(data),dtype=data.dtype)
    if sz_len == 3:     
        for i in tqdm(range(sz[0]),desc="Progress"):
            subset = data[i,:,:]
            ly = np.int(np.floor(shift_matrix[i,0]))
            hy = np.int(np.ceil(shift_matrix[i,0]))
            lx = np.int(np.floor(shift_matrix[i,1]))
            hx = np.int(np.ceil(shift_matrix[i,1]))
            lxly_subset = np.roll(subset,(ly,lx),axis=(0,1))
            lxhy_subset = np.roll(subset,(hy,lx),axis=(0,1))
            hxly_subset = np.roll(subset,(ly,hx),axis=(0,1))
            hxhy_subset = np.roll(subset,(hy,hx),axis=(0,1))
            ry = shift_matrix[i,0] - ly
            rx = shift_matrix[i,1] - lx
            shift_data[i,:,:] = (1-rx)*(1-ry)*lxly_subset + rx*(1-ry)*hxly_subset \
                                +(1-rx)*ry*lxhy_subset + rx*ry*hxhy_subset  
    elif sz_len == 4:        
        for i in tqdm(range(sz[0]),desc="Progress"):
            subset = data[i,:,:,:]
            l_bound = np.int(np.floor(shift_matrix[i]))
            h_bound = np.int(np.ceil(shift_matrix[i]))      
            l_subset = np.roll(subset,l_bound,axis=0)
            h_subset = np.roll(subset,h_bound,axis=0)
            r = shift_matrix[i] - l_bound
            shift_data[i,:,:,:] = (1-r)*l_subset + r*h_subset
    elif sz_len == 5:
        for i in tqdm(range(sz[0]),desc="Progress"):
            subset = data[i,:,:,:,:]
            ly = np.int(np.floor(shift_matrix[i,0]))
            hy = np.int(np.ceil(shift_matrix[i,0]))
            lx = np.int(np.floor(shift_matrix[i,1]))
            hx = np.int(np.ceil(shift_matrix[i,1]))
            lxly_subset = np.roll(subset,(ly,lx),axis=(0,1))
            lxhy_subset = np.roll(subset,(hy,lx),axis=(0,1))
            hxly_subset = np.roll(subset,(ly,hx),axis=(0,1))
            hxhy_subset = np.roll(subset,(hy,hx),axis=(0,1))
            ry = shift_matrix[i,0] - ly
            rx = shift_matrix[i,1] - lx
            shift_data[i,:,:,:,:] = (1-rx)*(1-ry)*lxly_subset + rx*(1-ry)*hxly_subset \
                                    +(1-rx)*ry*lxhy_subset + rx*ry*hxhy_subset
    else:
        print('Dimension of the data has to be 3D, 4D or 5D')
        print('3D: regular stack of images stacking along the first axis. \n \
               4D: angular-line scans with 2D diffraction data, [angle,pos,im_row,im_col] \n \
               5D: angular-grid scans with 2D diffraction data, [angle,pos_y,pos_x,im_row,im_col]')
    
    return shift_data

def trans_coor3D(X,Y,Z,M):
    sz = np.shape(X) 
    X = np.reshape(X,(1,-1))
    Y = np.reshape(Y,(1,-1))
    Z = np.reshape(Z,(1,-1))
    pos = np.array(M@np.concatenate([X,Y,Z],axis=0))
    cX = np.reshape(pos[0,:],sz)
    cY = np.reshape(pos[1,:],sz)
    cZ = np.reshape(pos[2,:],sz)
    return cX, cY, cZ

def create_grid(X,Y,Z,M):
 
    cX, cY, cZ = trans_coor3D(X,Y,Z,M)
    
    x = np.unique(np.round(cX,2))
    y = np.unique(np.round(cY,2))
    z = np.unique(np.round(cZ,2))
    
    min_x = np.amin(x)
    max_x = np.amax(x)

    min_y = np.amin(y)
    max_y = np.amax(y)

    min_z = np.amin(z)
    max_z = np.amax(z)
    
    vec = M[:,0]+M[:,1]+M[:,2]
    
    dx = np.abs(vec[0])
    dy = np.abs(vec[1])
    dz = np.abs(vec[2])
    
    x_rng = np.arange(min_x,max_x,dx)
    y_rng = np.arange(min_y,max_y,dy)
    z_rng = np.arange(min_z,max_z,dz)
    
    Vx, Vy, Vz = np.meshgrid(x_rng,y_rng,z_rng)
    pix_sz = np.reshape([dx,dy,dz],(3,1))
    
    return pix_sz, Vx, Vy, Vz


def interp3_oblique(X,Y,Z,V,M,Vx,Vy,Vz):
    
    sz = np.shape(V)
    maxX = np.amax(X)
    minX = np.amin(X)
    maxY = np.amax(Y)
    minY = np.amin(Y)
    maxZ = np.amax(Z)
    minZ = np.amin(Z)
    
    dX = (maxX-minX)/(sz[1]-1)
    dY = (maxY-minY)/(sz[0]-1)
    dZ = (maxZ-minZ)/(sz[2]-1)
    q_X, q_Y, q_Z = trans_coor3D(Vx,Vy,Vz,M)  
    
    n_X, r_X = np.divmod((q_X - minX)/dX,1)
    n_Y, r_Y = np.divmod((q_Y - minY)/dY,1)
    n_Z, r_Z = np.divmod((q_Z - minZ)/dZ,1)
    
    #n_X = np.array(np.floor((q_X - minX)/dX),dtype="int32") 
    #n_Y = np.array(np.floor((q_Y - minY)/dY),dtype="int32") 
    #n_Z = np.array(np.floor((q_Z - minZ)/dZ),dtype="int32") 
    
    #r_X = np.array((q_X - minX)/dX) - n_X
    #r_Y = np.array((q_Y - minY)/dY) - n_Y
    #r_Z = np.array((q_Z - minZ)/dZ) - n_Z
    
    n_X = np.array(np.ndarray.flatten(n_X),dtype='int32')
    n_Y = np.array(np.ndarray.flatten(n_Y),dtype='int32')
    n_Z = np.array(np.ndarray.flatten(n_Z),dtype='int32') 
    
    r_X = np.ndarray.flatten(r_X)
    r_Y = np.ndarray.flatten(r_Y)
    r_Z = np.ndarray.flatten(r_Z)
    
    #r_X = np.ndarray.flatten(np.mod(q_X-minX, dX))
    #r_Y = np.ndarray.flatten(np.mod(q_Y-minY, dY))
    #r_Z = np.ndarray.flatten(np.mod(q_Z-minZ, dZ))

    mask = np.ones(np.shape(n_X))
    mask[n_X <0] = 0
    mask[n_X >= sz[1]-1] = 0
    mask[n_Y <0] = 0
    mask[n_Y >= sz[0]-1] = 0
    mask[n_Z <0] = 0
    mask[n_Z >= sz[2]-1] = 0
    
    n_X[n_X < 0] = 0
    n_X[n_X >= sz[1]-1] = sz[1]-2
    
    n_Y[n_Y < 0] = 0
    n_Y[n_Y >= sz[0]-1] = sz[0]-2
    
    n_Z[n_Z < 0] = 0
    n_Z[n_Z >= sz[2]-1] = sz[2]-2
    
    ind1 = (n_Y, n_X, n_Z)
    ind2 = (n_Y+1, n_X, n_Z)
    ind3 = (n_Y, n_X+1, n_Z)
    ind4 = (n_Y, n_X, n_Z+1)
    ind5 = (n_Y+1, n_X+1, n_Z)
    ind6 = (n_Y+1, n_X, n_Z+1)
    ind7 = (n_Y, n_X+1, n_Z+1)
    ind8 = (n_Y+1, n_X+1, n_Z+1)
    
    Vq = V[ind1]*(1-r_X)*(1-r_Y)*(1-r_Z)+V[ind2]*(1-r_X)*r_Y*(1-r_Z) \
        +V[ind3]*r_X*(1-r_Y)*(1-r_Z)+V[ind4]*(1-r_X)*(1-r_Y)*r_Z \
        +V[ind5]*r_X*r_Y*(1-r_Z)+V[ind6]*(1-r_X)*r_Y*r_Z \
        +V[ind7]*r_X*(1-r_Y)*r_Z+V[ind8]*r_X*r_Y*r_Z
    #Vq[np.isnan(Vq)] = 0
    #Vq[Vq < 0] = 0
    return np.reshape(Vq*mask,np.shape(Vx))
class RSM:
    def __init__(self,det_data,energy,delta,gamma,num_angle,th_step,pix,det_dist,offset):
        # input det_data [angle,position, det_row,det_col]
        # output rsm [position,q_y, q_x, q_z]
        self.energy = energy
        self.delta = -delta*np.pi/180
        self.gamma = -gamma*np.pi/180
        self.num_angle = num_angle
        self.th_step = th_step*np.pi/180
        self.pix = pix
        self.det_dist = det_dist
        self.offset = offset
        self.det_data = det_data
        self.k = 1e4/(12.398/energy)
    def calcRSM(self,coor,data_store = 'reduced'):
        sz = np.shape(self.det_data)
        data_type = self.det_data.dtype
        sz_len = np.size(sz)
        det_row = sz[sz_len-2]
        det_col = sz[sz_len-1]
        Mx = np.matrix([[1., 0., 0.],[0.,np.cos(self.delta),-np.sin(self.delta)],[0.,np.sin(self.delta),np.cos(self.delta)]])
        My = np.matrix([[np.cos(self.gamma),0.,np.sin(self.gamma)],[0.,1.,0.],[-np.sin(self.gamma),0.,np.cos(self.gamma)]])
        M_D2L = My@Mx
        M_L2D = np.linalg.inv(M_D2L)
        kx_lab = M_D2L@np.array([[1.],[0.],[0.]])*self.k*(self.pix/self.det_dist)
        ky_lab = M_D2L@np.array([[0.],[1.],[0.]])*self.k*(self.pix/self.det_dist)
        k_0 = self.k*M_L2D@np.array([[0.],[0.],[1.]])
        h = self.k*np.array([[0.],[0.],[1.]])-k_0
        rock_z = np.cross((M_L2D@np.array([[0.],[1.],[0.]])).T,h.T).T
        kz = -rock_z*self.th_step
        kz_lab = M_D2L@kz
        M_O2L = np.concatenate([kx_lab,ky_lab,kz_lab],axis = 1)
        M_L2O = np.linalg.inv(M_O2L)
        x_rng = np.linspace(1-round(det_col/2),det_col-round(det_col/2),det_col)
        y_rng = np.linspace(1-round(det_row/2),det_row-round(det_row/2),det_row)
        z_rng = np.linspace(1-round(self.num_angle/2),self.num_angle-round(self.num_angle/2),self.num_angle)
        X, Y, Z = np.meshgrid(x_rng, y_rng, z_rng)
        X = X+self.offset[1]
        Y = Y+self.offset[0]
        ux_cryst = kz_lab/np.linalg.norm(kz_lab)
        uz_cryst = M_D2L@h/np.linalg.norm(M_D2L@h)
        uy_cryst = np.cross(uz_cryst.T,ux_cryst.T).T
        M_C2L = np.concatenate([ux_cryst,uy_cryst,uz_cryst],axis = 1)
        M_C2O = M_L2O@M_C2L
        M_O2C = np.linalg.inv(M_C2O)

        self.M_O2L = M_O2L
        self.M_L2O = M_L2O
        self.M_O2C = M_O2C
        self.M_C2O = M_C2O
        #self.M_O2B = M_O2B
        #self.M_B2O = M_B2O
        self.M_C2L = M_C2L
        self.M_L2C = np.linalg.inv(M_C2L)
        
        if coor == 'lab':
            M = M_O2L
            M_inv = M_L2O
        elif coor == 'cryst':
            M = M_O2C
            M_inv = M_C2O
        elif coor == 'cryst_beam_integrated':
            M = M_O2L
            M_inv = M_L2O
            orig_store = data_store
            data_store = 'full'
        else:
            print('coor must be lab or cryst')
        self.coor = coor
         
        pix_sz, xq, yq, zq = create_grid(X,Y,Z,M)
        trans_sz = np.shape(xq) 
        
        
        # move angle axis (first) to to the last, and then reshape positions to one list
        self.det_data = np.squeeze(np.swapaxes(np.expand_dims(self.det_data,axis = -1),0,-1),0)
        sz = np.shape(self.det_data)
        self.det_data = np.reshape(self.det_data,[-1,sz[-3],sz[-2],sz[-1]])
        new_sz = np.shape(self.det_data)
        
        if data_store =='full':
            self.full_data = np.zeros((new_sz[0],trans_sz[0],trans_sz[1],trans_sz[2]),dtype=data_type)
        else:
            self.qxz_data = np.zeros((new_sz[0],trans_sz[1],trans_sz[2]),dtype=data_type)
            self.qyz_data = np.zeros((new_sz[0],trans_sz[0],trans_sz[2]),dtype=data_type)
        for i in tqdm(range(new_sz[0]),desc="Progress"):
            vq = interp3_oblique(X, Y, Z, self.det_data[i,:,:,:], M_inv, xq, yq, zq)
            if data_store == 'full':
                self.full_data[i,:,:,:] = vq
            else:
                self.qxz_data[i,:,:] = np.sum(vq,0)
                self.qyz_data[i,:,:] = np.sum(vq,1)
        
        #del self.det_data
        if coor == 'cryst_beam_integrated':
            pix_sz, cryst_xq, cryst_yq, cryst_zq = create_grid(X,Y,Z,self.M_O2C)
            cryst_sz = np.shape(cryst_xq)
            self.cryst_data = np.zeros((new_sz[0],cryst_sz[0],cryst_sz[1],cryst_sz[2]),dtype=data_type)
            trans_sz = cryst_sz
            for i in tqdm(range(new_sz[0]),desc="Progress"):
                vq = rsm_cen_x_y(self.full_data[i,:,:,:])
                vq = interp3_oblique(xq, yq, zq, vq, self.M_C2L,cryst_xq, cryst_yq, cryst_zq)
                self.cryst_data[i,:,:,:] = vq
            self.full_data = self.cryst_data
            xq = cryst_xq
            yq = cryst_yq
            zq = cryst_zq
                
        ''' 
        
        if np.size(sz) == 5:
            if data_store == 'full':
                self.full_data = np.zeros((sz[1],sz[2],trans_sz[0],trans_sz[1],trans_sz[2]),dtype=data_type)
            else:
                self.qxz_data = np.zeros((sz[1],sz[2],trans_sz[1],trans_sz[2]),dtype=data_type)
                self.qyz_data = np.zeros((sz[1],sz[2],trans_sz[0],trans_sz[2]),dtype=data_type)
                
            for i in tqdm(range(sz[1]),desc="Progress"):
                for j in range(sz[2]):
                    data = self.det_data[:,i,j,:,:]
                    # move angle axis to the last
                    data = np.swapaxes(np.swapaxes(data,0,1),1,2)
                    vq = interp3_oblique(X, Y, Z, data, M_inv, xq, yq, zq)
                    if data_store == 'full':
                        self.full_data[i,j,:,:,:] = vq
                    else:
                        self.qxz_data[i,j,:,:] = np.sum(vq,0)
                        self.qyz_data[i,j,:,:] = np.sum(vq,1)
        elif np.size(sz) == 4:
            if data_store == 'full':
                self.full_data = np.zeros((sz[1],trans_sz[0],trans_sz[1],trans_sz[2]),dtype=data_type)
            else:
                self.qxz_data = np.zeros((sz[1],trans_sz[1],trans_sz[2]),dtype=data_type)
                self.qyz_data = np.zeros((sz[1],trans_sz[0],trans_sz[2]),dtype=data_type) 
            for i in tqdm(range(sz[1]),desc="Progress"):
                data = self.det_data[:,i,:,:]
                data = np.swapaxes(np.swapaxes(data,0,1),1,2)
                vq = interp3_oblique(X, Y, Z, data, M_inv, xq, yq, zq)
                if data_store == 'full':
                    self.full_data[i,:,:,:] = vq
                else:
                    self.qxz_data[i,:,:] = np.sum(vq,0)
                    self.qyz_data[i,:,:] = np.sum(vq,1)
        elif np.size(sz) == 3:
            if data_store == 'full':
                self.full_data = np.zeros((trans_sz[0],trans_sz[1],trans_sz[2]),dtype=data_type)
            else:
                self.qxz_data = np.zeros((trans_sz[1],trans_sz[2]),dtype=data_type)
                self.qyz_data = np.zeros((trans_sz[0],trans_sz[2]),dtype=data_type)
            data = self.det_data
            data = np.swapaxes(np.swapaxes(data,0,1),1,2)
            vq = interp3_oblique(X, Y, Z, data, M_inv, xq, yq, zq)
            if data_store == 'full':
                self.full_data = vq
            else:
                self.qxz_data = np.sum(vq,0)
                self.qyz_data = np.sum(vq,1)
        '''     
        
        self.xq = xq
        self.yq = yq
        self.zq = zq
        self.h = h
        self.X = X
        self.Y = Y
        self.Z = Z
       
        # reshape the result
        if not data_store == 'full':
            del self.det_data
            self.qxz_data = np.reshape(self.qxz_data,np.concatenate((sz[0:-3],[trans_sz[1],trans_sz[2]]),axis=0))
            self.qyz_data = np.reshape(self.qyz_data,np.concatenate((sz[0:-3],[trans_sz[0],trans_sz[2]]),axis=0))
            print("raw det_data is deleted")
            print("qxz_data: [pos,qx,qz] with dimensions of {}".format(self.qxz_data.shape))
            print("qyz_data: [pos,qy,qz] with dimensions of {}".format(self.qyz_data.shape))      
        else:
            self.full_data = np.reshape(self.full_data,np.concatenate((sz[0:-3],trans_sz[0:3]),axis=0))
            print("det_data: raw aligned det data, [pos,det_row,det_col,angles] with dimensions of {}".format(self.det_data.shape))
            print("full_data: 3D rsm, [pos,qy,qx,qz] with dimensions of {}".format(self.full_data.shape))
        self.data_store = data_store
      
        
    def transBEAM(self,pix,det_dist,energy,im):
        # load a reference trasmission image
        # transfer it to crystal coordinates
        # use it as psf for deconvolution of RSM 
        
        # create a 3D array in lab coordinates with the central frame is
        # the input image, and its corresponding q values are X,Y,and Z
        
        k = 1e4/(12.398/energy)
        det_row,det_col = np.shape(im)
        z_num = (det_row + det_col)//2
        
        x_rng = np.linspace(1-round(det_col/2),det_col-round(det_col/2),det_col)
        y_rng = np.linspace(1-round(det_row/2),det_row-round(det_row/2),det_row)
        z_rng = np.linspace(1-round(z_num/2),z_num-round(z_num/2),z_num)
        X, Y, Z = np.meshgrid(x_rng, y_rng, z_rng)
        X = X*(pix/det_dist)*k
        Y = Y*(pix/det_dist)*k
        Z = Z*(pix/det_dist)*k
        data = np.zeros(np.shape(X))
        data[:,:,z_num//2] = im
        #data[:,:,z_num//2-1] = im

        # create a grid in crystal coordinates to be assinged values
        # based on the reference image

        # compute the range in crystal coordinates
        pix_sz, xq, yq, zq = create_grid(X,Y,Z,self.M_L2C)
        xq_rng = np.abs(xq[0,0,0] - xq[0,-1,0])
        yq_rng = np.abs(yq[0,0,0] - yq[-1,0,0])
        zq_rng = np.abs(zq[0,0,0] - zq[0,0,-1])
        #print('xq_rng = %4.3f\tyq_rng = %4.3f\tzq_rng = %4.3f'%(xq_rng,yq_rng,zq_rng))
        dxq = self.xq[0,1,0]-self.xq[0,0,0]
        dyq = self.yq[1,0,0]-self.yq[0,0,0]
        dzq = self.zq[0,0,1]-self.zq[0,0,0]

        xq_list = np.arange(-(xq_rng/2)*np.sign(dxq),xq_rng/2,dxq)
        yq_list = np.arange(-(yq_rng/2)*np.sign(dyq),yq_rng/2,dyq)
        zq_list = np.arange(-(zq_rng/2)*np.sign(dzq),zq_rng/2,dzq)

        xq, yq, zq = np.meshgrid(xq_list, yq_list, zq_list)

        vq = interp3_oblique(X, Y, Z, data,self.M_C2L, xq, yq, zq)

        self.beam_cryst = vq
    
    def integrateBeam(self):
        if self.coor == 'cryst':
            pix_sz, xq, yq, zq = create_grid(self.xq,self.yq,self.zq,self.M_C2L)
            if not self.data_store == 'full':
                sz = np.shape(self.qxz_data)
                xq_sz = np.shape(self.xq)
                self.qxz_data = np.reshape(self.qxz_data,[-1,sz[-2],sz[-1]])
                tot_pos = self.qxz_data.shape[0]
                for i in tqdm(range(tot_pos),desc='Progress'):
                    data = np.zeros(np.shape(self.xq))
                    data[xq_sz[0]//2,:,:] = self.qxz_data[i,:,:]
                    vq = interp3_oblique(self.xq,self.yq,self.zq,data,self.M_L2C, xq, yq, zq)
                    vq = rsm_cen_x_y(vq)
                    new_vq = interp3_oblique(xq,yq,zq,vq,self.M_C2L, self.xq, self.yq, self.zq)
                    self.qxz_data[i,:,:] = np.sum(new_vq,0)
                self.qxz_data = np.reshape(self.qxz_data,sz)
                
                sz = np.shape(self.qyz_data)
                yq_sz = np.shape(self.yq)
                self.qyz_data = np.reshape(self.qyz_data,[-1,sz[-2],sz[-1]])
                tot_pos = self.qyz_data.shape[0]
                for i in tqdm(range(tot_pos),desc='Progress'):
                    data = np.zeros(np.shape(self.yq))
                    data[:,yq_sz[1]//2:,:] = self.qyz_data[i,:,:]
                    vq = interp3_oblique(self.xq,self.yq,self.zq,data,self.M_L2C, xq, yq, zq)
                    vq = rsm_cen_x_y(vq)
                    new_vq = interp3_oblique(xq,yq,zq,vq,self.M_C2L, self.xq, self.yq, self.zq)
                    self.qyz_data[i,:,:] = np.sum(new_vq,1)
                self.qyz_data = np.reshape(self.qyz_data,sz)
            else:
                sz = np.shape(self.full_data)
                xq_sz = np.shape(self.xq)
                self.full_data = np.reshape(self.full_data,[-1,sz[-3],sz[-2],sz[-1]])
                tot_pos = self.full_data.shape[0]
                for i in tqdm(range(tot_pos),desc='Progress'):
                    data = self.full_data[i,:,:,:]
                    vq = interp3_oblique(self.xq,self.yq,self.zq,self.full_data[i,:,:,:],self.M_L2C, xq, yq, zq)
                    vq = rsm_cen_x_y(vq)
                    new_vq = interp3_oblique(xq,yq,zq,vq,self.M_C2L, self.xq, self.yq, self.zq)
                    self.full_data[i,:,:,:] = new_vq
                self.full_data = np.reshape(self.full_data,sz)
                
    def calcSTRAIN(self, method):

        if self.data_store == 'full':
            #sz = np.shape(self.full_data)
            qz_pos = np.sum(np.sum(self.full_data,-2),-2)
            qx_pos = np.sum(np.sum(self.full_data,-1),-2)
            qy_pos = np.sum(np.sum(self.full_data,-1),-1)
        else:
            #sz = np.shape(self.qxz_data)
            qz_pos = np.sum(self.qxz_data,-2)
            qx_pos = np.sum(self.qxz_data,-1)
            qy_pos = np.sum(self.qyz_data,-1)    
            
        sz = np.shape(qz_pos)
        if np.size(sz) == 3:
            shift_qz = np.zeros((sz[0],sz[1]))
            shift_qx = np.zeros((sz[0],sz[1]))
            shift_qy = np.zeros((sz[0],sz[1]))
            for i in tqdm(range(sz[0])):
                for j in range(sz[1]):
                    if method == 'com':
                        shift_qz[i,j] = cen_of_mass(qz_pos[i,j,:])
                        shift_qy[i,j] = cen_of_mass(qy_pos[i,j,:])
                        shift_qx[i,j] = cen_of_mass(qx_pos[i,j,:])
        elif np.size(sz) == 2:
            shift_qz = np.zeros((sz[0]))
            shift_qx = np.zeros((sz[0]))
            shift_qy = np.zeros((sz[0]))
            for i in tqdm(range(sz[0])):
                if method == 'com':
                    shift_qz[i] = cen_of_mass(qz_pos[i,:])
                    shift_qy[i] = cen_of_mass(qy_pos[i,:])
                    shift_qx[i] = cen_of_mass(qx_pos[i,:])
        self.strain = -shift_qz*(self.zq[0,0,1]-self.zq[0,0,0])/np.linalg.norm(self.h)
        self.tilt_x = shift_qx*(self.xq[0,1,0]-self.xq[0,0,0])/np.linalg.norm(self.h)
        self.tilt_y = shift_qy*(self.yq[1,0,0]-self.yq[0,0,0])/np.linalg.norm(self.h)
        self.tot = np.sum(qz_pos,-1)
    
    def disp(self):
        
        fig = plt.figure(1)
        fig.set_size_inches(8, 6)
        fig.set_dpi(160)
        
        sz = np.shape(self.strain)
        if np.size(sz) == 2:
            ax = plt.subplot(2,2,1)
            im = ax.imshow(self.tot)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('total intensity (cts)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        
            ax = plt.subplot(2,2,2)
            im = ax.imshow(self.strain*100)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('strain (%)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        
            ax = plt.subplot(2,2,3)
            im = ax.imshow(self.tilt_x*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('tilt_x (degree)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
         
            ax = plt.subplot(2,2,4)
            im = ax.imshow(self.tilt_y*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('tilt_y (degree)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im,cax=cax)
        elif np.size(sz) == 1:
            ax = plt.subplot(2,2,1)
            im = ax.plot(self.tot)
            ax.set_xlabel('x')
            ax.set_ylabel('total intensity (cts)')
             
            ax = plt.subplot(2,2,2)
            im = ax.plot(self.strain*100)
            ax.set_xlabel('x')
            ax.set_ylabel('strain (%)')
             
            ax = plt.subplot(2,2,3)
            im = ax.plot(self.tilt_x*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('tilt_x (degree)')
         
            ax = plt.subplot(2,2,4)
            im = ax.imshow(self.tilt_y*180/np.pi)
            ax.set_xlabel('x')
            ax.set_ylabel('tilt_y (degree)')
        plt.tight_layout()
        #plt.savefig('./result.png')
        
    def save(self,output_path):

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("Directory '%s' created" %output_path)
        
        file_name = ''.join([output_path,'tot_intensity_map.tif'])
        tf.imsave(file_name,np.asarray(self.tot,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'strain_map.tif'])
        tf.imwrite(file_name,np.asarray(self.strain,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'tilt_x_map.tif'])
        tf.imwrite(file_name,np.asarray(self.tilt_x,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'tilt_y_map.tif'])
        tf.imwrite(file_name,np.asarray(self.tilt_y,dtype=np.float32),imagej=True)

        file_name = ''.join([output_path,'pos_rsm_xz.obj'])
        if self.data_store == 'full':
            pickle.dump(np.sum(self.full_data,-3),open(file_name,'wb'), protocol = 4)
        else:
            pickle.dump(self.qxz_data,open(file_name,'wb'), protocol = 4)

        file_name = ''.join([output_path,'pos_rsm_yz.obj'])
        if self.data_store == 'full':
            pickle.dump(np.sum(self.full_data,-2),open(file_name,'wb'),protocol = 4)
        else:
            pickle.dump(self.qyz_data,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'xq.obj'])
        pickle.dump(self.xq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'yq.obj'])
        pickle.dump(self.yq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'zq.obj'])
        pickle.dump(self.zq,open(file_name,'wb'),protocol = 4)

        file_name = ''.join([output_path,'h.obj'])
        pickle.dump(self.h,open(file_name,'wb'),protocol = 4)

        if plt.fignum_exists(1):
            file_name = ''.join([output_path,'results.png'])
            plt.savefig(file_name)
def cen_of_mass(c):
    c = c.ravel()
    tot = np.sum(c)
    n = np.size(c)
    a = 0
    idx = n//2
    for i in range(n):
        a = a + c[i]
        if a > tot/2:
            idx = i - (a-tot/2)/c[i]
            break
    return idx
def rsm_cen_x_y(data):
    sz = np.shape(data)
    new_data = np.zeros(data.shape,dtype=data.dtype)
    im_xz = np.sum(data,0)
    im_yz = np.sum(data,1)
    
    if len(sz) == 3:
        for i in range(sz[2]):
            
            x_cen = cen_of_mass(im_xz[:,i])
            y_cen = cen_of_mass(im_yz[:,i])
            tot = np.sum(im_xz[:,i])
            row = int(np.floor(y_cen))
            col = int(np.floor(x_cen))
            new_data[row,col,i] = (1 - (y_cen - row))*(1 - (x_cen - col))*tot
            new_data[row,col+1,i] = (1 - (y_cen - row))*((x_cen - col))*tot
            new_data[row+1,col,i] = ((y_cen - row))*(1 - (x_cen - col))*tot
            new_data[row+1,col+1,i] = ((y_cen - row))*((x_cen - col))*tot
    else:
        print("must be 3D rsm data in lab (beam) coordinates")
    return new_data

def get_path(scan_id, key_name='merlin1', db=db):
    """Return file path with given scan id and keyname.
    """
    
    h = db[int(scan_id)]
    e = list(db.get_events(h, fields=[key_name]))
    #id_list = [v.data[key_name] for v in e]
    id_list = [v['data'][key_name] for v in e]
    rootpath = db.reg.resource_given_datum_id(id_list[0])['root']
    flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
    flist = set(flist)
    fpath = [os.path.join(rootpath, file_path) for file_path in flist]
    return fpath

def interactive_map(names,im_stack,label,data_4D, cmap='jet', clim=None, marker_color = 'black'):

    l = len(names)
    im_sz = np.shape(im_stack)
    l = np.fmin(l,im_sz[0])

    num_maps = l + 1
    layout_row = np.round(np.sqrt(num_maps))
    layout_col = np.ceil(num_maps/layout_row)
    layout_row = int (layout_row)
    layout_col = int (layout_col)

    fig, axs = plt.subplots(layout_row,layout_col)
    size_y = layout_row*4
    size_x = layout_col*6
    if size_x < 8:
        size_y = size_y*8/size_x
        size_x = 8
    if size_y < 6:
        size_x = size_x*6/size_y
        size_y = 6
    fig.set_size_inches(size_x,size_y)
    for i in range(l):
        axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
        axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
    axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[0,0,:,:],cmap=cmap,clim=clim)
    axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
    if layout_col*layout_row > num_maps:
        for i in range(num_maps,layout_row*layout_col):
            axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
    fig.tight_layout()


    def onclick(event):
        global row, col
        col, row = event.xdata, event.ydata
        if col is not None and row is not None and col <= im_sz[2] and row <= im_sz[1]:
            row = int(np.round(row))
            col = int(np.round(col))
            for i in range(l):
                axs[np.unravel_index(i,[layout_row,layout_col])].clear()
                axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
                axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
                axs[np.unravel_index(i,[layout_row,layout_col])].plot(col,row,marker='o',markersize=4, color=marker_color)
            axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[row,col,:,:],cmap=cmap,clim=clim)
            axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def load_diff_data(sid,scaler_names,det_name, mon = None):
    h = db[sid]
    df = h.table()
    s = scan_command(sid)
    x_mot = s.split()[0]
    y_mot = s.split()[4]
    scan_col = int (s.split()[3])
    scan_row = int (s.split()[7])
    diff_data = list(h.data(det_name))
    im_stack = []
    print(f'row = {scan_row} col = {scan_col}')
    for name in scaler_names:

        if name in df:
            tmp = df[name]
        else:
            tmp = (df['Det1_{}'.format(name)] + df['Det2_{}'.format(name)] + df['Det3_{}'.format(name)])
        #print(np.shape(tmp))
        tmp = np.reshape(np.asarray(tmp),(1,scan_row, scan_col))

        if len(im_stack) == 0 :
            im_stack = tmp
        else:
            im_stack = np.concatenate([im_stack,tmp],axis=0)
    if mon is not None:
        mon_var = df[mon]
        im_stack = im_stack/np.expand_dims(mon_var,0)
    sz = np.shape(diff_data)
    return im_stack, np.reshape(diff_data,(scan_row,scan_col,sz[2],sz[3]))

def create_movie(desc, names,im_stack,label,data_4D,path,cmap='jet',color='white'):
    # desc: a dictionary. Example,
    # desc ={
    #    'title':'Movie',
    #    'artist': 'hyan',
    #    'comment': 'Blanket film',
    #    'save_file': 'movie_blanket_film.mp4',
    #    'fps': 15,
    #    'dpi': 100
    # }
    # names: names of the individual im in im_stack
    # label: name of the 4D dataset
    # data_4D: the 4D dataset, for example, [row, col, qx, qz]
    # path: sampled positions for the movie, a list of [row, col]
    # cmap: color scheme of the plot
    # color: color of the marker
    
    l = len(names)
    im_sz = np.shape(im_stack)
    l = np.fmin(l,im_sz[0])

    num_maps = l + 1
    layout_row = np.round(np.sqrt(num_maps))
    layout_col = np.ceil(num_maps/layout_row)
    layout_row = int (layout_row)
    layout_col = int (layout_col)
    #plt.figure()
    fig, axs = plt.subplots(layout_row,layout_col)
    size_y = layout_row*4
    size_x = layout_col*6
    if size_x < 8:
        size_y = size_y*8/size_x
        size_x = 8
    if size_y < 6:
        size_x = size_x*6/size_y
        size_y = 6
    fig.set_size_inches(size_x,size_y)
    for i in range(l):
        axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
        axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
    axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[0,0,:,:],cmap=cmap)
    axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
    if layout_col*layout_row > num_maps:
        for i in range(num_maps,layout_row*layout_col):
            axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
    fig.tight_layout()
    
    def update_fig(row,col,cmap=cmap,color=color):

        plt.cla()
        for i in range(l):
            axs[np.unravel_index(i,[layout_row,layout_col])].imshow(im_stack[i,:,:],cmap=cmap)
            axs[np.unravel_index(i,[layout_row,layout_col])].plot(col,row,marker='o',markersize=2, color=color)
            axs[np.unravel_index(i,[layout_row,layout_col])].set_title(names[i])
        axs[np.unravel_index(l,[layout_row,layout_col])].imshow(data_4D[row,col,:,:],cmap=cmap)
        axs[np.unravel_index(l,[layout_row,layout_col])].set_title(label)
        if layout_col*layout_row > num_maps:
            for i in range(num_maps,layout_row*layout_col):
                axs[np.unravel_index(i,[layout_row,layout_col])].axis('off')
        fig.tight_layout()
        #fig.canvas.draw_idle()
        return 
     
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=desc['title'], artist=desc['artist'],
                comment=desc['comment'])
    writer = FFMpegWriter(fps=desc['fps'], metadata=metadata)
     
    with writer.saving(fig, desc['save_file'], dpi=desc['dpi']):
        writer.grab_frame()
         
        for j in tqdm(range(len(path)),desc='Progress'):       
            update_fig(path[j,0],path[j,1],cmap=cmap,color=color)
            writer.grab_frame()
    writer.finish()

def get_file_creation_time(file_path):
    try:
        return os.path.getctime(file_path)
    except OSError:
        # If there is an error (e.g., file not found), return 0
        return 0

def sort_files_by_creation_time(file_list):
    # Sort the file list based on their creation time
    return sorted(file_list, key=lambda file: get_file_creation_time(file))