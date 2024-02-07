#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:33:24 2023

@author: ioannis
"""
import tempfile
import tarfile
import xmltodict
from pathlib import Path

import numpy as np
import pandas as pd
# from prodict import Prodict
# import re
import ast
from typing import Tuple
import json

from otb_matrices import otb_patch_map
# INPUTS

# just a small file

def elec_2dmap(array_code: str):
    if not array_code in otb_patch_map.keys():
        print(f'Electrode array "{array_code}" was not found')
        return None
    return otb_patch_map[array_code]['ElChannelMap']

    

def extract_epochs(data: np.ndarray, fsamp:int, events:list, pre_offset_sec:float=0, post_offset_sec:float=0):
    
    num_epochs = len(events)
    data_epoched = [None] * num_epochs
    pre = np.floor(pre_offset_sec*fsamp).astype(int)
    post = np.floor(post_offset_sec*fsamp).astype(int)

    for idx, evnt in enumerate(events):
        data_epoched[idx] = data[:, evnt['start']-pre: evnt['end']+post]
        
    return data_epoched

def import_otb(filename: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Imports .otb files and returns data matrix with channel timeseries and 
    their corresponding metadata

    Parameters
    ----------
    filename : str
        `.otb` filename to be imported

    Returns
    -------
    data: np.ndarray, (n_channels, n_samps)
        Time-serieses (n_channels, n_samps)
    channel_df: pd.DataFrame
        Channel metadata
    """
    # Unzip file to temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    with tarfile.open(filename) as tar:
        print('Imput file: {}'.format(filename))
        tar.extractall(path=temp_dir.name)
        print('Extracted .otb file in temporary dir!')

    # Load session metadata
    with open(temp_path / 'patient.xml') as fd:
        # Avoid @ for attributes
        session = xmltodict.parse(fd.read(), attr_prefix='')

    # Read device and acquition specific metadata
    sigs = list(temp_path.rglob('*.sig'))
    if len(sigs) > 1:
        raise ('More than 1 .sig files were found, check ... This code expects one sig file per .otb file')

    abstract_fname = sigs[0].stem + '.xml'
    with open(temp_path / abstract_fname) as fd:
        # Avoid @ for attributes
        abst = xmltodict.parse(fd.read(), attr_prefix='')
        device = abst['Device']
        PowerSupply = 5

        # ensure channel metadata are in homogenious format i.e. list[dict]
        for i in range(len(device['Channels']['Adapter'])):
            # case of single channel
            if isinstance(device['Channels']['Adapter'][i]['Channel'], dict):
                device['Channels']['Adapter'][i]['Channel'] = [
                    device['Channels']['Adapter'][i]['Channel']]

        # convert device's arithmetic string attributes to numbers
        for key, value in device.items():
            if isinstance(value, str) and value.isnumeric():
                # string to int/float, safely
                device[key] = ast.literal_eval(value)

    # Extract channel metadata
    # 1. extending it on channel level, 2. apply simple DF operations

    # Data import and type casting
    channel_df = pd.DataFrame(device['Channels']['Adapter']).explode(
        'Channel').reset_index(drop=True)
    channel_df = channel_df.astype({'Gain': 'float',
                                    'HighPassFilter': 'float',
                                    'LowPassFilter': 'float',
                                    'AdapterIndex': 'int',
                                    'ChannelStartIndex': 'int'})

    # Extract adapter specific information to the channel
    adapter_info = channel_df.apply(lambda row:
                                    {'adapter_name': row['Channel']['Prefix'],
                                     'sensor': row['Channel']['ID'],
                                     'sensor_description': row['Channel']['Description'],
                                     # relative index
                                     'rel_index': int(row['Channel']['Index']),
                                     'side': row['Channel']['Side'],
                                        'muscle': row['Channel']['Muscle']
                                     },
                                    axis='columns', result_type='expand')

    channel_df = pd.concat([channel_df, adapter_info], axis='columns')
    channel_df.drop('Channel', axis=1, inplace=True)

    # Detect HDsEMG array number
    tmp = channel_df['adapter_name'].str.findall(r"MULTIPLE IN ([0-9]{1})")
    channel_df['array_num'] = tmp.apply(
        lambda x:  int(x[0]) if len(x) == 1 else None)

    # Make sure channel indeces are correct
    channel_df['channel_index'] = channel_df['ChannelStartIndex'] + \
        channel_df['rel_index']
    channel_df.set_index('channel_index', verify_integrity=True)

    # add sampling frequency and Digital to Analog multiplier
    channel_df['fsamp'] = device['SampleFrequency']
    # d2a_fn = lambda Vref, nAD, gain: Vref/(2**nAD)*1e6/gain
    channel_df['dig2analog_factor'] = channel_df['Gain'].apply(
        lambda G: PowerSupply/(2**device['ad_bits'])*1e3/G)

    # rename columns
    channel_df.rename(columns={'ID': 'channel_type',
                               'Gain': 'gain',
                               'HighPassFilter': 'HP_filter',
                               'LowPassFilter': 'LP_filter'},
                      inplace=True)

    # extract only the relative
    channel_meta_df = channel_df[['channel_index', 'array_num', 'rel_index',
                                  'sensor', 'channel_type', 'gain', 'HP_filter', 'LP_filter',
                                  'fsamp', 'dig2analog_factor',
                                  'side', 'muscle', 'adapter_name']]

    """
    Load raw signal
    """
    # h=fopen(fullfile('tmpopen',signals(nSig).name),'r');
    # data=fread(h,[nChannel{nSig} Inf],'short');
    # binary file read: https://stackoverflow.com/a/14720675
    n_channels = len(channel_meta_df)
    with open(sigs[0], 'rb') as fd:
        data = np.fromfile(fd, np.int16).reshape((-1, n_channels)).T

    d2a_factor = channel_meta_df['dig2analog_factor'].to_numpy()

    # Digital to analog
    # multiply vector elements, row
    data = np.einsum('i,ij->ij', d2a_factor, data)

    temp_dir.cleanup()  # delete temporary folder

    return data, channel_meta_df

def buffer_to_array(buffer:bytes, n_channels: int, n_samps: int, dtype:np.dtype, index_order:str='C', byte_ordering:str='I')->np.array:
    """ Converts a bytestream buffer to 2d or 1d array, expecting multidimentional timeseries array.

    Parameters
    ----------
    buffer : buffer_like
        An object that exposes the buffer interface.
    n_channels : int
        Number of channels i.e. n_rows of the 2d array, represented in the buffer.
    n_samps : int
        Number of sample i.e. n_cols of the 2d array, , represented in the buffer.
    dtype : data-type
        Data-type of the returned array
    byte_ordering : str
        Byte ordering (for a single sample). Should match the byte representation from the byte encoder side. Can be 'little', 'big' for little and big endian respectively.  ee numpy.dtype.newbyteorder.
    index_order : str
        Index/order of different samples e.g C-like or F-like array indexing. See numpy.reshape for more.

    Returns
    -------
    np.array
        2D (or 1D) array, Multidimentional timeseries array, expected dimentions (n_channels, n_samps)
    """
    
    # TODO: perform check based on dimentions
    dt = np.dtype(dtype).newbyteorder(byte_ordering)
    dat = np.frombuffer(buffer, dtype=dt, count=n_channels*n_samps)
    dat = dat.reshape((n_channels, n_samps), order=index_order)
    return dat

def array_to_buffer(data:np.array, index_order:str='C', convert_byte_ordering_to:str='I')-> bytes:
    """ Converts a 2D array to bytes.

    Parameters
    ----------
    data : np.array, (n_channels, n_samps)
        Multidimentional timeseries array, expected dimentions (n_channels, n_samps)
    index_order : str, optional
        Indexing of the array in the memory i.e. C-like or F-like, by default 'C'. See `numpy.tobytes` for more
    convert_byte_ordering_to : str, optional
        Defines the change of byte ordering i.e. to little or big endian, By default 'I' i.e. no chage. See `numpy.dtype.newbyteorder` for more 

    Returns
    -------
    bytes
        Byte stream of the converted

    Notes
    ----
    - `index_order` and `convert_byte_ordering_to`, should be changed only to match the buffer type expected. Most probably you don't have use them.
        - E.g in @otb_array_to_buffer` is built such it resembles the otb bytestream (little endian, and F-like indexing)
    - index_order: Byte ordering, 'F' -> concat columnwise, 'C' concat row-wise
    - Haven't been checked for multidimentional arrays, dim>2
    """
    if convert_byte_ordering_to!='I': # 'I' = ignore
        dt = data.dtype.newbyteorder(convert_byte_ordering_to)        
    return data.tobytes(order=index_order)


def otb_buffer_to_array(buffer:  np.array, n_channels: int, n_samps:int, dtype=np.int16)-> np.array:
    """ Converts an OTB bytestream/buffer to data array (n_channels,  n_samps)

    Parameters
    ----------
    buffer : np.array
        An object that exposes the buffer interface (OTB like bytestream).
    n_channels : int
        Number of channels of the array, represented in the buffer.
    n_samps : int
        Number of samples (1-channel) of the array, represented in the buffer.
    dtype : _type_, optional
        Datatype of the 2d array, by default np.int16, for OTB. 

    Returns
    -------
    np.array
        2D array of data (n_channels, n_samps)

    Notes
    -----
    OTB like datastream: little endian, F-like memory layout (and probably 1 samp=16bit)
    """
    return buffer_to_array(buffer, n_channels, n_samps, dtype,  index_order='F', byte_ordering='little')

def otb_array_to_buffer(data: np.array)-> bytes:
    """ Conerts a 2D array to OTB like bytesteam/buffer

    Parameters
    ----------
    data : np.array
        2D array of data (n_channels, n_samps)

    Returns
    -------
    bytes
        OTB like bytestream of the input array

    Example
    ----
    >>> array = np.arange(256,  dtype=np.int16).reshape((64, 4))
    >>> buffer = otb_array_to_buffer(array)
    >>> array_2 = otb_buffer_to_array(buffer, array.shape[0], array.shape[1])
    >>> array == array_2
    """
    return array_to_buffer(data, index_order='F', convert_byte_ordering_to='little')


if __name__ == '__main__':
    filename = "C:/Wenlong Li/TUM/Master/第一学期/FP/Data/Hao/2023-09-06_17h50_Hao_ramp_30MVC_and_sin/X_Iocz20230906173031_03_Hao_ramp_30MVC_and_sin.otb+"
    data, channel_df = import_otb(filename)
    fs = channel_df['fsamp'][0]