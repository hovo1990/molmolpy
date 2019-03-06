# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# @file    filter_items.py
# @brief   filter_items for parser directory
# @author  Hovakim Grabski
#
# <!--------------------------------------------------------------------------
# Copyright (c) 2016-2019,Hovakim Grabski.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the molmolpy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ------------------------------------------------------------------------ -->



import numpy as np
import scipy as sp
import pandas as pd
import difflib
from molmolpy.utils import helper as hlp



def filter_neighbour_frequency(frames_data_original, total_length = None):
    frames_data = frames_data_original.copy()
    test = 1
    res_seq_frequency_dict  = {}

    # results_dict = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}

    for frame in frames_data:
        curr_data = frames_data[frame]
        for res_seq in curr_data:
            #print(res_seq)
            if res_seq not in res_seq_frequency_dict.keys():
                res_seq_frequency_dict.update({res_seq:curr_data[res_seq]})
                res_seq_frequency_dict[res_seq].update({'frames':[frame]})
            elif res_seq in res_seq_frequency_dict.keys():
                res_seq_frequency_dict[res_seq]['freq'] += 1
                res_seq_frequency_dict[res_seq]['frames'].append(frame)

    frequency_dict = {}
    for res_seq in res_seq_frequency_dict:
        freq = res_seq_frequency_dict[res_seq]['freq']
        print(freq)
        frequency_dict.update({freq: res_seq_frequency_dict[res_seq]})
        frequency_dict[freq].update({'nameVIP':res_seq})

        if total_length is not None:
            percentage = (freq * 100) / total_length
            frequency_dict[freq].update({'percentage': percentage})

    freq_stuff = sorted(list(frequency_dict.keys()))
    print(freq_stuff[-10:-1])

    test = 1
    return res_seq_frequency_dict, frequency_dict, freq_stuff




def run_neighbour_analysis_parallel(neighbour_frame_index, topology, neighbours_data_frame):
    frame_neighbours_freq = {neighbour_frame_index: {}}
    for neighbours in neighbours_data_frame:
        # info = traj.topology.atom(neighbours)
        # info = traj_topology.atom(neighbours)
        info = topology.iloc[neighbours]

        res_name = info['resName']
        res_seq = info['resSeq']
        res_index = info['serial']

        name_all = res_name + str(res_seq)

        curr_frame_index = neighbour_frame_index
        if name_all not in list(frame_neighbours_freq.keys()):
            # print('yay')
            frame_neighbours_freq[neighbour_frame_index].update({name_all: {'freq': 1, 'resName': res_name, 'resSeq': res_seq,
                                                    'resIndex': res_index, 'frameIndex': curr_frame_index}})


    return frame_neighbours_freq


def run_neighbour_ligand_analysis_parallel(neighbour_frame_index, topology, neighbours_data_frame, ligand='QRC'):
    frame_neighbours_freq = {neighbour_frame_index: {}}
    for neighbours in neighbours_data_frame:
        # info = traj.topology.atom(neighbours)
        # info = traj_topology.atom(neighbours)
        info = topology.iloc[neighbours]

        res_name = info['resName']
        res_seq = info['resSeq']
        res_index = info['serial']
        atom_name = info['name']

        name_all = res_name + str(res_seq)

        curr_frame_index = neighbour_frame_index
        if res_name == ligand:
            test = 1
            frame_neighbours_freq[neighbour_frame_index].update(
                {atom_name: {'freq': 1, 'resName': res_name, 'resSeq': res_seq,
                            'resIndex': res_index, 'frameIndex': curr_frame_index, 'atomName':atom_name}})
            test = 1

        # curr_frame_index = neighbour_frame_index
        # if name_all not in list(frame_neighbours_freq.keys()):
        #     # print('yay')
        #     frame_neighbours_freq[neighbour_frame_index].update({name_all: {'freq': 1, 'resName': res_name, 'resSeq': res_seq,
        #                                             'resIndex': res_index, 'frameIndex': curr_frame_index}})


    return frame_neighbours_freq




@hlp.timeit
def filter_non_nan_blocks(dataframe, axis_name=None):
    dictionary_data = {}
    temp = []
    for i in dataframe[axis_name]:
        if not np.isnan(i):
            temp.append(i)
        else:
            dictionary_data.update({str(len(temp)):temp})
            temp = []
    if len(temp) > 0:
        dictionary_data.update({str(len(temp)): temp})
        temp = []
    test = 1
    return dictionary_data



@hlp.timeit
def filter_similar_lists(list_of_lists):
    data = np.copy(list_of_lists)
    sorted_idx = np.lexsort(data.T)
    sorted_data = data[sorted_idx, :]

    # Get unique row mask
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))

    # Get unique rows
    out = sorted_data[row_mask]

    return out


@hlp.timeit
def filter_similar_lists_slow(list_of_lists):
    filtered_list = []

    for array_list in list_of_lists:
        if len(filtered_list) == 0:
            filtered_list.append(array_list)
        else:
            for elem in filtered_list:
               eq = np.array_equal(elem, array_list)
               if eq is False:
                   filtered_list.append(array_list)

    return filtered_list




@hlp.timeit
def filter_similar_lists_difflib(list_of_lists):
    filtered_list = []

    for array_list in list_of_lists:
        if len(filtered_list) == 0:
            filtered_list.append(array_list)
        else:
            for elem in filtered_list:
                sm = difflib.SequenceMatcher(None, elem, array_list)
                ratio = sm.ratio()
                if ratio < 1.0:
                    filtered_list.append(array_list)

    return filtered_list

