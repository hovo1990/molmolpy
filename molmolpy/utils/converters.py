# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# @file    __init__.py
# @brief   init for parser directory
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

# from . import molecule_object

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md


def convert_mdtraj_atom_nums_to_resseq(topology, atom_numbers, atom_to_extract='CA'):
    resseq_list = []
    resseq_index  = []

    for i in list(range(len(atom_numbers))):
        temp = topology.atom(atom_numbers[i])
        temp_rest = temp.residue.resSeq
        name = temp.name
        if name == atom_to_extract:
            resseq_list.append(temp_rest)
            resseq_index.append(i)

    test =1
    return resseq_list, resseq_index


def convert_values_to_rgba(values, cmap='jet', type='seaborn'):
    color_rgba = []

    if type == 'plt':
        color_converter = plt.get_cmap(cmap)
    else:
        color_converter = cmap

    for i in values:
        temp = color_converter(i)
        color_rgba.append(temp)
    return color_rgba


def convert_to_colordata(cluster_labels, seaborn_colors):
    color_data = cluster_labels[:]
    # labels = cluster_labels
    #
    # unique_labels = list(set(color_data))
    # print('Unique labels ', unique_labels)
    #
    # for k in unique_labels:  # Need to modify WORKS
    #     # print('k is ',k)
    #     # k == -1 then it is an outlier
    #     if k != -1:
    #         cluster_data = []
    #         color_data[labels == k] = seaborn_colors[k]
    color_data_real = []
    for i in color_data:
        color_data_real.append(seaborn_colors[i])

    return color_data_real


def convert_seaborn_color_to_rgb(color_list):
    rgb_colors = []
    for color in color_list:
        full_list = []
        for value in color:
            value *= 255
            full_list.append(value)
        rgb_colors.append(full_list)
    return rgb_colors


def convert_pandas_to_numpy(pd_framework):
    nd_array = pd_framework.as_matrix()
    return nd_array


def convert_pandas_for_dbi_analysis(pandas_framework, cluster_labels):
    cluster_list = []
    X = pandas_framework
    labels = cluster_labels
    unique_labels = list(set(labels))
    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        # k == -1 then it is an outlier
        if k != -1:
            xyz = X[labels == k]
            nd_array = convert_pandas_to_numpy(xyz)
            cluster_list.append(nd_array)
    return cluster_list


def convert_data_to_pandas(data_x, data_y, x_axis_name='x', y_axis_name='y'):
    # converted_data = pd.DataFrame()

    data = {x_axis_name: data_x, y_axis_name: data_y}
    converted_data = pd.DataFrame(data)
    # print('shape is ', dock_df.shape)
    return converted_data


def convert_mdtraj_for_dbi_analysis(data, cluster_labels):
    cluster_list = []
    X = data
    labels = cluster_labels
    unique_labels = list(set(labels))
    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        # k == -1 then it is an outlier
        if k != -1:
            xyz = X[labels == k]
            nd_array = xyz
            cluster_list.append(nd_array)
    return cluster_list
