# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# @file    extra_tools.py
# @brief   Extra Tools
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

from collections import OrderedDict

import math
import numpy as np


# This part is for helper functions
def cleanVal(val):
    tempus = val.split(' ')
    new_list = []
    # print tempus
    for i in tempus:
        if i == '':
            pass
        else:
            new_list.append(i)
    return new_list


def extract_vina_score(text_output):
    list_output = text_output.split('\n')
    for temp in list_output:
        #print(temp)
        if 'VINA RESULT:' in temp:
            tempus = cleanVal(temp)
            return float(tempus[2])
        test = 1
    test = 1


def extract_flexaid_score(text_output):
    list_output = text_output.split('\n')
    for temp in list_output:
        #print(temp)
        if 'CF=' in temp:
            tempus = temp.split('=')
            return float(tempus[-1])
        test = 1
    test = 1


def extract_ledock_score(text_output):
    list_output = text_output.split('\n')
    for temp in list_output:
        #print(temp)
        if 'Score' in temp:
            tempus = temp.split('Score:')
            tempus1 = tempus[-1].split(' ')
            return float(tempus1[-2])
        test = 1
    test = 1

def extract_centroid_pybel_atoms(atoms_list):
    atom_coords = []
    for atom in atoms_list:
        atom_coord = atom.coords
        atom_coords.append(atom_coord)

    atom_coords_np = np.array(atom_coords)
    mean_x = np.mean(atom_coords_np[:,0])
    mean_y = np.mean(atom_coords_np[:,1])
    mean_z = np.mean(atom_coords_np[:,2])
    test = 1
    return mean_x, mean_y, mean_z



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def convert_coords_to_local(data, centre):
    '''
    Move from global coordinates to local

    :param data: vector of atoms
    :param centre: vector of center of mass
    :return:
    '''
    test = 1
    new_data =list(data[:])
    for i in range(len(new_data)):
        new_data[i] = round(float(new_data[i]) - centre[0],3)
        new_data[i] = round(float(new_data[i]) - centre[1],3)
        new_data[i] = round(float(new_data[i]) - centre[2],3)

    test =1
    # new_data =[tuple(i) for i in new_data]
    new_data = np.array(new_data)
    return new_data


def convert_coords_to_global(data, centre):
    '''
    Move from local coordinates to global
    :param data:
    :param centre:
    :return:
    '''
    new_data = list(data[:])
    for i in range(len(new_data)):
        new_data[i] = round(float(data[i]) + centre[0],3)
        new_data[i] = round(float(data[i]) + centre[1],3)
        new_data[i] = round(float(data[i]) + centre[2],3)
    # new_data =[tuple(i) for i in new_data]
    new_data = np.array(new_data)
    return new_data



def calc_euclidean_distance(x,y):
    '''
    Calculate euclidean distance between 2 atoms

    :param x: 3d coordinates 1
    :param y: 3d coordinates 2
    :return:
    '''

    results = (x[0]-y[0])**2 +(x[1]-y[1])**2 + (x[2]-y[2])**2
    distance = np.sqrt(results)

    return distance


def write_roman(num):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num > 0:
                roman_num(num)
            else:
                break

    return "".join([a for a in roman_num(num)])
