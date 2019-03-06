# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    __init__.py
# @brief   protein analysis modules
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
import mdtraj as md
from itertools import combinations
import pandas as pd




def find_dssp_domain(dssp_data_original, type='H'):
    dssp_data = pd.DataFrame(dssp_data_original)

    all_type_elements = dssp_data[dssp_data == type]

    # for elem in dssp_data:
    #     elementus = dssp_data.iloc[:,elem]
    #     value = elementus.values[0]
    #     print(value)
    import numpy as np

    counter = 0
    temp_type_indexes = []
    type_list_dict = {}
    type_list_all = []
    for elem in all_type_elements:
        elementus = all_type_elements.iloc[:, elem]
        value = elementus.values[0]

        try:
            if value == type:
                temp_type_indexes.append(elem)
            elif value != type and elem > 0:
                # TODO very important
                # This part determines inital type domains
                if type == 'H':
                    if len(temp_type_indexes) > 3:
                        counter += 1
                        type_list_dict.update({counter: temp_type_indexes})
                        type_list_all.append(temp_type_indexes)
                elif type == 'E':
                    if len(temp_type_indexes) > 1:
                        counter += 1
                        type_list_dict.update({counter: temp_type_indexes})
                        type_list_all.append(temp_type_indexes)
                temp_type_indexes = []
        except:
            pass
            # if np.isnan(value) == True and elem>0:
            #     counter += 1
            #     h_list_dict.update({counter:temp_H_indexes})
            #     temp_H_indexes=[]


            # print(value)

    test = 1

    # Label Helixes

    # np.ceil(np.mean(h_list_dict[1]))

    # This part is finding continues aminoacid, if gap is less than 3
    # consider it continuous
    new_list = []
    merged_index = None
    for index in range(len(type_list_all)):
        if merged_index == index:
            continue
        try:
            list1 = type_list_all[index]
            list2 = type_list_all[index + 1]

            last_elem = list1[-1]
            first_elem = list2[0]

            if first_elem - last_elem <= 3:
                temp_list = list1 + list2
                merged_index = index + 1
                new_list.append(temp_list)
            else:
                new_list.append(list1)
        except:
            new_list.append(list1)

    test = 1

    type_list_dict_fixed = {}
    for index in range(len(new_list)):
        type_list_dict_fixed.update({index + 1: new_list[index]})


    test = 1

    return type_list_dict_fixed



def find_helixes(dssp_data_original):
    dssp_data = pd.DataFrame(dssp_data_original)

    all_helixes = dssp_data[dssp_data == 'H']
    # for elem in dssp_data:
    #     elementus = dssp_data.iloc[:,elem]
    #     value = elementus.values[0]
    #     print(value)
    import numpy as np

    counter = 0
    temp_H_indexes = []
    h_list_dict = {}
    h_list_all = []
    for elem in all_helixes:
        elementus = all_helixes.iloc[:, elem]
        value = elementus.values[0]

        try:
            if value == 'H':
                temp_H_indexes.append(elem)
            elif value != 'H' and elem > 0:
                # TODO very important
                # if len(temp_H_indexes)>0:
                if len(temp_H_indexes) > 3:
                    counter += 1
                    h_list_dict.update({counter: temp_H_indexes})
                    h_list_all.append(temp_H_indexes)
                temp_H_indexes = []
        except:
            pass
            # if np.isnan(value) == True and elem>0:
            #     counter += 1
            #     h_list_dict.update({counter:temp_H_indexes})
            #     temp_H_indexes=[]


            # print(value)

    test = 1

    # Label Helixes

    # np.ceil(np.mean(h_list_dict[1]))
    new_list = []
    merged_index = None
    for index in range(len(h_list_all)):
        if merged_index == index:
            continue
        try:
            list1 = h_list_all[index]
            list2 = h_list_all[index + 1]

            last_elem = list1[-1]
            first_elem = list2[0]

            if first_elem - last_elem <= 3:
                temp_list = list1 + list2
                merged_index = index + 1
                new_list.append(temp_list)
            else:
                new_list.append(list1)
        except:
            new_list.append(list1)

    h_list_dict_fixed = {}
    for index in range(len(new_list)):
        h_list_dict_fixed.update({index + 1: new_list[index]})


    return h_list_dict_fixed

    # if label_helix is True:
    #     for helix in h_list_dict_fixed:
    #         helix_num = int(helix)
    #         curr_data = h_list_dict_fixed[helix]
    #         residue = int(np.ceil(np.mean(curr_data)))
    #
    #         # curr_expression = '"α{0}"'.format(helix_num)
    #         curr_expression = '"H{0}"'.format(helix_num)
    #
    #         curr_label_selection = 'resi {0} and n. ca'.format(residue)
    #         # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
    #         pymol.cmd.label(selection=curr_label_selection, expression=curr_expression)
    #
    #         sleep(0.5)


def best_hummer_q(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used

    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`

    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """

    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers

    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i, j) for (i, j) in combinations(heavy, 2)
         if abs(native.topology.atom(i).residue.index - \
                native.topology.atom(j).residue.index) > 3])

    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    print("Number of native contacts", len(native_contacts))

    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q


# TODO example for multiprocessing
def calc_sasa(trajectory, parallel=True, parallel_way='multiprocessing', n_sphere_points=960):
    if parallel is False:
        sasa = md.shrake_rupley(trajectory, n_sphere_points=n_sphere_points)

        print(trajectory)
        print('sasa data shape', sasa.shape)

        total_sasa = sasa.sum(axis=1)
        print('total sasa shape ', total_sasa.shape)
    else:

        if parallel_way == 'ipyparallel':
            # TODO start by for single machine ipcluster start -n 4
            print('Parallel calculation Haha')
            # from IPython.parallel import Client

            from ipyparallel import Client

            c = Client()

            results = c[:].map(md.shrake_rupley, trajectory)
            sasa_list = results.get()

            sasa = np.vstack(sasa_list)
            # sasa_temp = [x[0] for x in sasa_list]
            # sasa = np.array(sasa_temp)
            # print('sasa shape ', sasa.shape)

            total_sasa = sasa.sum(axis=1)
            print('total sasa shape ', total_sasa.shape)
        else:
            import multiprocessing
            num_proc = multiprocessing.cpu_count()
            print('Your number of CPUs is ', num_proc)

            pool = multiprocessing.Pool(num_proc)

            results = pool.map(md.shrake_rupley, trajectory)
            sasa_list = results

            sasa = np.vstack(sasa_list)
            # sasa_temp = [x[0] for x in sasa_list]
            # sasa = np.array(sasa_temp)
            # print('sasa shape ', sasa.shape)

            total_sasa = sasa.sum(axis=1)
            print('total sasa shape ', total_sasa.shape)

    return sasa, total_sasa


def calc_shifts_nmr(trajectory, tool_to_use='sparta', pH=5.0, temperature=298.00):
    if tool_to_use == 'sparta':
        shifts = md.nmr.chemical_shifts_spartaplus(trajectory)
    elif tool_to_use == 'shift2x':
        shifts = md.nmr.chemical_shifts_shiftx2(trajectory, pH=pH, temperature=temperature)
    elif tool_to_use == 'ppm':
        print('This is turned problematic under Linux so it has been commented')
        # shifts = md.nmr.chemical_shifts_ppm(trajectory)
    return shifts


def autocorr(x):
    "Compute an autocorrelation with numpy"
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    return result / result[0]
