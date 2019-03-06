# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# @file    pymol_tools.py
# @brief   pymol viz scripts
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

from re import *
import math
import time
import re

import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd

import pymol
from pymol.cgo import *
from math import *
from pymol import cmd

from molmolpy.utils import converters
from molmolpy.utils import plot_tools

def spectrumbar(*args, **kwargs):
    """
    Author Sean M. Law
    University of Michigan
    seanlaw_(at)_umich_dot_edu

    USAGE

    While in PyMOL

    run spectrumbar.py

    spectrumbar (RGB_Colors,radius=1.0,name=spectrumbar,head=(0.0,0.0,0.0),tail=(10.0,0.0,0.0),length=10.0, ends=square)

    Parameter     Preset         Type     Description
    RGB_Colors    [1.0,1.0,1.0]  N/A      RGB colors can be specified as a
                                          triplet RGB value or as PyMOL
                                          internal color name (i.e. red)
    radius        1.0            float    Radius of cylindrical spectrum bar
    name          spectrumbar    string   CGO object name for spectrum bar
    head          (0.0,0.0,0.0)  float    Starting coordinate for spectrum bar
    tail          (10.0,0.0,0.0) float    Ending coordinate for spectrum bar
    length        10.0           float    Length of spectrum bar
    ends          square         string   For rounded ends use ends=rounded

    Examples:

    spectrumbar red, green, blue
    spectrumbar 1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0

    The above two examples produce the same spectrumbar!

    spectrumbar radius=5.0
    spectrumbar length=20.0

    """

    rgb = [1.0, 1.0, 1.0]
    name = "spectrumbar"
    radius = 1.0
    ends = "square"
    x1 = 0
    y1 = 0
    z1 = 0
    x2 = 10
    y2 = 0
    z2 = 0
    num = re.compile('[0-9]')
    abc = re.compile('[a-z]')

    for key in kwargs:
        if (key == "radius"):
            radius = float(kwargs["radius"])
        elif (key == "name"):
            name = kwargs["name"]
        elif (key == "head"):
            head = kwargs["head"]
            # head = head.strip('" []()')
            # x1, y1, z1 = map(float, head.split(','))
            x1 = head[0]
            y1 = head[1]
            z1 = head[2]
        elif (key == "tail"):
            tail = kwargs["tail"]
            # test
            #tail = list(tail)
            # tail = tail.strip('" []()')
            # x2, y2, z2 = map(float, tail.split(','))
            x2 = tail[0]
            y2 = tail[1]
            z2 = tail[2]
        elif (key == "length"):
            # if (abc.search(kwargs["length"])):
            #     print("Error: The length must be a value")
            #     return
            # else:
            x2 = float(kwargs["length"]);
        elif (key == "ends"):
            ends = kwargs["ends"]
        elif (key != "_self"):
            print("Ignoring unknown option \"" + key + "\"")
        else:
            continue

    args = list(args)
    if (len(args) >= 1):
        rgb = []

    for arg in args[0]:
        if type(arg) is str:
            if (str(cmd.get_color_tuple(args[0])) != None):
                rgb.extend(cmd.get_color_tuple(arg))
                test = 1
            else:
                return
        else:
            rgb.extend(arg)
            test = 1


    test = 1

    # while (len(args) >= 1):
    #     # if (num.search(args[0]) and abc.search(args[0])):
    #     #     if (str(cmd.get_color_tuple(args[0])) != "None"):
    #     #         rgb.extend(cmd.get_color_tuple(args.pop(0)))
    #     #     else:
    #     #         return
    #     test = 1
    #     rgb.extend(cmd.get_color_tuple(args.pop(0)))
    #
    #     if type(args[0]) is str:
    #         if (str(cmd.get_color_tuple(args[0])) != None):
    #             rgb.extend(cmd.get_color_tuple(args.pop(0)))
    #         else:
    #             return
    #     elif (num.search(args[0])):
    #         rgb.extend([float(args.pop(0))])
    #     elif (abc.search(args[0])):
    #         if (str(cmd.get_color_tuple(args[0])) != "None"):
    #             rgb.extend(cmd.get_color_tuple(args.pop(0)))
    #         else:
    #             return
    #     else:
    #         print
    #         "Error: Unrecognized color format \"" + args[0] + "\""
    #         return

    if (len(rgb) % 3):
        print
        "Error: Missing RGB value"
        print
        "Please double check RGB values"
        return

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    if (len(rgb) == 3):
        rgb.extend([rgb[0]])
        rgb.extend([rgb[1]])
        rgb.extend([rgb[2]])
    t = 1.0 / (len(rgb) / 3.0 - 1)
    c = len(rgb) / 3 - 1
    s = 0
    bar = []

    while (s < c):
        if (len(rgb) > 0):
            r = rgb.pop(0)
            g = rgb.pop(0)
            b = rgb.pop(0)
        if (s == 0 and ends == "rounded"):
            bar.extend(
                [COLOR, float(r), float(g), float(b), SPHERE, x1 + (s * t) * dx, y1 + (s * t) * dy, z1 + (s * t) * dz,
                 radius])
        bar.extend([CYLINDER])
        bar.extend([x1 + (s * t) * dx, y1 + (s * t) * dy, z1 + (s * t) * dz])
        bar.extend([x1 + (s + 1) * t * dx, y1 + (s + 1) * t * dy, z1 + (s + 1) * t * dz])
        bar.extend([radius, float(r), float(g), float(b)])
        if (len(rgb) >= 3):
            bar.extend([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
        else:
            bar.extend([float(r), float(g), float(b)])
        if (s == c - 1 and ends == "rounded"):
            bar.extend([COLOR, float(r), float(g), float(b), SPHERE, x1 + (s + 1) * t * dx, y1 + (s + 1) * t * dy,
                        z1 + (s + 1) * t * dz, radius])
        s = s + 1

    cmd.delete(name)
    cmd.load_cgo(bar, name)

    return

# cmd.extend("spectrumbar", spectrumbar)





def generate_pymol_viz_for_thread( receptor, exhaust_data, save_name):

    from time import sleep

    pymol.finish_launching()

    pymol.cmd.reinitialize()

    # Set background color to white
    pymol.cmd.bg_color("white")

    receptor_file = receptor
    pymol.cmd.load(receptor_file, 'receptorFile')
    pymol.cmd.publication('receptorFile')

    pymol_objects = {}

    data_to_open = {}
    for exhaust_mol in exhaust_data:
        molecule = exhaust_data[exhaust_mol]
        filepath = molecule.file_path
        data_type = molecule.sample_info_type
        num = molecule.sample_info_num

        data_to_open.update({num: filepath})

    data_keys = list(data_to_open.keys())

    int_keys = sorted([int(x) for x in data_keys])

    for i in int_keys:
        filepath = data_to_open[str(i)]
        correct_file_pymol_name = 'exha' + '_' + str(i)
        pymol.cmd.load(filepath, correct_file_pymol_name)
        pymol.cmd.publication(correct_file_pymol_name)

        pymol_objects.update({str(i): {'topol': correct_file_pymol_name}})
        sleep(0.5)

    test = 1

    # in the future
    # pymol.cmd.cealign()
    # This works
    print('Finished Pymol for Exhaust Visualization  ---- >')
    save_state_name = save_name
    pymol.cmd.save(save_state_name)

    # pymol.cmd.quit()

    # sleep(2)

    # return pymol_objects


def generate_pymol_residue_energy_viz(centroid_pdb_file,
                                      dssp_data_original,
                                      residue_data,


                                   save_name, label_atom='C12',
                                   ligand_name='HSL',
                                   show_residue_labels=True,
                                      show_residue_energy=True,
                                      spectrum_show=False,
                                      label_helix=True):
    import pymol
    from time import sleep

    pymol.finish_launching()

    pymol.cmd.reinitialize()

    # Set background color to white
    pymol.cmd.bg_color("white")

    visual_file = centroid_pdb_file


    pymol.cmd.load(visual_file, 'visualFile')
    pymol.cmd.publication('visualFile')

    pymol_objects = {}



    #
    from molmolpy.utils import converters
    import seaborn as sns

    test = 1

    colors = sns.cubehelix_palette(n_colors=len(residue_data), dark=0.5, light=0.92, reverse=True)

    residue_color_data = converters.convert_seaborn_color_to_rgb(colors)


    # sns.plt.show()


    energy_list = []

    for i in range(len(residue_data)):
        residue_index = residue_data['index'].iloc[i]
        residue_num = residue_data['ResidueNum'].iloc[i]
        residue_name = residue_data['Residue'].iloc[i]
        residue_energy = residue_data['TotalEnergy'].iloc[i]
        residue_sd = residue_data['TotalEnergySD'].iloc[i]




        curr_selection = 'resi {0}'.format(residue_num)
        pymol.cmd.show(representation='sticks', selection=curr_selection)

        curr_color = 'residue_color_{0}'.format(i)
        pymol.cmd.set_color(curr_color, residue_color_data[i])
        pymol.cmd.color(curr_color, curr_selection)

        curr_label_selection = 'resi {0} and n. ca'.format(residue_num)
        # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
        curr_expression = '"{0}"'.format(residue_name)

        temp = "{0}\n{1}±{2}".format(residue_name, round(residue_energy, 2),  round(residue_sd, 2))
        energy_list.append(temp)

        if show_residue_energy is True:
            curr_expression = '"{0} {1}±{2}"'.format(residue_name, round(residue_energy, 2),
                                                     round(residue_sd, 2))


        # final_expresion = '{0}'.format(curr_expression)
        pymol.cmd.label(selection=curr_label_selection, expression=curr_expression)
        sleep(0.5)


    plot_tools.custom_palplot_vertical(colors, ylabel=energy_list)

    # Test

    # spectrumbar colors
    # spectrumbar(['red', 'blue','green','yellow','purple'], radius=1.0, name='spectrumbar', head=(0.0, 0.0, 0.0), tail=(10.0, 0.0, 0.0), length=10.0,
    #             ends='square')


    if spectrum_show is True:
        spectrumbar(colors, radius=1.0, name='spectrumbar', head=(0.0, 0.0, 0.0),
                    tail=(10.0, 0.0, 0.0), length=10.0,
                    ends='square')


    #label Helix

    helix_start = None
    helix_end = None
    helix_num = 0
    helix_dict = {}
    # find Helixes


    dssp_data = pd.DataFrame(dssp_data_original)

    all_helixes = dssp_data[dssp_data =='H']
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
        elementus = all_helixes.iloc[:,elem]
        value = elementus.values[0]

        try:
            if value =='H':
                temp_H_indexes.append(elem)
            elif value !='H' and elem>0:
                # TODO very important
                # if len(temp_H_indexes)>0:
                if len(temp_H_indexes)>3:
                    counter += 1
                    h_list_dict.update({counter:temp_H_indexes})
                    h_list_all.append(temp_H_indexes)
                temp_H_indexes=[]
        except:
            pass
            # if np.isnan(value) == True and elem>0:
            #     counter += 1
            #     h_list_dict.update({counter:temp_H_indexes})
            #     temp_H_indexes=[]


        #print(value)

    test = 1

    #Label Helixes

    # np.ceil(np.mean(h_list_dict[1]))
    new_list = []
    merged_index = None
    for index in range(len(h_list_all)):
        if merged_index == index:
            continue
        try:
            list1 = h_list_all[index]
            list2 = h_list_all[index+1]

            last_elem = list1[-1]
            first_elem = list2[0]

            if first_elem-last_elem<=3:
                temp_list = list1+list2
                merged_index = index+1
                new_list.append(temp_list)
            else:
                new_list.append(list1)
        except:
            new_list.append(list1)


    h_list_dict_fixed = {}
    for index in range(len(new_list)):
        h_list_dict_fixed.update({index+1:new_list[index]})



    if label_helix is True:
        for helix in h_list_dict_fixed:
            helix_num = int(helix)
            curr_data = h_list_dict_fixed[helix]
            residue = int(np.ceil(np.mean(curr_data)))

            # curr_expression = '"α{0}"'.format(helix_num)
            curr_expression = '"H{0}"'.format(helix_num)

            curr_label_selection = 'resi {0} and n. ca'.format(residue)
            # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
            pymol.cmd.label(selection=curr_label_selection, expression=curr_expression)

            sleep(0.5)


    #



        # colors = sns.cubehelix_palette(n_colors=clust_num, rot=.5, dark=0, light=0.85)
            #
            #
            # colors_rgb = converters.convert_seaborn_color_to_rgb(colors)
            #
            # colors_ = colors
            # cluster_labels = clusters_info[clust_num]['labels']
            # colors_data = converters.convert_to_colordata(cluster_labels, colors)
            # cluster_colors = colors_data




    test = 1
    # save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
    #                                                    'filename': filename,
    #                                                    'colors': colors[clust],
    #                                                    'rgbColors': rgb_colors[clust],
    #                                                    'currModels': cluster,
    #                                                    'key': clust}})





    test = 1

    # in the future
    # pymol.cmd.cealign()
    # This works
    print('Finished Pymol for pymol md viz Cluster Visualization  ---- >')
    save_state_name = save_name
    pymol.cmd.save(save_state_name)

    # pymol.cmd.quit()

    # sleep(1)





def generate_pymol_interaction_viz(centroid_data, dssp_data,filtered_data,
                                   save_name, label_atom='C12',
                                   ligand_name='HSL',
                                   percentage_on=True,
                                   percentage_criteria=90):
    import pymol
    from time import sleep

    pymol.finish_launching()

    pymol.cmd.reinitialize()

    # Set background color to white
    pymol.cmd.bg_color("white")

    visual_file = centroid_data['filename']


    pymol.cmd.load(visual_file, 'visualFile')
    pymol.cmd.publication('visualFile')

    pymol_objects = {}


    filtered_data = filtered_data
    freq_dict = filtered_data['freqDict']
    freq_stuff =filtered_data['freqStuff']


    # top_list = 15
    # top_data = freq_stuff[-top_list:-1]
    top_list = 15
    top_data = freq_stuff[:]

    ligand_name='HSL'
    residues_list = []
    name_list = []
    percentage_list =[]
    # Filter HOH and HSL
    for top in top_data:
        data = freq_dict[top]
        name_vip = data['nameVIP']
        #print(name_vip)

        if 'HOH' not in name_vip and ligand_name not in name_vip:
            #print(name_vip)
            residue = data['resSeq']

            percentage = data['percentage']


            residues_list.append(residue)
            name_list.append(name_vip)
            percentage_list.append(percentage)

    from molmolpy.utils import converters
    import seaborn as sns

    # colors = sns.cubehelix_palette(n_colors=len(residues_list), rot=.5, dark=0, light=0.85)

    # a lot better
    # colors = sns.cubehelix_palette(n_colors=len(residues_list), dark=0.5, light=0.92)
    colors = sns.color_palette(palette='Paired', n_colors=len(residues_list))
    colors_rgb = converters.convert_seaborn_color_to_rgb(colors)

    # Label aa residues
    for i in range(len(residues_list)):

        curr_percentage = percentage_list[i]

        if curr_percentage >= percentage_criteria:
            curr_expression = '"{0}-{1}%"'.format(name_list[i], round(curr_percentage,2))

            residue = residues_list[i]
            curr_selection = 'resi {0} and n. ca'.format(residue)


            curr_selection = 'resi {0}'.format(residue)
            pymol.cmd.show(representation='sticks', selection=curr_selection)


            curr_color = 'residue_color_{0}'.format(i)
            pymol.cmd.set_color(curr_color, colors_rgb[i])
            pymol.cmd.color(curr_color, curr_selection)

            curr_label_selection = 'resi {0} and n. ca'.format(residue)
            # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
            pymol.cmd.label(selection=curr_label_selection, expression=curr_expression)

            sleep(0.5)


    helix_start = None
    helix_end = None
    helix_num = 0
    helix_dict = {}
    # find Helixes

    all_helixes = dssp_data[dssp_data =='H']
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
        elementus = all_helixes.iloc[:,elem]
        value = elementus.values[0]

        try:
            if value =='H':
                temp_H_indexes.append(elem)
            elif value !='H' and elem>0:
                # TODO very important
                # if len(temp_H_indexes)>0:
                if len(temp_H_indexes)>3:
                    counter += 1
                    h_list_dict.update({counter:temp_H_indexes})
                    h_list_all.append(temp_H_indexes)
                temp_H_indexes=[]
        except:
            pass
            # if np.isnan(value) == True and elem>0:
            #     counter += 1
            #     h_list_dict.update({counter:temp_H_indexes})
            #     temp_H_indexes=[]


        #print(value)

    test = 1

    #Label Helixes

    # np.ceil(np.mean(h_list_dict[1]))
    new_list = []
    merged_index = None
    for index in range(len(h_list_all)):
        if merged_index == index:
            continue
        try:
            list1 = h_list_all[index]
            list2 = h_list_all[index+1]

            last_elem = list1[-1]
            first_elem = list2[0]

            if first_elem-last_elem<=3:
                temp_list = list1+list2
                merged_index = index+1
                new_list.append(temp_list)
            else:
                new_list.append(list1)
        except:
            new_list.append(list1)


    h_list_dict_fixed = {}
    for index in range(len(new_list)):
        h_list_dict_fixed.update({index+1:new_list[index]})


    for helix in h_list_dict_fixed:
        helix_num = int(helix)
        curr_data = h_list_dict_fixed[helix]
        residue = int(np.ceil(np.mean(curr_data)))

        # curr_expression = '"α{0}"'.format(helix_num)
        curr_expression = '"H{0}"'.format(helix_num)

        curr_label_selection = 'resi {0} and n. ca'.format(residue)
        # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
        pymol.cmd.label(selection=curr_label_selection, expression=curr_expression)

        sleep(0.5)






        # colors = sns.cubehelix_palette(n_colors=clust_num, rot=.5, dark=0, light=0.85)
            #
            #
            # colors_rgb = converters.convert_seaborn_color_to_rgb(colors)
            #
            # colors_ = colors
            # cluster_labels = clusters_info[clust_num]['labels']
            # colors_data = converters.convert_to_colordata(cluster_labels, colors)
            # cluster_colors = colors_data




    test = 1
    # save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
    #                                                    'filename': filename,
    #                                                    'colors': colors[clust],
    #                                                    'rgbColors': rgb_colors[clust],
    #                                                    'currModels': cluster,
    #                                                    'key': clust}})





    test = 1

    # in the future
    # pymol.cmd.cealign()
    # This works
    print('Finished Pymol for pymol md viz Cluster Visualization  ---- >')
    save_state_name = save_name
    pymol.cmd.save(save_state_name)

    # pymol.cmd.quit()

    # sleep(1)


def generate_pymol_viz_clust_exhaust_percent_for_thread( receptor, exhaust_data, save_name):
    import pymol
    from time import sleep

    pymol.finish_launching()

    pymol.cmd.reinitialize()

    # Set background color to white
    pymol.cmd.bg_color("white")

    receptor_file = receptor
    pymol.cmd.load(receptor_file, 'receptorFile')
    pymol.cmd.publication('receptorFile')

    pymol_objects = {}

    # save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
    #                                                    'filename': filename,
    #                                                    'colors': colors[clust],
    #                                                    'rgbColors': rgb_colors[clust],
    #                                                    'currModels': cluster,
    #                                                    'key': clust}})



    data_to_open = {}
    for exhaust_mol in exhaust_data:
        # molecule = exhaust_data[exhaust_mol]
        # filepath = molecule.file_path
        # data_type = molecule.sample_info_type
        # num = molecule.sample_info_num

        test = 1

        curr_data = exhaust_data[exhaust_mol]

        curr_index = exhaust_mol
        correct_file_pymol_name = 'exhaust_clust_{0}'.format(curr_index + 1)

        correct_topol_filepath = curr_data['relativePath']
        pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
        pymol.cmd.publication(correct_file_pymol_name)

        curr_color = 'exhaus_cluster_color_{0}'.format(curr_index + 1)
        pymol.cmd.set_color(curr_color, curr_data['colors'])
        pymol.cmd.color(curr_color, correct_file_pymol_name)

        # correct_file_pymol_simple_name = key + '_simple_{0}'.format(curr_index)
        # pymol.cmd.load(simplified_object, correct_file_pymol_simple_name)
        # pymol.cmd.show_as(representation='dots', selection=correct_file_pymol_simple_name)
        #
        # pymol.cmd.color(curr_color, correct_file_pymol_simple_name)
        #
        # pymol_objects.update({centroid_data_index: {'topol': correct_file_pymol_name,
        #                                             'simple': correct_file_pymol_simple_name}})
        sleep(0.5)

    test = 1

    # in the future
    # pymol.cmd.cealign()
    # This works
    print('Finished Pymol for Exhaust Cluster Visualization  ---- >')
    save_state_name = save_name
    pymol.cmd.save(save_state_name)

    # pymol.cmd.quit()

    # sleep(1)


def generate_exhaust_pymol_viz_thread( type='data'):
    '''This is to make sure that pymol methods run separately'''
    import threading, time

    # sample_files = obtain_samples()
    # samples_data = load_samples()


    print('Start of Pymol Exhaust show smethod --->  ')
    save_state_name = receptor_name + '_' + molecule_name + '_' + type + '_exaustiveness_pymolViz.pse'

    if type == 'data':
        generate_pymol_viz_for_thread(receptor_file_viz, samples_data, save_state_name)
    elif type == 'cluster_percentage':
        save_state_name = receptor_name + '_' + molecule_name + '_' + type + 'percentage_exaustiveness_pymolViz.pse'
        data_to_pass = save_extract_files_list['centroid']
        generate_pymol_viz_clust_exhaust_for_thread(receptor_file_viz,
                                                         data_to_pass, save_state_name, percentage=True)
    else:
        data_to_pass = save_extract_files_list['centroid']
        generate_pymol_viz_clust_exhaust_for_thread(receptor_file_viz, data_to_pass, save_state_name)

    time.sleep(5)

    # t = threading.Thread(target=generate_pymol_viz_for_thread,
    #                      args=(key, receptor_file_viz, full_data_mdtraj_analysis, save_state_name))
    #
    #
    #
    # t.start()
    # #
    # #
    # while t.is_alive():
    #     # time.sleep(0.01)
    #     pass
    # #     # print next iteration of ASCII spinner

    print('Finished Pymol method ---> verify yolo')
