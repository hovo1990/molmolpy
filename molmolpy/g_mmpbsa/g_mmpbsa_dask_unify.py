# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    __init__.py
# @brief   G_MMPBSA DASK PROJECT
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
import itertools

import time

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
import multiprocessing

import mdtraj as md

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import folder_utils

import json

from molmolpy.utils import helper as hlp

# matplotlib.style.use('ggplot')
sns.set(style="darkgrid")

low_seed = 1
high_seed = 999999999

mgltools_utilities = '/home/john1990/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'


class GMMPBSA_unify_object(object):
    """
    Usage example


        >>> EPI_folder = '/media/Work/MEGA/Programming/StressHormones/dock_EPI'

        >>> EPI_samples = '/media/Work/MEGA/Programming/StressHormones/'
        >>>
        >>>
        >>> receptor_file = EPI_folder + os.sep + 'centroid_model_clust2.pdbqt'
        >>> ligand_file = EPI_folder + os.sep + 'EPI.pdbqt'
        >>> molname = 'EPI'
        >>> receptor_name = 'LasR'
        >>> run_type = 'vina_sample'
        >>>
        >>>
        >>>
        >>> receptor_file = EPI_folder + os.sep + 'centroid.pdb'
        >>> ligand_file = EPI_folder + os.sep + 'EPI.pdb'
        >>> molname = 'EPI'
        >>> receptor_name = 'LasR'
        >>>
        >>>
        >>> EPI_uber_dock = uber_docker.UberDockerObject(receptor_file, ligand_file, '.', molname=molname, receptor_name=receptor_name)
        >>>
        >>>
        >>> EPI_uber_dock.prepare_uber_dock_protocol()
        >>> EPI_uber_dock.run_uber_dock_protocol()

    Use together

        >>> self.prepare_uber_dock_protocol() for preparation
        >>> self.run_uber_dock_protocol()

    or seperately

        >>> EPI_uber_dock.calculate_max_radius_from_com()
        >>> EPI_uber_dock.calculate_cube_edges()
        >>> EPI_uber_dock.calculate_box_edges_from_com()
        >>>
        >>>
        >>> EPI_uber_dock.prepare_uber_docker()
        >>>
        >>>
        >>> #This is for rDock, and it works so comment this part for a while
        >>> EPI_uber_dock.prepare_rdock_settings()
        >>> EPI_uber_dock.generate_rdock_cavity()
        >>> # Prepare and run Dock programs
        >>> EPI_uber_dock.prep_rDock_dock_run_commands()
        >>> EPI_uber_dock.run_rDock_simulation(parallel=True,  waitTime=15)
        >>>
        >>> #This is for FlexAid
        >>> EPI_uber_dock.prepare_flexaid_settings()
        >>> EPI_uber_dock.process_flexaid_ligand()
        >>> EPI_uber_dock.get_flexaid_clefts()
        >>> EPI_uber_dock.flexaid_generate_ga_dat_parameters()
        >>> EPI_uber_dock.flexaid_generate_config_input()
        >>> EPI_uber_dock.prep_FlexAid_dock_run_commands()
        >>> EPI_uber_dock.run_FlexAid_simulation(parallel=True,  waitTime=15)
        >>>
        >>>
        >>> # This is for Autodock vina
        >>> EPI_uber_dock.set_up_Vina_Box()
        >>> EPI_uber_dock.prepare_Vina_run()
        >>> EPI_uber_dock.prepVinaSim_uberDock()
        >>> EPI_uber_dock.runVinaSim_uber()



    Molecule object loading of pdb and pbdqt file formats.
    Then converts to pandas dataframe.

    Create MoleculeObject by parsing pdb or pdbqt file.
    2 types of parsers can be used: 1.molmolpy 2. pybel
    Stores molecule information in pandas dataframe as well as numpy list.
    Read more in the :ref:`User Guide <MoleculeObject>`.
    Parameters
    ----------
    filename : str, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.


    >>> LasR_MOR_mmpbsa_calc = g_mmpbsa_dask.GMMPBSAObject(traj, topol_file, tpr_file, mdp_file, index_file, first_index, second_index, molname, receptor_name)
    >>>
    >>>
    >>>
    >>> LasR_MOR_mmpbsa_calc.prepare_g_mmpbsa_dask_protocol(client)
    >>>
    >>>
    >>> LasR_MOR_mmpbsa_calc.prepare_for_dask_cluster(parallel=True)
    >>> #
    >>> # LasR_MOR_mmpbsa_calc.run_dask_docking(client)


    Notes
    -----
    See examples/cluster/plot_dbscan.py for an example.
    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n).
    Sparse neighborhoods can be precomputed using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>`
    with ``mode='distance'``.
    References
    ----------
    """

    def __init__(self,
                folder,

                 molname='Unknown',
                 receptor_name='Unknown',
                 folder_path='.',
                 job_name='Unknown',
                 load_state_file=None):

        self.folder = folder
        apolar = self.find_sample_files(filename='apolar_', folder=self.folder)

        self.num_samples = list(range(len(apolar)))

        test=1

        # unified gmmpbsa xvg files
        self.unify_apolar_files()
        self.unify_polar_files()
        self.unify_energyMM_files()

        # unifies dat files
        self.unify_contrib_apol_files()
        self.unify_contrib_pol_files()
        self.unify_contrib_MM_files()

    def unify_apolar_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'apolar_{0}.xvg'.format(sample)



            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename,'r') as f:
                    for line in f:
                        #print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'apolar_final.xvg'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING APOLAR FILES \n')
        test = 1


    def unify_polar_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'polar_{0}.xvg'.format(sample)



            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename,'r') as f:
                    for line in f:
                        #print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'polar_final.xvg'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING POLAR FILES \n')
        test = 1

    def unify_energyMM_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'energy_MM_{0}.xvg'.format(sample)



            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename,'r') as f:
                    for line in f:
                        #print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'energy_MM_final.xvg'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING energy MM FILES \n')
        test = 1



    def unify_contrib_apol_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'contrib_apol_{0}.dat'.format(sample)



            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename,'r') as f:
                    for line in f:
                        #print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'contrib_apol_final.dat'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING contrib_apol FILES \n')
        test = 1




    def unify_contrib_pol_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'contrib_pol_{0}.dat'.format(sample)



            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename,'r') as f:
                    for line in f:
                        #print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'contrib_pol_final.dat'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING contrib_pol FILES \n')
        test = 1

    def unify_contrib_MM_files(self):
        full_string = ''

        for sample in self.num_samples:
            filename = self.folder + os.sep + 'contrib_MM_{0}.dat'.format(sample)

            if sample == 0:
                file_open = open(filename, 'r')
                temp = file_open.read()
                full_string += temp
            else:
                with open(filename, 'r') as f:
                    for line in f:
                        # print(line)
                        if '#' in line or '@' in line:
                            continue
                        else:
                            full_string += line

                test = 1

            file_open.close()
            test = 1

        filename = self.folder + os.sep + 'contrib_MM_final.dat'
        file_open = open(filename, 'w')
        file_open.write(full_string)
        file_open.close()

        print('FINISHED JOINING contrib_MM FILES \n')
        test = 1




    def find_sample_files(self, filename, folder):
        try:
            VIP = []
            for dirname, dirnames, filenames in os.walk(folder):
                for i in filenames:
                    # print i
                    if 'apolar_' in i and 'final' not in i:
                        VIP.append(i)
                        # This is not necessary since info is inside pdbqt file
                        # elif 'vina_sample_' in i:
                        #     VIP.append(i)
            return VIP
        except Exception as e:
            print("error in find_files: ", e)
            sys.exit(0)

    def find_sample_folders(self, folder_path='.', dir_name='vina_sample'):
        try:
            dir_names = []
            for dirname, dirnames, filenames in os.walk(folder_path):
                # print(dirname, '-')
                if dir_name in dirname:  #
                    # print(dir_name)
                    dir_names.append(dirname)
            # print sorted(dir_names)
            return sorted(dir_names)
        except Exception as e:
            print("Problem with finding folders : ", e)
            sys.exit(0)
