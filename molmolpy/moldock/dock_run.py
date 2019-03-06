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
import itertools

import hdbscan
import matplotlib
import matplotlib.cm as cm
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
import os
import sys
import pickle

import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from sklearn import mixture

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import run_tools

import pickle
import json
import subprocess as sub

from molmolpy.utils import helper as hlp

# matplotlib.style.use('ggplot')
sns.set(style="darkgrid")


class VinaRunObject(object):
    """
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


    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

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
                 receptor_file=None,
                 ligand_file=None,
                 folder_path='.',
                 num_samples_run=100,
                 exhaust_vars=None,
                 run_type='exhaustiveness',
                 molname='Unknown',
                 receptor_name='Unknown',
                 load_state_file=None):

        self.load_state_file = load_state_file

        if load_state_file is not None:
            self.load_state_data_json(self.load_state_file)
        else:
            print('VinaRunObject has been created')

            self.receptor_file = receptor_file
            self.ligand_file = ligand_file

            # If center and size has been given or not
            self.setup_box = False
            self.prep_exhaust_run = False
            self.prep_sample_run = False
            self.prep_ultra_run = False


            self.folder_exists = False

            print('Receptor file path\n')
            print(self.receptor_file)
            print('Ligand file path\n')
            print(self.ligand_file)

            self.samples_run = list(range(1, num_samples_run + 1))
            self.exhaustiveness = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

            # Running vina,whether it's for exhaustiveness or traditional run
            self.folder_path = folder_path

            self.command_run_list = []
            self.command_samples_run_list = []

            self.molecule_name = molname
            self.receptor_name = receptor_name

            self.state_data = {}

            self.state_data_samples = {}


            # original data before transformation


            # Add receptor name

    def prepare_exhaustiveness_run(self, exhaust_vars=None, run_type='exhaustiveness'):
        if self.setup_box is False:
            print('Please setup simulation box')
            sys.exit(0)

        self.run_type = run_type

        # Vina takes exhaustiveness parameter
        if exhaust_vars is None:
            self.exhaustiveness = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.exhaustiveness = exhaust_vars

        self.prep_exhaust_run = True
        self.run_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + self.run_type
        self.sim_folder_run = self.folder_path + os.sep + self.run_folder_name
        # Create folder don't forget



        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.directories = folder_utils.find_folder_in_path(self.folder_path, self.run_folder_name)
        print('TADA ', self.directories)

        self.json_state_file = self.sim_folder_run + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type + '.json'

        # This will hold information about run states


        if len(self.directories) == 0:
            print('Creating folder for vina run\n')
            print('Vina run type: {0}'.format(self.run_type))
            print(self.sim_folder_run)
            folder_utils.create_folder(self.sim_folder_run)
            self.folder_exists = True

            self.state_data.update({'receptorFile': self.receptor_file,
                                    'ligandFile': self.ligand_file,
                                    'exhaustivenessList': self.exhaustiveness,
                                    'samplesList': self.samples_run,
                                    'folderPath': self.folder_path,
                                    'runType': self.run_type,
                                    'molName': self.molecule_name,
                                    'receptorName': self.receptor_name,
                                    'simRunFolder': self.sim_folder_run,
                                    'directory': self.directories,
                                    'setup': self.setup_box,
                                    'folderCreated': self.folder_exists,
                                    'simStates': {}})

            self.prepVinaSim_exhaust()
            self.save_state_data_json()

            self.load_state_called = False


        else:
            self.load_state_file = self.json_state_file
            self.load_state_called = True
            self.load_state_data_json(self.load_state_file)

    def prepare_ultra_run(self, exhaust_vars=None, run_type='ultra'):
        if self.setup_box is False:
            print('Please setup simulation box')
            sys.exit(0)

        self.run_type = run_type

        # Vina takes exhaustiveness parameter
        if exhaust_vars is None:
            self.exhaustiveness = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        else:
            self.exhaustiveness = exhaust_vars

        self.num_samples = 10
        self.num_samples_list = list(range(1, 11))

        self.prep_ultra_run = True
        self.run_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + self.run_type
        self.sim_folder_run = self.folder_path + os.sep + self.run_folder_name
        # Create folder don't forget



        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.directories = folder_utils.find_folder_in_path(self.folder_path, self.run_folder_name)
        print('TADA ', self.directories)

        self.json_state_file = self.sim_folder_run + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type + '.json'

        # This will hold information about run states


        if len(self.directories) == 0:
            print('Creating folder for vina run\n')
            print('Vina run type: {0}'.format(self.run_type))
            print(self.sim_folder_run)
            folder_utils.create_folder(self.sim_folder_run)
            self.folder_exists = True

            self.state_data.update({'receptorFile': self.receptor_file,
                                    'ligandFile': self.ligand_file,
                                    'exhaustivenessList': self.exhaustiveness,
                                    'samplesList': self.num_samples_list,
                                    'folderPath': self.folder_path,
                                    'runType': self.run_type,
                                    'molName': self.molecule_name,
                                    'receptorName': self.receptor_name,
                                    'simRunFolder': self.sim_folder_run,
                                    'directory': self.directories,
                                    'setup': self.setup_box,
                                    'folderCreated': self.folder_exists,
                                    'simStates': {}})

            self.prepVinaSim_ultra()
            self.save_state_data_json()

            self.load_state_called = False


        else:
            self.load_state_file = self.json_state_file
            self.load_state_called = True
            self.load_state_data_json(self.load_state_file)

    def prepare_samples_collection_run(self, standard_exhaust=128,
                                       num_samples_run=100,
                                       run_type='samples_run'):

        if self.setup_box is False:
            print('Please setup simulation box')
            sys.exit(0)

        self.run_type_samples = run_type

        self.prep_samples_run = True

        self.samples_exhaust = standard_exhaust
        self.samples_run = list(range(1, num_samples_run + 1))

        self.run_folder_name_samples = self.receptor_name + '_' + self.molecule_name + '_' + self.run_type_samples
        self.sim_folder_run_samples = self.folder_path + os.sep + self.run_folder_name_samples
        # Create folder don't forget

        # Exhaustiveness for all samples


        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.directories_samples = folder_utils.find_folder_in_path(self.folder_path, self.run_folder_name_samples)
        print('TADA ', self.directories_samples)

        self.json_samples_state_file = self.sim_folder_run_samples + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type_samples + '.json'

        # This will hold information about run states


        if len(self.directories_samples) == 0:
            print('Creating folder for vina samples run\n')
            print('Vina run type: {0}'.format(self.run_type_samples))
            print(self.sim_folder_run_samples)
            folder_utils.create_folder(self.sim_folder_run_samples)
            self.folder_exists_samples = True

            self.state_data_samples.update({'receptorFile': self.receptor_file,
                                            'ligandFile': self.ligand_file,
                                            'exhaustivenessList': self.exhaustiveness,
                                            'samples_exhaust': self.samples_exhaust,
                                            'samplesList': self.samples_run,
                                            'folderPath': self.folder_path,
                                            'runType': self.run_type_samples,
                                            'molName': self.molecule_name,
                                            'receptorName': self.receptor_name,
                                            'simRunFolder': self.sim_folder_run_samples,
                                            'directory': self.directories_samples,
                                            'setup': self.setup_box,
                                            'folderCreated': self.folder_exists_samples,
                                            'simStates': {}})

            self.prepVinaSim_samples()
            self.save_state_data_json(filedata=self.state_data_samples, filename=self.json_samples_state_file)

            self.load_state_called_samples = False

            self.prep_sample_run = True

        else:
            self.load_state_file_samples = self.json_samples_state_file
            self.load_state_called_samples = True
            self.load_samples_state_data_json(self.load_state_file_samples)
            self.prep_sample_run = True

    def get_exhaust_run_folder_name(self):
        curr_folder = os.getcwd()
        return curr_folder + os.sep + self.run_folder_name

    def get_samples_run_folder_name(self):
        curr_folder = os.getcwd()
        print("Yippie yi kay", curr_folder)
        return curr_folder + os.sep + self.run_folder_name_samples

    def save_state_data_json(self, filedata=None, filename=None):
        '''
        :param filename: Saves state file
        :return:
        '''
        # import json
        # with open(filename, 'w') as outfile:
        #     json.dump(self.cluster_models, outfile)
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        # TODO create folder for run saving state run

        # filename = self.sim_folder_run + os.sep + self.receptor_name + '_' + self.molecule_name + '.json'
        if filename is None and filedata is None:
            filename = self.json_state_file
            filedata = self.state_data
        else:
            filedata = filedata
            filename = filename
        json.dump(filedata, open(filename, "w"), sort_keys=True, indent=4)

    # TODO should I add json saving of information or not?
    def load_samples_state_data_json(self, filename):
        '''

        :param filename: load json state data
        :return:
        '''
        # self.absolute_path = os.path.abspath(filename)
        self.load_state_called_samples = True

        print(os.path.abspath(__file__))
        self.state_data_samples = json.load(open(filename, "r"))

        # os.chdir('HSL_exhaustiveness')

        self.receptor_file = self.state_data_samples['receptorFile']
        self.ligand_file = self.state_data_samples['ligandFile']
        self.exhaustiveness = self.state_data_samples['exhaustivenessList']
        self.samples_run = self.state_data_samples['samplesList']
        self.folder_path = self.state_data_samples['folderPath']
        self.run_type = self.state_data_samples['runType']
        self.molecule_name = self.state_data_samples['molName']
        self.receptor_name = self.state_data_samples['receptorName']

        # TODO test
        self.samples_exhaust = self.state_data_samples['samples_exhaust']
        self.sim_folder_run_samples = self.state_data_samples['simRunFolder']  # .split('/')[-1]
        self.directories_samples = self.state_data_samples['directory']
        self.setup_box = self.state_data_samples['setup']
        self.folder_exists = self.state_data_samples['folderCreated']

        self.x_center = self.state_data_samples['boxSettings']['center_x']
        self.y_center = self.state_data_samples['boxSettings']['center_y']
        self.z_center = self.state_data_samples['boxSettings']['center_z']
        self.x_size = self.state_data_samples['boxSettings']['size_x']
        self.y_size = self.state_data_samples['boxSettings']['size_y']
        self.z_size = self.state_data_samples['boxSettings']['size_z']
        self.num_modes = self.state_data_samples['boxSettings']['numModes']

    # TODO should I add json saving of information or not?
    def load_state_data_json(self, filename):
        '''

        :param filename: load json state data
        :return:
        '''
        # self.absolute_path = os.path.abspath(filename)
        self.load_state_called = True

        print(os.path.abspath(__file__))
        self.state_data = json.load(open(filename, "r"))

        # os.chdir('HSL_exhaustiveness')

        self.receptor_file = self.state_data['receptorFile']
        self.ligand_file = self.state_data['ligandFile']
        self.exhaustiveness = self.state_data['exhaustivenessList']
        self.samples_run = self.state_data['samplesList']
        self.folder_path = self.state_data['folderPath']
        self.run_type = self.state_data['runType']
        self.molecule_name = self.state_data['molName']
        self.receptor_name = self.state_data['receptorName']

        # TODO test
        self.sim_folder_run = self.state_data['simRunFolder']  # .split('/')[-1]
        self.directories = self.state_data['directory']
        self.setup_box = self.state_data['setup']
        self.folder_exists = self.state_data['folderCreated']

        self.x_center = self.state_data['boxSettings']['center_x']
        self.y_center = self.state_data['boxSettings']['center_y']
        self.z_center = self.state_data['boxSettings']['center_z']
        self.x_size = self.state_data['boxSettings']['size_x']
        self.y_size = self.state_data['boxSettings']['size_y']
        self.z_size = self.state_data['boxSettings']['size_z']
        self.num_modes = self.state_data['boxSettings']['numModes']

    def set_up_Vina_Box(self, x_center,
                        y_center,
                        z_center,
                        x_size,
                        y_size,
                        z_size,
                        num_modes=1000):
        #
        # if self.load_state_called is True:
        #     return


        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center

        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        self.num_modes = num_modes

        self.setup_box = True

        self.state_data.update({'setup': self.setup_box,
                                'boxSettings': {'center_x': self.x_center,
                                                'center_y': self.y_center,
                                                'center_z': self.z_center,
                                                'size_x': self.x_size,
                                                'size_y': self.y_size,
                                                'size_z': self.z_size,
                                                'numModes': self.num_modes}
                                })

        self.state_data_samples = self.state_data.copy()

        # TODO this part needs to be thought out

    def hold_nSec(self, n):
        for i in range(1, n + 1):
            print(i)
            time.sleep(1)  # Delay for 1 sec
        print('Ok %s secs have pass' % (n))

    @hlp.timeit
    def prepVinaExhaustCommand(self, exhaust):
        try:
            if self.setup_box is not False:
                # print("Running Vina")
                # TODO need to think about seed
                self.save_run_name = 'vina_' + self.run_type + '_' + str(exhaust)
                command_to_run = "vina --receptor {0} " \
                                 "--ligand {1} " \
                                 "--center_x {2} " \
                                 "--center_y {3} " \
                                 "--center_z {4} " \
                                 "--size_x {5} " \
                                 "--size_y {6} " \
                                 "--size_z {7} " \
                                 "--exhaustiveness {8} " \
                                 "--num_modes {9} " \
                                 "--seed 10 " \
                                 "--log {10}.txt " \
                                 "--out {11}_out.pdbqt".format(self.receptor_file,
                                                               self.ligand_file,
                                                               self.x_center,
                                                               self.y_center,
                                                               self.z_center,
                                                               self.x_size,
                                                               self.y_size,
                                                               self.z_size,
                                                               exhaust,
                                                               self.num_modes,
                                                               self.save_run_name,
                                                               self.save_run_name)
                print(command_to_run)
                self.command_run_list.append(command_to_run)
                print("Launching new Sim")

                self.state_data['simStates'].update({str(exhaust): {'save_run_name': self.save_run_name,
                                                                    'commandRun': command_to_run,
                                                                    'runFinished': False}})
                # try:
                #     os.system(command_to_run)
                # except KeyboardInterrupt:
                #     # quit
                #     sys.exit()
                print("Vina run finished")
            else:
                print('Please setup vina box settings')
        except Exception as e:
            print("error in runSim: ", e)
            sys.exit(0)

    @hlp.timeit
    def prepVinaUltraCommand(self, sample_num, exhaust):
        try:
            if self.setup_box is not False:
                # print("Running Vina")
                # TODO need to think about seed
                self.save_run_name = 'vina_' + self.run_type + '_' + str(sample_num) + '_' + str(exhaust)
                command_to_run = "vina --receptor {0} " \
                                 "--ligand {1} " \
                                 "--center_x {2} " \
                                 "--center_y {3} " \
                                 "--center_z {4} " \
                                 "--size_x {5} " \
                                 "--size_y {6} " \
                                 "--size_z {7} " \
                                 "--exhaustiveness {8} " \
                                 "--num_modes {9} " \
                                 "--log {10}.txt " \
                                 "--out {11}_out.pdbqt".format(self.receptor_file,
                                                               self.ligand_file,
                                                               self.x_center,
                                                               self.y_center,
                                                               self.z_center,
                                                               self.x_size,
                                                               self.y_size,
                                                               self.z_size,
                                                               exhaust,
                                                               self.num_modes,
                                                               self.save_run_name,
                                                               self.save_run_name)
                print(command_to_run)
                self.command_run_list.append(command_to_run)
                print("Launching new Sim")


                try:
                    if len(self.state_data['simStates'][str(exhaust)]) > 0:
                        self.state_data['simStates'][str(exhaust)].update(
                            {str(sample_num):
                                 {'save_run_name': self.save_run_name,
                                  'commandRun': command_to_run,
                                  'runFinished': False}})
                except:
                    self.state_data['simStates'].update({str(exhaust):
                                                             {str(sample_num):
                                                                  {'save_run_name': self.save_run_name,
                                                                   'commandRun': command_to_run,
                                                                   'runFinished': False}}})

                test = 1
                # try:
                #     os.system(command_to_run)
                # except KeyboardInterrupt:
                #     # quit
                #     sys.exit()
                print("Vina run finished")
            else:
                print('Please setup vina box settings')
        except Exception as e:
            print("error in runSim: ", e)
            sys.exit(0)

    @hlp.timeit
    def prepVinaSampleCommand(self, sample_num):
        # try:
        if self.setup_box is not False:
            # print("Running Vina")
            # TODO need to think about seed
            self.save_run_name = 'vina_' + self.run_type_samples + '_' + str(sample_num)
            command_to_run = "vina --receptor {0} " \
                             "--ligand {1} " \
                             "--center_x {2} " \
                             "--center_y {3} " \
                             "--center_z {4} " \
                             "--size_x {5} " \
                             "--size_y {6} " \
                             "--size_z {7} " \
                             "--exhaustiveness {8} " \
                             "--num_modes {9} " \
                             "--seed 10 " \
                             "--log {10}.txt " \
                             "--out {11}_out.pdbqt".format(self.receptor_file,
                                                           self.ligand_file,
                                                           self.x_center,
                                                           self.y_center,
                                                           self.z_center,
                                                           self.x_size,
                                                           self.y_size,
                                                           self.z_size,
                                                           self.samples_exhaust,
                                                           self.num_modes,
                                                           self.save_run_name,
                                                           self.save_run_name)
            print(command_to_run)
            self.command_samples_run_list.append(command_to_run)
            print("Launching new Sim")

            self.state_data_samples['simStates'].update({str(sample_num): {'save_run_name': self.save_run_name,
                                                                           'commandRun': command_to_run,
                                                                           'runFinished': False}})
            # try:
            #     os.system(command_to_run)
            # except KeyboardInterrupt:
            #     # quit
            #     sys.exit()
            print("Vina sample run command prep finished")
        else:
            print('Please setup vina box settings')
            # except Exception as e:
            #     print("error in Sample runSim: ", e)
            #     sys.exit(0)

    @hlp.timeit
    def prepVinaSim_exhaust(self):
        for exhaust in self.exhaustiveness:
            self.prepVinaExhaustCommand(exhaust)
            print('Now continue :D')

        self.save_state_data_json()
        # finally:
        #     shutdown()

    def prepVinaSim_ultra(self):
        for exhaust in self.exhaustiveness:
            for sample_num in self.num_samples_list:
                self.prepVinaUltraCommand(sample_num, exhaust)
                print('Now continue :D')

        self.save_state_data_json()
        # finally:
        #     shutdown()

    @hlp.timeit
    def prepVinaSim_samples(self):
        for sample_num in self.samples_run:
            self.prepVinaSampleCommand(sample_num)
            print('Now continue :D')

        self.save_state_data_json(filedata=self.state_data_samples, filename=self.json_samples_state_file)
        # finally:
        #     shutdown()

    @hlp.timeit
    def runVinaSim_exhaust(self, waitTime=100):

        if self.prep_exhaust_run is True:
            for exhaust in self.exhaustiveness:
                self.runVina(exhaust, waitTime=waitTime)
        else:
            print("You have not called 'prepare_exhaustiveness_run' method")
            sys.exit(1)
            # So that CPU cools down

    @hlp.timeit
    def runVinaSim_samples(self, waitTime=100):

        if self.prep_sample_run is True:
            for sample_num in self.samples_run:
                self.runVina(sample_num, run_type='sample_run', waitTime=waitTime)
        else:
            print("You have not called 'prepare_samples_collection_run' method")
            sys.exit(1)
            # So that CPU cools down

    @hlp.timeit
    def runVinaSim_ultra(self, waitTime=100):

        if self.prep_ultra_run is True:
            for exhaust in self.exhaustiveness:
                for sample_num in self.num_samples_list:
                    self.runVinaUltra(sample_num,exhaust, run_type='ultra', waitTime=waitTime)
        else:
            print("You have not called 'prepare_samples_collection_run' method")
            sys.exit(1)
            # So that CPU cools down



    @hlp.timeit
    def runVinaUltra(self, key, exhaust,  run_type='exhaustiveness', waitTime=100):
        # Because json doesn't integers as key
        key = str(key)
        if self.setup_box is not False:
            # print("Running Vina")
            if run_type == 'exhaustiveness':
                command_to_run = self.state_data['simStates'][str(key)]['commandRun']
            elif run_type == 'ultra':
                command_to_run = self.state_data['simStates'][str(exhaust)][str(key)]['commandRun']
            else:
                command_to_run = self.state_data_samples['simStates'][str(key)]['commandRun']
            print(command_to_run)

            # TODO very careful with folder saving
            try:
                if run_type == 'exhaustiveness':
                    run_state = self.state_data['simStates'][str(key)]['runFinished']
                    run_folder_run = self.sim_folder_run
                elif run_type == 'ultra':
                    run_state = self.state_data['simStates'][str(exhaust)][str(key)]['runFinished']
                    run_folder_run = self.sim_folder_run
                else:
                    run_state = self.state_data_samples['simStates'][str(key)]['runFinished']
                    run_folder_run = self.sim_folder_run_samples

                if run_state is False:
                    print("Launching new Sim")
                    os.chdir(run_folder_run)

                    # curr_folder = os.getcwd()
                    # print("Yippie yi kay", curr_folder)
                    try:
                        # THIS ONE WORKS
                        results = run_tools.runCommandPopen(command_to_run)

                        # Prototype
                        # results = run_tools.runCommandPopen_thread(command_to_run)

                        if results == 0:

                            # TODO os.chdir not effective
                            os.chdir('..')
                            if run_type == 'exhaustiveness':
                                self.state_data['simStates'][str(key)]['runFinished'] = True
                                self.save_state_data_json()
                            elif run_type == 'ultra':
                                self.state_data['simStates'][str(exhaust)][str(key)]['runFinished'] = True
                                self.save_state_data_json()
                            else:
                                self.state_data_samples['simStates'][str(key)]['runFinished'] = True
                                self.save_state_data_json(filedata=self.state_data_samples,
                                                          filename=self.json_samples_state_file)
                            print("Vina Ultra run finished")

                            # Move Waiting here
                            self.hold_nSec(waitTime)
                            print('Now continue :D')
                            print('-*-' * 50)
                            # os.system(command_to_run)
                        else:
                            print('Run cancelled')
                            sys.exit()

                    except Exception:
                        print('Run cancelled')
                        sys.exit()


            except KeyboardInterrupt:
                # quit
                sys.exit()

        else:
            print('Please setup vina box settings')



    @hlp.timeit
    def runVina(self, key, run_type='exhaustiveness', waitTime=100):
        # Because json doesn't integers as key
        key = str(key)
        if self.setup_box is not False:
            # print("Running Vina")
            if run_type == 'exhaustiveness':
                command_to_run = self.state_data['simStates'][str(key)]['commandRun']
            else:
                command_to_run = self.state_data_samples['simStates'][str(key)]['commandRun']
            print(command_to_run)

            # TODO very careful with folder saving
            try:
                if run_type == 'exhaustiveness':
                    run_state = self.state_data['simStates'][str(key)]['runFinished']
                    run_folder_run = self.sim_folder_run
                else:
                    run_state = self.state_data_samples['simStates'][str(key)]['runFinished']
                    run_folder_run = self.sim_folder_run_samples

                if run_state is False:
                    print("Launching new Sim")
                    os.chdir(run_folder_run)

                    # curr_folder = os.getcwd()
                    # print("Yippie yi kay", curr_folder)
                    try:
                        # THIS ONE WORKS
                        results = run_tools.runCommandPopen(command_to_run)

                        # Prototype
                        # results = run_tools.runCommandPopen_thread(command_to_run)

                        if results == 0:

                            # TODO os.chdir not effective
                            os.chdir('..')
                            if run_type == 'exhaustiveness':
                                self.state_data['simStates'][str(key)]['runFinished'] = True
                                self.save_state_data_json()
                            else:
                                self.state_data_samples['simStates'][str(key)]['runFinished'] = True
                                self.save_state_data_json(filedata=self.state_data_samples,
                                                          filename=self.json_samples_state_file)
                            print("Vina run finished")

                            # Move Waiting here
                            self.hold_nSec(waitTime)
                            print('Now continue :D')
                            print('-*-' * 50)
                            # os.system(command_to_run)
                        else:
                            print('Run cancelled')
                            sys.exit()

                    except Exception:
                        print('Run cancelled')
                        sys.exit()


            except KeyboardInterrupt:
                # quit
                sys.exit()

        else:
            print('Please setup vina box settings')

    def get_molecule_name(self):
        return self.molecule_name

    def get_receptor_name(self):
        return self.receptor_name

    def set_molecule_name(self, mol_name):
        self.molecule_name = mol_name

    def set_receptor_name(self, receptor_name):
        self.receptor_name = receptor_name

    # This might need to get modified
    def find_sample_files(self, folder):
        try:
            VIP = []
            for dirname, dirnames, filenames in os.walk(folder):
                for i in filenames:
                    # print i
                    if 'out' in i:
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

    def save_pretty_info(self):
        pass

    def save_json_info(self):
        pass
