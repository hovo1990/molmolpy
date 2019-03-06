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
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
import os
import sys
import pickle

from collections import namedtuple

import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
from scipy.signal import argrelmin



import hdbscan

from sklearn.neighbors import NearestNeighbors

from sklearn import mixture

import multiprocessing
import fileinput

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import fix_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import extra_tools
from molmolpy.utils import fingerprint_tools

import os
import sys
import pickle
import math
import multiprocessing

import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from sklearn import mixture

import sympy as sym
import mdtraj as md
import pybel
import openbabel

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

from rdkit.Chem import MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

from molmolpy.moldock import rdock_run
from molmolpy.moldock import run_dock_tools
from molmolpy.moldock import prepare_molecules_vina

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import run_tools
from molmolpy.utils import extra_tools

import pickle
import json
import subprocess as sub

from molmolpy.utils import helper as hlp

from molmolpy.utils import helper as hlp

# from deepchem import dock

atom_types = {'Car': 'C',
              'O2': 'O',
              'O3': 'O',
              'H': 'H',
              'HO': 'H',
              'C2': 'C',
              'C3': 'C',
              'N3': 'N'

              }

# matplotlib.style.use('ggplot')
sns.set(style="darkgrid", context='paper')


class UberDockerAnalysisObject(object):
    """
    Usage example


            >>> LasR_HSL_uber_dock_analysis.load_rDock_output()
            >>> LasR_HSL_uber_dock_analysis.load_vina_output()
            >>> LasR_HSL_uber_dock_analysis.load_flexaid_output()
            >>>
            >>> LasR_HSL_uber_dock_analysis.prepare_analysis_framework()

            >>> nb_cust = 5
            >>> custom_eps_set = 2.0
            >>> LasR_DQC_uber_dock_analysis.test_plot_knn_dist_knee_centroids(nb=nb_cust, custom_ahline=custom_eps_set)
            >>> LasR_DQC_uber_dock_analysis.cluster_density_uber_centroids(custom_eps=custom_eps_set, minPoints=nb_cust)
            >>> LasR_DQC_uber_dock_analysis.cluster_density_uber_centroids_ensemble_plot()
            >>>
            >>>

            >>> LasR_DQC_uber_dock_analysis.plot_ensemble_energy(plot_type='violin')
            >>> LasR_DQC_uber_dock_analysis.plot_ensemble_energy(plot_type='box')


            >>>  this works
            >>>  LasR_DQC_uber_dock_analysis.export_ensemble_custom_cluster_models()
            >>>
            >>>
            >>>  # LasR_DQC_uber_dock_analysis.extract_centroids_auto_custom()
            >>>
            >>>  LasR_DQC_uber_dock_analysis.automorphism_ligand()
            >>>  LasR_DQC_uber_dock_analysis.fix_ensemble_topology_ultra_using_pybel()
            >>>  LasR_DQC_uber_dock_analysis.extract_pybel_fixed_centroids_auto_custom()



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
                 load_state_file=None):

        self.load_state_file = load_state_file

        self.load_state_data_json(self.load_state_file)

        test = 1

        self.save_extract_files_list = {}

        self.save_extract_files_list_fixed = {}

        self.docking_pre_pca_analysis_data = {}

        # original data before transformation

        # Add receptor name

    def prep_mdtraj_object(self):
        '''
        Prepare receptor mdtraj object

        get mdtraj topology and save as pandas dataframe

        Calculate pdb receptor center of mass


        :return:
        '''
        self.receptor_mdtraj = md.load_pdb(self.receptor_file)

        self.receptor_mdtraj_topology = self.receptor_mdtraj.topology
        self.receptor_mdtraj_topology_dataframe = self.receptor_mdtraj.topology.to_dataframe()

        self.center_of_mass_receptor = md.compute_center_of_mass(self.receptor_mdtraj)[0]

        self.x_center = math.ceil(self.center_of_mass_receptor[0] * 10)
        self.y_center = math.ceil(self.center_of_mass_receptor[1] * 10)
        self.z_center = math.ceil(self.center_of_mass_receptor[2] * 10)

        self.receptor_pybel = pybel.readfile("pdb", self.receptor_file).__next__()
        self.ligand_pybel = pybel.readfile("pdb", self.ligand_file).__next__()

        test = 1

    def calculate_max_radius_from_com(self):
        '''
        Calculate maximum radius from receptor center of mass

        Distance is nm
        :return:
        '''
        test = 1

        self.max_radius = 0
        self.max_radius_atom_index = 0

        for atom_index in range(len(self.receptor_mdtraj.xyz[0])):
            atom = self.receptor_mdtraj.xyz[0][atom_index]

            curr_distance = extra_tools.calc_euclidean_distance(self.center_of_mass_receptor, atom)

            if curr_distance >= self.max_radius:
                self.max_radius = curr_distance
                self.max_radius_atom_index = atom_index

            # print(atom)
            # print(atom_index)
            test = 1

        test = 1

        self.max_radius_angstrom = math.ceil(self.max_radius * 10)

    def calculate_box_edges_from_com(self):
        '''
        Calculate box size from com
        using angles

        Distance is nm
        :return:
        '''
        test = 1

        self.max_radius = 0
        self.max_radius_atom_index = 0

        self.x_max_dist = 0
        self.y_max_dist = 0
        self.z_max_dist = 0

        self.x_dim = 0
        self.y_dim = 0
        self.z_dim = 0

        self.min_x_angle = 180
        self.min_y_angle = 180
        self.min_z_angle = 180

        self.x_atom = None
        self.y_atom = None
        self.z_atom = None

        for atom_index in range(len(self.receptor_mdtraj.xyz[0])):
            atom = self.receptor_mdtraj.xyz[0][atom_index]

            unit_vec_x = np.array([1, 0, 0])
            unit_vec_y = np.array([0, 1, 0])
            unit_vec_z = np.array([0, 0, 1])

            curr_distance = extra_tools.calc_euclidean_distance(self.center_of_mass_receptor, atom)

            local_coordinates = extra_tools.convert_coords_to_local(atom, self.center_of_mass_receptor)
            local_x_coordinates = extra_tools.convert_coords_to_local(unit_vec_x, self.center_of_mass_receptor)
            local_y_coordinates = extra_tools.convert_coords_to_local(unit_vec_y, self.center_of_mass_receptor)
            local_z_coordinates = extra_tools.convert_coords_to_local(unit_vec_z, self.center_of_mass_receptor)

            v1 = atom

            # v1 = local_coordinates
            angle_x = extra_tools.angle_between(unit_vec_x, v1)
            angle_y = extra_tools.angle_between(unit_vec_y, v1)
            angle_z = extra_tools.angle_between(unit_vec_z, v1)

            # angle_x = extra_tools.angle_between(local_x_coordinates ,v1)
            # angle_y = extra_tools.angle_between(local_y_coordinates,v1)
            # angle_z = extra_tools.angle_between(local_z_coordinates,v1)

            # angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

            curr_angle_x = np.degrees(angle_x)
            curr_angle_y = np.degrees(angle_y)
            curr_angle_z = np.degrees(angle_z)

            # if curr_angle_x < self.min_x_angle and curr_distance > self.x_dim:
            #     self.x_dim = curr_distance
            #     self.min_x_angle = curr_angle_x
            #
            #
            # if curr_angle_y < self.min_y_angle and curr_distance > self.y_dim:
            #     self.y_dim = curr_distance
            #     self.min_y_angle = curr_angle_y
            #
            # if curr_angle_z < self.min_z_angle and curr_distance > self.z_dim:
            #     self.z_dim = curr_distance
            #     self.min_z_angle = curr_angle_z

            if curr_angle_x <= self.min_x_angle:
                self.x_dim = curr_distance
                self.min_x_angle = curr_angle_x
                self.x_atom = self.receptor_mdtraj.topology.atom(atom_index)

            if curr_angle_y <= self.min_y_angle:
                self.y_dim = curr_distance
                self.min_y_angle = curr_angle_y
                self.y_atom = self.receptor_mdtraj.topology.atom(atom_index)

            if curr_angle_z <= self.min_z_angle:
                self.z_dim = curr_distance
                self.min_z_angle = curr_angle_z
                self.z_atom = self.receptor_mdtraj.topology.atom(atom_index)

            # print(self.receptor_mdtraj.topology.atom(atom_index))
            # print(curr_angle_x,curr_angle_y,curr_angle_z)
            # print('-----------------------------------------')

            if curr_distance >= self.max_radius:
                self.max_radius = curr_distance
                self.max_radius_atom_index = atom_index

            # print(atom)
            # print(atom_index)
            test = 1

        test = 1

        self.max_radius_angstrom = math.ceil(self.max_radius * 10)

        final_x_dim = 2 * self.x_dim * 10
        final_y_dim = 2 * self.y_dim * 10
        final_z_dim = 2 * self.z_dim * 10

        self.final_x_dim = (2 * self.y_dim * 10) + 2
        self.final_y_dim = (2 * self.z_dim * 10) + 2
        self.final_z_dim = (2 * self.x_dim * 10) + 2

        print(self.center_of_mass_receptor * 10)
        print('x atom ', self.x_atom)
        print('y atom ', self.y_atom)
        print('z atom ', self.z_atom)
        print(final_x_dim, final_y_dim, final_z_dim)

    def calculate_cube_edges(self):
        '''
        Calculate box edges from radius which is in angstroms
        :return:
        '''
        # Prototype traditional
        self.cube_x = self.max_radius_angstrom / np.sqrt(3)
        self.cube_y = self.max_radius_angstrom / np.sqrt(3)

        self.cube_z = np.sqrt(self.max_radius_angstrom ** 2 - self.cube_x ** 2 - self.cube_y ** 2)

        test_calc = self.cube_x ** 2 + self.cube_y ** 2 + self.cube_z ** 2

        self.real_cube_x = self.cube_x * 3
        self.real_cube_y = self.cube_y * 3
        self.real_cube_z = self.cube_z * 3

        test = 1

        # Prototype sympy

    def get_uber_dock_run_folder_name(self):
        curr_folder = os.getcwd()
        return curr_folder + os.sep + self.run_folder_name

    def prepare_uber_dock_protocol(self):
        '''
        prepare uber dock protocol for rDock,FlexAid, Vina
        :return:
        '''
        self.calculate_max_radius_from_com()
        self.calculate_cube_edges()
        self.calculate_box_edges_from_com()

        self.prepare_uber_docker()

        # This is
        # for rDock, and it works so comment this part for a while
        self.prepare_rdock_settings()
        self.generate_rdock_cavity()
        # Prepare and run Dock programs
        self.prep_rDock_dock_run_commands()
        # EPI_uber_dock.run_rDock_simulation(parallel=True, waitTime=15)

        # This is
        # for FlexAid
        self.prepare_flexaid_settings()
        self.process_flexaid_ligand()
        self.get_flexaid_clefts()
        self.flexaid_generate_ga_dat_parameters()
        self.flexaid_generate_config_input()
        self.prep_FlexAid_dock_run_commands()
        # EPI_uber_dock.run_FlexAid_simulation(parallel=True, waitTime=15)

        # This is for Autodock vina
        self.set_up_Vina_Box()
        self.prepare_Vina_run()
        self.prepVinaSim_uberDock()
        # EPI_uber_dock.runVinaSim_uber()

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

        self.receptor_file_original = self.state_data['receptorFile']
        self.ligand_file_original = self.state_data['ligandFile']

        self.ligand_file = self.state_data['ligandFile']

        self.folder_path = self.state_data['folderPath']
        self.run_type = self.state_data['runType']

        self.molecule_name = self.state_data['molName']
        self.receptor_name = self.state_data['receptorName']

        # TODO test
        self.sim_folder_run = self.state_data['simRunFolder']  # .split('/')[-1]
        self.directories = self.state_data['directory']
        self.setup_box = self.state_data['setup']
        self.folder_exists = self.state_data['folderCreated']

        self.absolute_json_state_file = self.state_data['absoluteJsonStates']
        self.uber_dock_folder = self.state_data['uberDockRunFolder']
        self.json_state_file = self.state_data['jsonStates']

        test = 1

        # self.rdock_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + 'rDock'
        # self.rdock_absolute_folder_name = self.uber_dock_folder + os.sep + self.rdock_folder_name

        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        # self.directories = folder_utils.find_folder_in_path(self.uber_dock_folder, self.rdock_folder_name)
        # print('TADA ', self.directories)

        test = 1
        # This will hold information about run states
        # self.uber_dock_folder = self.get_uber_dock_run_folder_name()

        ########################################################################################
        # LeDock settings part
        self.ledock_data = self.state_data['dockSoftware']['LeDock']
        test = 1

        # Try to load initial rDock settings
        try:
            # self.rdock_absolute_folder_name = self.rdock_data['rDockAbsFolder']
            #
            # self.receptor_mol2 = self.rdock_data['rDockAbsFolder']
            # self.ligand_sd = self.rdock_data['rDockAbsFolder']
            #
            # self.absolute_receptor_mol2 = self.rdock_data['rDockAbsFolder']
            # self.absolute_ligand_sd = self.rdock_data['rDockAbsFolder']

            # self.rdock_folder_exists = self.rdock_data['rDockAbsFolder']

            self.receptor_ledock_pdb = self.ledock_data['receptor_pdb']
            self.ligand_ledock_mol2 = self.ledock_data['ligand_mol2']

            self.absolute_receptor_ledock_pdb = self.ledock_data['abs_receptor_pdb']
            self.absolute_ligand_ledock_mol2 = self.ledock_data['abs_ligand_mol2']
            self.ledock_folder_exists = self.ledock_data['LeDockFolderStatus']
            self.ledock_absolute_folder_name = self.ledock_data['LeDockAbsFolder']

            self.ledock_absolute_output_folder = self.ledock_absolute_folder_name

            self.ledock_folder_name = self.ledock_data['LeDockFolderName']

            self.lepro_pdb_file = self.ledock_data['lepro_pdb']

        except:
            print('LeDock is empty verify yolo')

        test = 1

        try:
            self.setup_ledock_pameters = self.ledock_data['setup_LeDock']
            self.ledock_num_samples = self.ledock_data['num_samples']
            self.ledock_input_info = self.ledock_data['LeDockInputInfo']
            self.param_ledock_template = self.ledock_data['paramFull']
        except:
            print('LeDock setting part is empty verify yolo')

        try:
            self.ledock_param_title = self.ledock_data['LeDock_params']['title']
            self.rdock_title = self.ledock_data['LeDock_params']['title']

            self.receptor_file_ledock = self.ledock_data['LeDock_params']['receptorFile']
            self.ledock_rmsd = self.ledock_data['LeDock_params']['LeDockRMSD']

            self.ledock_xmin = self.ledock_data['LeDock_params']['xmin']
            self.ledock_xmax = self.ledock_data['LeDock_params']['xmax']
            self.ledock_ymin = self.ledock_data['LeDock_params']['ymin']
            self.ledock_ymax = self.ledock_data['LeDock_params']['ymax']
            self.ledock_zmin = self.ledock_data['LeDock_params']['zmin']
            self.ledock_zmax = self.ledock_data['LeDock_params']['zmax']

        except:
            print('LeDock_params is empty verify yolo')

        ########################################################################################
        # rDock settings part

        self.rdock_data = self.state_data['dockSoftware']['rDock']

        # Try to load initial rDock settings
        try:
            # self.rdock_absolute_folder_name = self.rdock_data['rDockAbsFolder']
            #
            # self.receptor_mol2 = self.rdock_data['rDockAbsFolder']
            # self.ligand_sd = self.rdock_data['rDockAbsFolder']
            #
            # self.absolute_receptor_mol2 = self.rdock_data['rDockAbsFolder']
            # self.absolute_ligand_sd = self.rdock_data['rDockAbsFolder']

            # self.rdock_folder_exists = self.rdock_data['rDockAbsFolder']

            self.receptor_mol2 = self.rdock_data['receptor_mol2']
            self.ligand_sd = self.rdock_data['ligand_sd']
            self.absolute_receptor_mol2 = self.rdock_data['abs_receptor_mol2']
            self.absolute_ligand_sd = self.rdock_data['abs_ligand_sd']
            self.rdock_folder_exists = self.rdock_data['rDockFolderStatus']
            self.rdock_absolute_folder_name = self.rdock_data['rDockAbsFolder']
            self.rdock_folder_name = self.rdock_data['rDockFolderName']
        except:
            print('rDock is empty verify yolo')

        test = 1

        try:
            self.setup_r_dock_pameters = self.rdock_data['setup_rDock']
            self.rdock_parm_name = self.rdock_data['prmName']
            self.rdock_parm_name_abs = self.rdock_data['prmNameAbs']
            self.param_template = self.rdock_data['paramFull']
        except:
            print('rDock is empty verify yolo')

        try:
            self.rdock_param_title = self.rdock_data['rDock_params']['title']
            self.rdock_title = self.rdock_data['rDock_params']['title']

            self.receptor_file = self.rdock_data['rDock_params']['receptorFile']
            self.receptor_flex = self.rdock_data['rDock_params']['receptorFlex']
            self.x_center = self.rdock_data['rDock_params']['x_center']
            self.y_center = self.rdock_data['rDock_params']['y_center']
            self.z_center = self.rdock_data['rDock_params']['z_center']
            self.radius = self.rdock_data['rDock_params']['radius']
            self.small_sphere = self.rdock_data['rDock_params']['smallSphere']
            self.large_sphere = self.rdock_data['rDock_params']['largeSphere']
            self.max_cavities = self.rdock_data['rDock_params']['maxCavities']
            self.vol_incr = self.rdock_data['rDock_params']['volIncr']
            self.grid_step = self.rdock_data['rDock_params']['gridStep']
        except:
            print('rDock_params is empty verify yolo')

        try:
            self.rDock_sim_states = self.state_data['dockSoftware']['rDock']['simStates']
            self.rdock_samples = self.state_data['dockSoftware']['rDock']['rDockSample_list']
            print('No need to generate rDock commands')
            self.prep_rDock_run = True
        except:
            print('rDock_params simStates is empty verify yolo')

        test = 1

        ##########################################################################################
        # FlexAid Part load settings

        self.flexaid_data = self.state_data['dockSoftware']['FlexAid']

        # Try to load initial flexAid settings
        try:
            # self.flexaid_absolute_folder_name = self.flexaid_data['FlexAidAbsFolder']
            #
            # self.receptor_flexaid_pdb = self.flexaid_data['rDockAbsFolder']
            # self.ligand_sd = self.flexaid_data['rDockAbsFolder']
            #
            # self.absolute_receptor_mol2 =self.flexaid_data['rDockAbsFolder']
            # self.absolute_ligand_sd = self.flexaid_data['rDockAbsFolder']

            # self.flexaid_folder_exists = self.flexaid_data['FlexAidFolderStatus']
            self.receptor_flexaid_initials = self.flexaid_data['receptorInitials']
            self.ligand_flexaid_initials = self.flexaid_data['ligandInitials']

            self.receptor_flexaid_pdb = self.flexaid_data['receptor_flexaid_pdb']
            self.receptor_flexaid_mol2 = self.flexaid_data['receptor_flexaid_mol2']
            self.ligand_flexaid_pdb = self.flexaid_data['ligand_flexaid_pdb']
            self.ligand_flexaid_mol2 = self.flexaid_data['ligand_flexaid_mol2']
            # self.ligand_flexaid_mol2 = self.flexaid_data['ligand_flexaid_mol2']
            self.absolute_receptor_flexaid_pdb = self.flexaid_data['abs_receptor_flexaid_pdb']
            self.absolute_receptor_flexaid_mol2 = self.flexaid_data['abs_receptor_flexaid_mol2']
            self.absolute_ligand_flexaid_pdb = self.flexaid_data['abs_ligand_flexaid_pdb']
            self.absolute_ligand_flexaid_mol2 = self.flexaid_data['abs_ligand_flexaid_mol2']
            self.flexaid_folder_exists = self.flexaid_data['FlexAidFolderStatus']
            self.flexaid_absolute_folder_name = self.flexaid_data['FlexAidAbsFolder']
            self.flexaid_folder_name = self.flexaid_data['FlexAidFolderName']

            self.flexaid_absolute_binding_sites_folder = self.flexaid_data['BindingSitesFolder']
            self.flexaid_absolute_clefts_folder = self.flexaid_data['CleftsFolder']
            self.flexaid_absolute_input_folder = self.flexaid_data['FlexAidInputFolder']
            self.flexaid_absolute_output_folder = self.flexaid_data['FlexAidOutputFolder']
            self.flexaid_absolute_processed_files_folder = self.flexaid_data['ProcessedFilesFolder']
        except:
            print('FlexAid is empty verify yolo')

        # Load Flexaid process ligand variables
        try:
            # self.flexaid_folder_exists = self.flexaid_data['FlexAidFolderStatus']
            self.flexaid_ligand_processed = self.flexaid_data['processLigand']['ligand_processed']

            self.flexaid_atom_index = self.flexaid_data['processLigand']['atomIndex']
            self.flexaid_res_name = self.flexaid_data['processLigand']['resName']
            self.flexaid_res_chain = self.flexaid_data['processLigand']['resChain']
            self.flexaid_res_number = self.flexaid_data['processLigand']['resNumber']


        except:
            print('FlexAid Process ligand is empty verify yolo')

        ###############################################################################################

        # Vina part
        self.vina_data = self.state_data['dockSoftware']['Vina']

        # Try to load initial Vina settings
        try:

            self.receptor_vina_initials = self.vina_data['receptorInitials']
            self.ligand_vina_initials = self.vina_data['ligandInitials']

            self.receptor_vina_pdb = self.vina_data['receptor_vina_pdb']
            self.receptor_vina_mol = self.vina_data['receptor_vina_pdbqt']
            self.ligand_vina_pdb = self.vina_data['ligand_vina_pdb']
            self.ligand_vina_pdbqt = self.vina_data['ligand_vina_pdbqt']
            # self.ligand_flexaid_mol2 = self.flexaid_data['ligand_flexaid_mol2']
            self.absolute_receptor_vina_pdb = self.vina_data['abs_receptor_vina_pdb']
            self.absolute_receptor_vina_pdbqt = self.vina_data['abs_receptor_vina_pdbqt']
            self.absolute_ligand_vina_pdb = self.vina_data['abs_ligand_vina_pdb']
            self.absolute_ligand_vina_pdbqt = self.vina_data['abs_ligand_vina_pdbqt']
            self.vina_folder_exists = self.vina_data['VinaFolderStatus']
            self.vina_absolute_folder_name = self.vina_data['VinaAbsFolder']
            self.vina_folder_name = self.vina_data['VinaFolderName']
            self.vina_num_samples_list = self.vina_data['samplesList']

        except:
            print('FlexAid is empty verify yolo')

    def load_rDock_output(self):
        folder_to_search = self.rdock_absolute_folder_name

        # files = folder_utils.find_simple_file(folder_to_search, 'sd')

        files = folder_utils.find_files_in_folder_uberDocker(folder_to_search, 'sd', include=['rdock_uberDock_sample'],
                                                             exclude=['txt'])

        self.rDock_samples_pybel = {}

        for file in sorted(files):

            index = int(file.split('_')[-1].split('.')[0])

            test = 1

            self.rDock_samples_pybel.update({index: {}})

            load_file = folder_to_search + os.sep + file

            sample_molecules = pybel.readfile("sd", load_file)  # .__next__()

            curr_mol_index = 1

            while True:
                try:
                    # obj = next(my_gen)
                    mol = sample_molecules.__next__()

                    self.rDock_samples_pybel[index].update({curr_mol_index: mol})

                    curr_mol_index += 1
                except StopIteration:
                    break

            # mol1 = molecules.__next__()

            # mol1_atoms = mol1.atoms

            test = 1

        test = 1

    def load_vina_output(self, verbose=True):
        folder_to_search = self.vina_absolute_folder_name

        # files = folder_utils.find_simple_file(folder_to_search, 'sd')

        files = folder_utils.find_files_in_folder_uberDocker(folder_to_search, 'pdbqt',
                                                             include=['vina_uberDock', 'out'],
                                                             exclude=['txt'])

        self.vina_samples_pybel = {}

        for file in sorted(files):

            if verbose is True:
                print("Here is the bug ")
                print('File name ', file)

                print(' split ', file.split('_'))

            index = int(file.split('_')[-2])

            test = 1

            self.vina_samples_pybel.update({index: {}})

            load_file = folder_to_search + os.sep + file

            sample_molecules = pybel.readfile("pdbqt", load_file)  # .__next__()

            curr_mol_index = 1

            while True:
                try:
                    # obj = next(my_gen)
                    mol = sample_molecules.__next__()

                    self.vina_samples_pybel[index].update({curr_mol_index: mol})

                    curr_mol_index += 1

                    test = 1
                except StopIteration:
                    break

            # mol1 = molecules.__next__()

            # mol1_atoms = mol1.atoms
            # self.vina_samples_pybel[1024][1].atoms[0].coords

            # mol.data

            test = 1

        test = 1

    def load_flexaid_output(self):
        folder_to_search = self.flexaid_absolute_output_folder

        # files = folder_utils.find_simple_file(folder_to_search, 'sd')

        files = folder_utils.find_files_in_folder_uberDocker(folder_to_search, 'pdb',
                                                             include=['FlexAid_uberDock_sample'],
                                                             exclude=['INI.pdb', 'fixed.pdb', 'txt', 'cad', 'rrd',
                                                                      'res'])

        self.flexaid_samples_pybel = {}

        test = 1

        for file in sorted(files):

            load_file = folder_to_search + os.sep + file
            # NEED TO FIX FILE HERE

            # FIX FIX ERROR twice writes
            fixed_output_filename = fix_tools.fix_flexaid_output(file, load_file)

            test = 1

            index = int(fixed_output_filename.split('_')[-3])

            curr_mol_index = int(fixed_output_filename.split('_')[-2])

            test = 1

            if index not in list(self.flexaid_samples_pybel.keys()):
                self.flexaid_samples_pybel.update({index: {}})

            mol = pybel.readfile("pdb", fixed_output_filename).__next__()

            self.flexaid_samples_pybel[index].update({curr_mol_index: mol})

            # mol = sample_molecules.__next__()

            # mol1 = molecules.__next__()

            # mol1_atoms = mol1.atoms
            # self.vina_samples_pybel[1024][1].atoms[0].coords

            test = 1

        test = 1

    def load_ledock_output(self):
        # folder_to_search = self.flexaid_absolute_output_folder
        folder_to_search = self.ledock_absolute_output_folder

        # files = folder_utils.find_simple_file(folder_to_search, 'sd')

        files = folder_utils.find_files_in_folder_uberDocker(folder_to_search, 'pdb',
                                                             include=[self.molecule_name, '_sample_', '_dock'],
                                                             exclude=['dok', 'mol2', 'list', 'in', 'txt'])

        self.ledock_samples_pybel = {}

        test = 1

        for file in sorted(files):

            load_file = folder_to_search + os.sep + file

            test = 1
            # NEED TO FIX FILE HERE

            # FIX FIX ERROR twice writes
            # fixed_output_filename = fix_tools.fix_flexaid_output(file, load_file)
            #
            #
            # test = 1
            #
            #
            index = int(load_file.split('_')[-2])

            curr_mol_index = int(load_file.split('_')[-1].split('dock')[-1].split('.pdb')[0])

            test = 1

            if index not in list(self.ledock_samples_pybel.keys()):
                self.ledock_samples_pybel.update({index: {}})

            # mol = pybel.readfile("pdb", fixed_output_filename).__next__()
            mol = pybel.readfile("pdb", load_file).__next__()

            self.ledock_samples_pybel[index].update({curr_mol_index: mol})

            # mol = sample_molecules.__next__()

            # mol1 = molecules.__next__()

            # mol1_atoms = mol1.atoms
            # self.vina_samples_pybel[1024][1].atoms[0].coords

            test = 1

        test = 1

    def output_all_flexaid_output(self):
        output_all = self.flexaid_absolute_folder_name
        filename = output_all + os.sep + 'all_flexaid.pdb'

        output = pybel.Outputfile("pdb", filename, overwrite=True)

        for sample in self.flexaid_samples_pybel:
            for molecule in self.flexaid_samples_pybel[sample]:
                test = 1
                output.write(self.flexaid_samples_pybel[sample][molecule])

        output.close()


    def output_all_ledock_output(self):
        output_all = self.ledock_absolute_folder_name
        filename = output_all + os.sep + 'all_ledock.pdb'

        output = pybel.Outputfile("pdb", filename,  overwrite=True)

        for sample in self.ledock_samples_pybel:
            for molecule in self.ledock_samples_pybel[sample]:
                test = 1
                output.write(self.ledock_samples_pybel[sample][molecule])

        output.close()


    def output_all_rdock_output(self):
        output_all = self.rdock_absolute_folder_name
        filename = output_all + os.sep + 'all_rdock.pdb'

        output = pybel.Outputfile("pdb", filename,  overwrite=True)

        for sample in self.rDock_samples_pybel:
            for molecule in self.rDock_samples_pybel[sample]:
                test = 1
                output.write(self.rDock_samples_pybel[sample][molecule])

        output.close()

    def output_all_rdock_output(self):
        output_all = self.rdock_absolute_folder_name
        filename = output_all + os.sep + 'all_rdock.pdb'

        output = pybel.Outputfile("pdb", filename,  overwrite=True)

        for sample in self.rDock_samples_pybel:
            for molecule in self.rDock_samples_pybel[sample]:
                test = 1
                output.write(self.rDock_samples_pybel[sample][molecule])

        output.close()

    def output_all_vina_output(self):
        output_all = self.vina_absolute_folder_name
        filename = output_all + os.sep + 'all_vina.pdb'

        output = pybel.Outputfile("pdb", filename,  overwrite=True)

        for sample in self.vina_samples_pybel:
            for molecule in self.vina_samples_pybel[sample]:
                test = 1
                output.write(self.vina_samples_pybel[sample][molecule])

        output.close()

    def prepare_analysis_framework(self, load_data=['LeDock', 'rDock', 'FlexAid', 'Vina']):

        columns_custom = ['Program', 'Sample', 'Model', 'X', 'Y', 'Z', 'Score', 'Marker', 'MarkerColor']

        self.analysis_dataframe = pd.DataFrame(columns=columns_custom)

        self.molecule_database = {}

        test = 1

        model = 1

        # self.flexaid_samples_pybel
        if 'LeDock' in load_data:
            for sample in self.ledock_samples_pybel:
                for molecule in self.ledock_samples_pybel[sample]:
                    curr_mol = self.ledock_samples_pybel[sample][molecule]

                    curr_score_text = curr_mol.data['REMARK']

                    curr_score = extra_tools.extract_ledock_score(curr_score_text)

                    curr_atoms = curr_mol.atoms

                    mean_x, mean_y, mean_z = extra_tools.extract_centroid_pybel_atoms(curr_atoms)

                    temp_list = [['LeDock', sample, molecule, mean_x, mean_y, mean_z, curr_score]]
                    # temp_frame = pd.DataFrame(data=temp_list, columns=columns_custom)

                    # self.analysis_dataframe = self.analysis_dataframe.append(temp_frame)

                    self.analysis_dataframe.loc[model] = ['LeDock', sample, molecule, mean_x, mean_y, mean_z,
                                                          curr_score, '*', 'y']
                    model += 1

            test = 1

            self.molecule_database.update({'LeDock': self.ledock_samples_pybel})

        if 'rDock' in load_data:
            for sample in self.rDock_samples_pybel:
                for molecule in self.rDock_samples_pybel[sample]:
                    curr_mol = self.rDock_samples_pybel[sample][molecule]

                    curr_score = float(curr_mol.data['SCORE'])
                    curr_atoms = curr_mol.atoms

                    mean_x, mean_y, mean_z = extra_tools.extract_centroid_pybel_atoms(curr_atoms)

                    temp_list = [['rDock', sample, molecule, mean_x, mean_y, mean_z, curr_score]]
                    # temp_frame = pd.DataFrame(data=temp_list, columns=columns_custom)

                    # self.analysis_dataframe = self.analysis_dataframe.append(temp_frame)

                    self.analysis_dataframe.loc[model] = ['rDock', sample, molecule, mean_x, mean_y, mean_z, curr_score,
                                                          'v', 'r']
                    model += 1

                    test = 1

            self.molecule_database.update({'rDock': self.rDock_samples_pybel})

        test = 1

        # self.flexaid_samples_pybel
        if 'FlexAid' in load_data:
            for sample in self.flexaid_samples_pybel:
                for molecule in self.flexaid_samples_pybel[sample]:
                    curr_mol = self.flexaid_samples_pybel[sample][molecule]

                    curr_score_text = curr_mol.data['REMARK']

                    curr_score = extra_tools.extract_flexaid_score(curr_score_text)

                    curr_atoms = curr_mol.atoms

                    mean_x, mean_y, mean_z = extra_tools.extract_centroid_pybel_atoms(curr_atoms)

                    temp_list = [['rDock', sample, molecule, mean_x, mean_y, mean_z, curr_score]]
                    # temp_frame = pd.DataFrame(data=temp_list, columns=columns_custom)

                    # self.analysis_dataframe = self.analysis_dataframe.append(temp_frame)

                    self.analysis_dataframe.loc[model] = ['FlexAid', sample, molecule, mean_x, mean_y, mean_z,
                                                          curr_score, 's', 'g']
                    model += 1

            test = 1

            self.molecule_database.update({'FlexAid': self.flexaid_samples_pybel})

        # self.flexaid_samples_pybel
        if 'Vina' in load_data:
            for sample in self.vina_samples_pybel:
                for molecule in self.vina_samples_pybel[sample]:
                    curr_mol = self.vina_samples_pybel[sample][molecule]

                    curr_score_text = curr_mol.data['REMARK']

                    curr_score = extra_tools.extract_vina_score(curr_score_text)

                    curr_atoms = curr_mol.atoms

                    mean_x, mean_y, mean_z = extra_tools.extract_centroid_pybel_atoms(curr_atoms)

                    temp_list = [['rDock', sample, molecule, mean_x, mean_y, mean_z, curr_score]]
                    # temp_frame = pd.DataFrame(data=temp_list, columns=columns_custom)

                    # self.analysis_dataframe = self.analysis_dataframe.append(temp_frame)

                    self.analysis_dataframe.loc[model] = ['Vina', sample, molecule, mean_x, mean_y, mean_z, curr_score,
                                                          'o', 'b']
                    model += 1

            self.molecule_database.update({'Vina': self.vina_samples_pybel})

        # self.molecule_database.update({'rDock':self.rDock_samples_pybel, 'FlexAid':self.flexaid_samples_pybel,
        #                                'Vina':self.vina_samples_pybel})

        test = 1

    @hlp.timeit
    def pca_analysis_pre(self, scale_features=False, custom=None):

        if custom is None:
            df = self.data_for_analysis
        else:
            df = custom

        from sklearn import preprocessing

        if scale_features is True:
            df = preprocessing.scale(df)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=10)
        pca.fit(df)
        T_pre = pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T_pre)

        T.columns = ['component1', 'component2']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T, T_pre, pca


    @hlp.timeit
    def pca_analysis(self, scale_features=False, custom=None):

        if custom is None:
            df = self.data_for_analysis
        else:
            df = custom

        from sklearn import preprocessing

        if scale_features is True:
            df = preprocessing.scale(df)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=10)
        pca.fit(df)
        T_pre= pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T_pre)

        T.columns = ['component1', 'component2']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T

    def collect_cluster_info_uber(self, clusters_info, clust_num, original_data):
        data = clusters_info[clust_num]
        print(data)

        labels = data['labels']
        # Make more flexible whether pca_data or not
        # pca_data = self.pca_data
        # original_data = self.analysis_reshape_structure__  # self.pca_data

        cluster_list = {}
        unique_labels = list(set(labels))
        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:

                selected_mols_list = original_data[labels == k]
                test = self.molecule_database

                cluster_data = []
                for index, row in selected_mols_list.iterrows():
                    # print(i)
                    program = row['Program']
                    sample = row['Sample']
                    model = row['Model']

                    curr_model = self.molecule_database[program][sample][model]
                    test = 1
                    cluster_data.append(curr_model)
                    # print(xyz.describe())
                cluster_list.update({k: cluster_data})
        # print(cluster_list)
        return cluster_list

    @hlp.timeit
    def test_plot_knn_dist_knee_centroids(self, custom_data=None, nb=5, custom_ahline=3):

        #         self.data_for_analysis_all

        test = 1
        # print(self.concatenated_analysis__[int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__)+ 1])

        # TODO CENTROID ANALYSIS

        # centroid_indexes = self.concatenated_analysis__[
        #     self.concatenated_analysis__['Type'] == 'centroid'].index.tolist()
        #
        # # temp_centroid = self.pca_data_all[centroid_indexes[0]:centroid_indexes[-1] + 1]
        #
        # data_centroid = self.concatenated_analysis__[
        #                 int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # self.data_centroid = data_centroid
        #
        # # self.concatenated_analysis__[self.data_cols]
        #
        # temp_centroid_pca = self.pca_data_all[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid_pre = self.concatenated_analysis__[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid = temp_centroid_pre[self.data_cols]

        # self.temp_centroid = temp_centroid

        test = 1
        # self.range_n_clusters = list(range(2, 10))

        # self.pca_data_all = self.pca_analysis(custom=self.analysis_dataframe[['X', 'Y', 'Z']])

        # temp_centroid_pca = self.pca_data_all

        if custom_data is None:
            temp_centroid = self.analysis_dataframe[['X', 'Y', 'Z']]
        else:
            temp_centroid = custom_data[['X', 'Y', 'Z']]

        # self.temp_centroid = temp_centroid

        # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        nbrs = NearestNeighbors(n_neighbors=nb).fit(temp_centroid)
        distances, indices = nbrs.kneighbors(temp_centroid)

        distances_sorted = np.sort(distances[:, -1])[::-1]
        test = 1

        pca_data_all = self.pca_analysis(custom=self.analysis_dataframe[['X', 'Y', 'Z']])
        nbrs_pca = NearestNeighbors(n_neighbors=nb).fit(pca_data_all)
        distances_pca, indices_pca = nbrs_pca.kneighbors(pca_data_all)

        distances_sorted_pca = np.sort(distances_pca[:, -1])[::-1]

        sns.set(font_scale=1.5)
        plt.figure(figsize=plot_tools.cm2inch(17.7, 10))
        plt.xlabel('Points (sample) sorted by distance')
        plt.ylabel('{0}-NN distance'.format(nb))
        plt.title('kNN distance plot')
        line1 = plt.plot(range(len(distances_sorted)), distances_sorted, linewidth=2.5, label='Original Data')
        line2 = plt.plot(range(len(distances_sorted_pca)), distances_sorted_pca, 'g--', linewidth=2.5,
                         label='PCA processed')

        plt.axhline(y=custom_ahline, color='b', linestyle='-')
        plt.legend()
        plt.tight_layout()

        if custom_data is None:
            save_name = 'knnDistPlot_{0}.png'.format(self.molecule_name)
        else:
            save_name = 'knnDistPlot_custom_{0}.png'.format(self.molecule_name)

        plt.savefig(save_name, dpi=1200)

        angle = np.rad2deg(np.arctan2(distances_sorted[-1] - distances_sorted[0],
                                      range(len(distances_sorted))[-1] - range(len(distances_sorted))[0]))

        test = 1

##########################################################################################################################
    # DOKCING VARIANCE

    @hlp.timeit
    def docking_COM_pca_cum_variance_analysis(self, show_plot=False, custom_dpi=1200,
                                     percentage=87, number_of_components=20):

        print('PCA Cumulative Variance analysis has been called for selection:{0}\n'.format('Docking Poses of Ligand'))
        print('-------------------------------\n')

        sns.set(style="ticks", context='paper')
        # fig = plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        sns.set(font_scale=1)

        data, data_pre, pca1 = self.pca_analysis_pre(custom=self.analysis_dataframe[['X', 'Y', 'Z']])

        # The amount of variance that each PC explains
        var = pca1.explained_variance_ratio_
        print('Explained variance ratio: ', var)

        self.docking_pre_pca_analysis_data.update({'COM': {'varExplainedRatio': pca1.explained_variance_ratio_,
                                                               'varExplained': pca1.explained_variance_,
                                                               'mean': pca1.mean_,
                                                               }
                                              })

        # Cumulative Variance explains
        var1 = np.cumsum(np.round(pca1.explained_variance_ratio_, decimals=4) * 100)

        print("Cumulative Variance explains ", var1)

        # plt.plot(var)
        plt.plot(var1)
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Proportion of Variance Explained")

        fig.savefig( 'Docking_COM_PCA_cumsum_analysis_' + '.png',
                    dpi=custom_dpi,
                    bbox_inches='tight')

        if show_plot is True:
            plt.show()

        print("PCA transformation finished successfully")
        print('-----------------------------------\n')



    @hlp.timeit
    def plot_pca_proportion_of_variance(self,  custom_dpi=1200):
        # np.round(pca1.explained_variance_ratio_, decimals=4) * 100

        # data, data_pre, pca = self.pca_analysis(custom=self.analysis_dataframe[['X', 'Y', 'Z']])

        # self.docking_pre_pca_analysis_data.update({'COM': {'varExplainedRatio': pca1.explained_variance_ratio_,
        #                                                        'varExplained': pca1.explained_variance_,
        #                                                        'mean': pca1.mean_,
        #                                                        }
        #                                       })


        explained_variance_ratio_ = self.docking_pre_pca_analysis_data['COM']['varExplainedRatio']

        prep_data = np.round(explained_variance_ratio_, decimals=4) * 100

        eigen_ranks = list(range(1, len(prep_data) + 1))

        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        plt.plot(eigen_ranks, prep_data, color='r')

        # plt.legend(loc="best", prop={'size': 8})
        xlabel = 'Eigenvalue Rank'
        ylabel = 'Proportion of Variance (%)'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # No title
        # plt.title(title)

        # remove part of ticks
        sns.despine()
        fig.tight_layout()
        fig.savefig('Docking_COM_ProportionOfVariance_' +'.png', dpi=custom_dpi,
                    bbox_inches='tight')

        print('RMSD plot created')
        print('-----------------------------------\n')





    @hlp.timeit
    def cluster_density_uber_centroids(self, analysis_type='PCA', custom_eps=0.5, minPoints=2,
                                       check_data=['LeDock', 'rDock', 'FlexAid', 'Vina'],
                                       trasparent_alpha=False, down_region=30
                                       ):

        #         self.data_for_analysis_all

        test = 1
        # print(self.concatenated_analysis__[int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__)+ 1])

        # TODO CENTROID ANALYSIS

        # centroid_indexes = self.concatenated_analysis__[
        #     self.concatenated_analysis__['Type'] == 'centroid'].index.tolist()
        #
        # # temp_centroid = self.pca_data_all[centroid_indexes[0]:centroid_indexes[-1] + 1]
        #
        # data_centroid = self.concatenated_analysis__[
        #                 int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # self.data_centroid = data_centroid
        #
        # # self.concatenated_analysis__[self.data_cols]
        #
        # temp_centroid_pca = self.pca_data_all[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid_pre = self.concatenated_analysis__[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid = temp_centroid_pre[self.data_cols]

        # self.temp_centroid = temp_centroid

        test = 1
        self.range_n_clusters = list(range(2, 10))

        self.pca_data_all = self.pca_analysis(custom=self.analysis_dataframe[['X', 'Y', 'Z']])

        temp_centroid_pca = self.pca_data_all

        temp_centroid = self.analysis_dataframe[['X', 'Y', 'Z']]
        self.temp_centroid = temp_centroid

        if analysis_type == 'PCA':
            self.X_dbscan = self.pca_data_all.values
        else:
            self.X_dbscan = temp_centroid.values

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler

        # USE DBSCAN
        self.dbscan_analysis = DBSCAN(eps=custom_eps, min_samples=minPoints).fit(self.X_dbscan)

        # Use HDBScan
        # self.dbscan_analysis = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(self.X_dbscan)



        self.core_samples_mask = np.zeros_like(self.dbscan_analysis.labels_, dtype=bool)
        self.core_samples_mask[self.dbscan_analysis.core_sample_indices_] = True
        self.labels_dbscan = self.dbscan_analysis.labels_

        # Number of clusters in labels, ignoring noise if present.
        self.dbscan_n_clusters_ = len(set(self.labels_dbscan)) - (1 if -1 in self.labels_dbscan else 0)
        print('Number of cluster is ', self.labels_dbscan)

        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        self.dbscan_unique_labels = set(self.labels_dbscan)
        self.dbscan_colors = [plt.cm.Spectral(each)
                              for each in np.linspace(0, 1, len(self.dbscan_unique_labels))]

        for k, col in zip(self.dbscan_unique_labels, self.dbscan_colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_dbscan == k)

            xy = self.X_dbscan[class_member_mask & self.core_samples_mask]

            hold_info = xy
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = self.X_dbscan[class_member_mask & ~self.core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

            if k != -1:
                test = 1
                center = calculate_cluster_center_only_docking(hold_info)
                # print('Hello center')

                down = down_region
                object_centroid_circles = ax.scatter(center[0], center[1],
                                                     marker='o', c="white", alpha=1, s=200)
                object_centroid_circles.set_zorder(60)

                c = center
                roman_number = extra_tools.write_roman(int(k + 1))
                object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=150,
                                                          c='b')
                object_centroid_circles_text.set_zorder(61)

        plt.tight_layout()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Estimated number of clusters: %d' % self.dbscan_n_clusters_)
        fig.savefig('DBSCAN_plot_clusters_{0}.png'.format(self.molecule_name), dpi=1200)

        test = 1

        self.uber_dock_ensemble = []
        self.uber_dock_ensemble_data = {}

        for k, col in zip(self.dbscan_unique_labels, self.dbscan_colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_dbscan == k)

            curr_data = self.analysis_dataframe[class_member_mask & self.core_samples_mask]
            test = 1

            curr_programs = curr_data['Program']

            curr_prog_values = curr_programs.values

            test = 1

            # this works
            if set(curr_prog_values) == set(check_data):
                self.uber_dock_ensemble.append(k)
                self.uber_dock_ensemble_data.update({k:curr_data})

        cluster_list = {}
        for k in self.uber_dock_ensemble:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:

                selected_mols_list = self.analysis_dataframe[self.labels_dbscan == k]
                test = self.molecule_database

                cluster_data = []
                for index, row in selected_mols_list.iterrows():
                    # print(i)
                    program = row['Program']
                    sample = row['Sample']
                    model = row['Model']

                    curr_model = self.molecule_database[program][sample][model]
                    test = 1
                    cluster_data.append(curr_model)
                    # print(xyz.describe())
                cluster_list.update({k: cluster_data})
        # print(cluster_list)

        self.ensemble_models = cluster_list

        test = 1

    @hlp.timeit
    def cluster_density_uber_centroids_deeper(self, analysis_type='PCA', custom_eps=0.5, minPoints=2,
                                       check_data=['LeDock', 'rDock', 'FlexAid', 'Vina'],
                                       trasparent_alpha=False, down_region=30
                                       ):

        #         self.data_for_analysis_all

        test = 1

        key = list(self.uber_dock_ensemble_data.keys())[0]

        data_to_work = self.uber_dock_ensemble_data[key]
        test = 1


        self.pca_data_all_deeper = self.pca_analysis(custom=data_to_work[['X', 'Y', 'Z']])

        temp_centroid_pca = self.pca_data_all_deeper

        temp_centroid = data_to_work[['X', 'Y', 'Z']]
        self.temp_centroid_deeper = temp_centroid

        if analysis_type == 'PCA':
            self.X_dbscan_deeper = self.pca_data_all_deeper.values
        else:
            self.X_dbscan_deeper = temp_centroid.values

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler

        # USE DBSCAN
        self.dbscan_analysis_deeper = DBSCAN(eps=custom_eps, min_samples=minPoints).fit(self.X_dbscan)

        # Use HDBScan
        # self.dbscan_analysis = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(self.X_dbscan)

        self.core_samples_mask_deeper = np.zeros_like(self.dbscan_analysis_deeper.labels_, dtype=bool)
        self.core_samples_mask_deeper[self.dbscan_analysis_deeper.core_sample_indices_] = True
        self.labels_dbscan_deeper = self.dbscan_analysis_deeper.labels_

        # Number of clusters in labels, ignoring noise if present.
        self.dbscan_n_clusters_deeper = len(set(self.labels_dbscan_deeper)) - (1 if -1 in self.labels_dbscan_deeper else 0)
        print('Number of cluster is ', set(self.labels_dbscan_deeper))

        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        self.dbscan_unique_labels_deeper = set(self.labels_dbscan_deeper)
        self.dbscan_colors_deeper = [plt.cm.Spectral(each)
                              for each in np.linspace(0, 1, len(self.dbscan_unique_labels_deeper))]

        for k, col in zip(self.dbscan_unique_labels_deeper, self.dbscan_colors_deeper):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_dbscan_deeper == k)

            test = 1
            xy = self.X_dbscan_deeper[class_member_mask & self.core_samples_mask_deeper]

            hold_info = xy
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = self.X_dbscan_deeper[class_member_mask & ~self.core_samples_mask_deeper]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

            if k != -1:
                test = 1
                center = calculate_cluster_center_only_docking(hold_info)
                # print('Hello center')

                down = down_region
                object_centroid_circles = ax.scatter(center[0], center[1],
                                                     marker='o', c="white", alpha=1, s=200)
                object_centroid_circles.set_zorder(60)

                c = center
                roman_number = extra_tools.write_roman(int(k + 1))
                object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=150,
                                                          c='b')
                object_centroid_circles_text.set_zorder(61)

        plt.tight_layout()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Estimated number of clusters: %d' % self.dbscan_n_clusters_deeper)
        fig.savefig('DBSCAN_plot_clusters_deeper_{0}.png'.format(self.molecule_name), dpi=1200)

        test = 1

        self.uber_dock_ensemble_deeper_deeper = []

        for k, col in zip(self.dbscan_unique_labels_deeper, self.dbscan_colors_deeper):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_dbscan_deeper == k)

            curr_data = data_to_work[class_member_mask & self.core_samples_mask]
            test = 1

            curr_programs = curr_data['Program']

            curr_prog_values = curr_programs.values

            test = 1

            # this works
            if set(curr_prog_values) == set(check_data):
                self.uber_dock_ensemble_deeper.append(k)

        cluster_list = {}
        for k in self.uber_dock_ensemble_deeper:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:

                selected_mols_list = data_to_work[self.labels_dbscan_deeper == k]
                test = self.molecule_database

                cluster_data = []
                for index, row in selected_mols_list.iterrows():
                    # print(i)
                    program = row['Program']
                    sample = row['Sample']
                    model = row['Model']

                    curr_model = self.molecule_database[program][sample][model]
                    test = 1
                    cluster_data.append(curr_model)
                    # print(xyz.describe())
                cluster_list.update({k: cluster_data})
        # print(cluster_list)

        self.ensemble_models_deeper = cluster_list

        test = 1















    @hlp.timeit
    def cluster_density_uber_centroids_ensemble_plot(self,
                                                     check_data=['LeDock', 'rDock', 'FlexAid', 'Vina'],
                                                     trasparent_alpha=False, down_region=30
                                                     ):

        #         self.data_for_analysis_all

        test = 1

        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        self.quantity_ensemble = 0

        for k, col in zip(self.dbscan_unique_labels, self.dbscan_colors):
            if k not in self.uber_dock_ensemble:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels_dbscan == k)

            xy = self.X_dbscan[class_member_mask & self.core_samples_mask]

            hold_info = xy
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = self.X_dbscan[class_member_mask & ~self.core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

            if k in self.uber_dock_ensemble:
                self.quantity_ensemble += len(hold_info )
                test = 1
                center = calculate_cluster_center_only_docking(hold_info)
                # print('Hello center')

                down = down_region
                object_centroid_circles = ax.scatter(center[0], center[1],
                                                     marker='o', c="white", alpha=1, s=200)
                object_centroid_circles.set_zorder(60)

                c = center
                roman_number = extra_tools.write_roman(int(k + 1))
                object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=150,
                                                          c='b')
                object_centroid_circles_text.set_zorder(61)

        # plt.tight_layout()
        plt.tight_layout()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        self.uber_ensemble_percentage_ = (self.quantity_ensemble / len(self.X_dbscan)* 100)

        ax.set_title('Estimated number of clusters from ensemble: {0} ({1:.3g}%)'.format(len(self.uber_dock_ensemble),
                                                                                         self.uber_ensemble_percentage_))
        #plt.show()
        fig.savefig('DBSCAN_plot_ensemble_clusters_{0}.png'.format(self.molecule_name), dpi=1200)
        test = 1




    def plot_ensemble_energy(self, type='centroid',   check_data=['LeDock', 'rDock', 'FlexAid', 'Vina'],
                                                     trasparent_alpha=False, down_region=30, plot_type='violin'):


        if type == 'centroid':
            type_to_write = 'region'
        else:
            type_to_write = type

        # TODO this is very important
        # if len(self.save_extract_files_list) < 1:
        #self.save_extract_files_list.update({type: {}})

        whole_size = len(self.analysis_dataframe)

        clust_nums = self.uber_dock_ensemble
        clust_vals = self.uber_dock_ensemble_data


        #self.clust_percentage_data = {}
        plt.clf()

        ledock_pandas = None
        rdock_pandas = None
        vina_pandas = None
        flexaid_pandas = None



        for clust in clust_nums:
            #plt.clf()
            cluster =clust_vals[clust]
            # print(cluster)
            # Make nomenclatur similar to plots



            #boxplot = sns.boxplot(x='Program', y='Score', data=cluster)
            #fig = boxplot.get_figure()
            #fig.savefig("output_{0}.png".format(clust), dpi=600)
            cluster['clusterNum'] = extra_tools.write_roman(int(clust+1))


            for prog in check_data:
                prog_info = cluster[cluster['Program']==prog]

                if prog == 'LeDock':
                    if ledock_pandas is None:
                        ledock_pandas = prog_info
                    else:
                        test = 1
                        ledock_pandas = pd.concat([ledock_pandas, prog_info]) #ledock_pandas.merge(prog_info)

                if prog == 'rDock':
                    if rdock_pandas is None:
                        rdock_pandas = prog_info
                    else:
                        rdock_pandas = pd.concat([rdock_pandas, prog_info])

                if prog == 'FlexAid':
                    if flexaid_pandas is None:
                        flexaid_pandas = prog_info
                    else:
                        flexaid_pandas = pd.concat([flexaid_pandas, prog_info])

                if prog == 'Vina':
                    if vina_pandas is None:
                        vina_pandas = prog_info
                    else:
                        vina_pandas = pd.concat([vina_pandas, prog_info])

                test = 1

            # boxplot = sns.boxplot(x='clusterNum', y='Score', hue='LeDock', data=cluster)
            # fig = boxplot.get_figure()
            # fig.savefig("output_{0}.png".format(clust), dpi=600)

            test = 1

        all_pandas = [ledock_pandas, rdock_pandas, flexaid_pandas, vina_pandas]
        for prog_name, prog_data in zip(check_data, all_pandas):
            test = 1
            plt.clf()

            fig, ax = plt.subplots()
            fig.set_size_inches(plot_tools.cm2inch(19, 14))

            if plot_type == 'violin':
                plot_func = sns.violinplot
            elif plot_type == 'box':
                plot_func = sns.boxplot
            plot = plot_func(ax= ax, x='clusterNum', y='Score', data=prog_data)
            #sns.despine()
            #fig = boxplot.get_figure()
            # set up titles and axis names
            ax.set_title(prog_name)
            ax.set_xlabel('Cluster')


            if prog_name == 'LeDock':
                ylabel = 'kcal/mol'
            elif prog_name == 'rDock':
                ylabel = 'Score'
            elif prog_name == 'Vina':
                ylabel = 'kcal/mol'
            elif prog_name == 'FlexAid':
                ylabel = 'Score'

            ax.set_ylabel(ylabel)

            fig.savefig("output_{0}_{1}.png".format(plot_type,prog_name), dpi=600)
            test = 1

        finish = 1






























    @hlp.timeit
    def cluster_uber_centroids(self):

        #         self.data_for_analysis_all

        test = 1
        # print(self.concatenated_analysis__[int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__)+ 1])

        # TODO CENTROID ANALYSIS

        # centroid_indexes = self.concatenated_analysis__[
        #     self.concatenated_analysis__['Type'] == 'centroid'].index.tolist()
        #
        # # temp_centroid = self.pca_data_all[centroid_indexes[0]:centroid_indexes[-1] + 1]
        #
        # data_centroid = self.concatenated_analysis__[
        #                 int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # self.data_centroid = data_centroid
        #
        # # self.concatenated_analysis__[self.data_cols]
        #
        # temp_centroid_pca = self.pca_data_all[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid_pre = self.concatenated_analysis__[
        #                     int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]
        #
        # temp_centroid = temp_centroid_pre[self.data_cols]

        # self.temp_centroid = temp_centroid

        test = 1
        self.range_n_clusters = list(range(2, 10))

        self.pca_data_all = self.pca_analysis(custom=self.analysis_dataframe[['X', 'Y', 'Z']])

        temp_centroid_pca = self.pca_data_all

        temp_centroid = self.analysis_dataframe[['X', 'Y', 'Z']]
        self.temp_centroid = temp_centroid

        # temp_centroid_pca.plot.scatter('component1','component2')
        # plt.show()

        type = 'centroid'
        centroid_cluster_analysis = self.cluster_analysis_custom(temp_centroid, type)
        print('Centroid clustering Finished \n')

        # TODO not a good solution for hdbscan turn it off for now
        # centroid_cluster_hdbscan = self.cluster_analysis_hdbscan(temp_centroid, type)

        # centroid_cluster_analysis_pca = self.cluster_analysis_custom(temp_centroid_pca)
        # print('Centroid PCA clustering Finished \n')

        clusters_info_centroid = centroid_cluster_analysis['clusterInfo']

        clust_num_centroid = centroid_cluster_analysis['clustNum']

        centroid_cluster_models = self.collect_cluster_info_uber(clusters_info_centroid, clust_num_centroid,
                                                                 self.analysis_dataframe)

        # test = 1

        # cluster_labels = centroid_data['dataClustering']['clusterInfo'][n_clusters]['labels']

        # # TODO RESHAPE ANALYSIS
        # reshape_indexes = self.concatenated_analysis__[self.concatenated_analysis__['Type'] == 'reshape'].index.tolist()
        # data_reshape = self.concatenated_analysis__[reshape_indexes[0]:reshape_indexes[-1] + 1]
        #
        # self.data_reshape = data_reshape
        #
        # temp_reshape_pca = self.pca_data_all[reshape_indexes[0]:reshape_indexes[-1] + 1]
        #
        # temp_reshape_pre = self.concatenated_analysis__[reshape_indexes[0]:reshape_indexes[-1] + 1]
        #
        # temp_reshape = temp_reshape_pre[self.data_cols]
        #
        # self.temp_reshape = temp_reshape
        #
        # type = 'reshape'
        # reshape_cluster_analysis = self.cluster_analysis_custom(temp_reshape, type)
        # print('Reshape clustering Finished \n')
        #
        # # reshape_cluster_analysis_pca = self.cluster_analysis_custom(temp_reshape_pca)
        # # print('Reshape PCA clustering Finished \n')
        #
        # clusters_info_reshape = reshape_cluster_analysis['clusterInfo']
        #
        # clust_num_reshape = reshape_cluster_analysis['clustNum']
        #
        # # data_to_return = {
        # #                   'colors':colors,
        # #                   'labels':cluster_labels,
        # #                   'colorData':cluster_colors,
        # #                   'clusterInfo': clusters_info,
        # #                   'clustNum': clust_num,
        # #                   'clusterAnalysisInfo': clust_analysis_info,
        # #                   'overrideClustNum': None}
        #
        # reshape_cluster_models = self.collect_cluster_info_v2(clusters_info_reshape, clust_num_reshape,
        #                                                       temp_reshape, data_reshape)
        #
        # # converters.convert_seaborn_color_to_rgb(self.cluster_colors_pre_rgb)
        #
        # # TODO update this part
        # self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid_pca,
        #                                       'dataToCluster': temp_centroid,
        #                                       'dataOriginal': data_centroid,
        #                                       'clusterModels': centroid_cluster_models,
        #                                       'dataClustering': centroid_cluster_analysis,
        #                                       'clustNum': clust_num_centroid,
        #                                       'overrideClustNum': None}}

        self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid_pca,
                                              'dataToCluster': temp_centroid,
                                              'dataOriginal': self.analysis_dataframe,
                                              'dataClustering': centroid_cluster_analysis,
                                              'clustNum': clust_num_centroid,
                                              'clusterModels': centroid_cluster_models,
                                              'overrideClustNum': None}}

        #                          'reshape': {'dataPCA': temp_reshape_pca,
        #                                      'dataPre': temp_reshape_pre,
        #                                      'dataToCluster': temp_reshape,
        #                                      'dataOriginal': data_reshape,
        #                                      'clusterModels': reshape_cluster_models,
        #                                      'dataClustering': reshape_cluster_analysis,
        #                                      'clustNum': clust_num_reshape,
        #                                      'overrideClustNum': None}}

        print('Finished ultra clustering')
        test = 1

        # TODO fix parallel implementation

    @hlp.timeit
    def show_uber_docking_pca_results(self, custom_dpi=1200, show_plot=False, lang='eng',
                                      trasparent_alpha=False, down_region=30):

        # self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid,
        #                                       'dataOriginal':data_centroid,
        #                                       'clusterModels':centroid_cluster_models,
        #                                       'dataClustering': centroid_cluster_analysis,
        #                                       'clustNum':clust_num_centroid},
        #                          'reshape': {'data': temp_reshape,
        #                                      'dataOriginal': data_reshape,
        #                                      'clusterModels': reshape_cluster_models,
        #                                      'dataClustering': reshape_cluster_analysis,
        #                                      'clustNum': clust_num_reshape}}

        # import seaborn as sns
        sns.set(style="white", context='paper', font_scale=1)

        centroid_data = self.ultra_clustering['centroid']

        if centroid_data['overrideClustNum'] is not None:
            n_clusters = centroid_data['overrideClustNum']
        else:
            n_clusters = centroid_data['clustNum']

        cluster_labels = centroid_data['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old version
        centers = centroid_data['dataClustering']['clusterInfo'][n_clusters]['centers']

        # colors = centroid_data['dataClustering']['colorData']
        # self.colors_ = colors
        #
        # self.cluster_colors = centroid_data['dataClustering']['rgbColors']

        colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['colorData']

        self.colors_ = colors

        self.cluster_colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['rgbColors']

        X = centroid_data['dataPCA']

        # New Version
        centers = calculate_cluster_center_docking(X, cluster_labels)

        test = 1
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        if lang == 'eng':
            plot1_title = "The silhouette plot for the various clusters."
            plot1_xlabel = "The silhouette coefficient values"
            plot1_ylabel = "Cluster label"

            plot2_title = "The visualization of the clustered data."
            # plot2_xlabel = "Feature space for the 1st feature"
            # plot2_ylabel = "Feature space for the 2nd feature"

            plot2_xlabel = "PC1"
            plot2_ylabel = "PC2"

            # plt.set_xlabel("PC1 (Å)")
            # plt.set_ylabel("PC2 (Å)")

            # plot_whole_titile = "Silhouette analysis for KMeans clustering on docking data with n_clusters = %d" % n_clusters
            plot_whole_titile = "Silhouette analysis for k-means clustering on docking data"
        else:
            plot1_title = "Силуэтный график для различных кластеров"
            plot1_xlabel = "Значение коэффициента силуэта"
            plot1_ylabel = "Маркировка кластеров"

            plot2_title = "Визуализация данных кластеризации"
            plot2_xlabel = "Пространство для 1-го признака"
            plot2_ylabel = " Пространство для 2-го признака"

            plot_whole_titile = "Силуэтный анализ на основе данных докинга, используя алгоритм k-средних = %d" % n_clusters

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)

        # This is part i need to modify
        # temp_x1 = X[self.analysis_dataframe]
        markers = list(self.analysis_dataframe[['Marker']].values.flatten())
        labels = list(self.analysis_dataframe[['Program']].values.flatten())
        # object_centroid = ax.scatter(X['component1'], X['component2'], marker=markers, s=80, lw=0, alpha=0.7,
        #                              c=colors)
        # object_centroid.set_zorder(1)

        # Plot individual docking program
        for xp, yp, m, color, label in zip(X['component1'], X['component2'], markers, colors, labels):
            object_centroid = ax.scatter(xp, yp, marker=m, s=80, lw=0, alpha=0.7,
                                         c=color, label=label)  # ,
            object_centroid.set_zorder(1)

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
                                             marker='o', c="white", alpha=1, s=50)
        object_centroid_circles.set_zorder(2)

        for i, c in enumerate(centers):
            roman_number = extra_tools.write_roman(int(i + 1))
            object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=350,
                                                      c='b')
            object_centroid_circles_text.set_zorder(3)

        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        # plt.rcParams['legend.numpoints'] = 1
        # legend = ax.legend(loc="lower left", markerscale=0.7, numpoints=1, fontsize=10)
        # TODO fix legend
        # red_line = matplotlib.lines.Line2D([], [], color='red', markersize=100, label='Blue line')
        #
        # blue_line = matplotlib.lines.Line2D([], [], color='blue', markersize=100, label='Green line')
        # purple_line = matplotlib.lines.Line2D([], [], color='purple', markersize=100, label='Green line')
        #
        # handles = [blue_line, red_line, purple_line]
        # labels = [h.get_label() for h in handles]
        #
        # ax.legend(handles=handles, labels=labels)

        sns.set(style="white", context='paper', font_scale=1)
        ax.grid('on')
        fig.tight_layout()

        self.simulation_name = 'LASR_HSL_UBERDOCK'

        fig.savefig(self.simulation_name + '_best_ultra_docking_PCA_analysis' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    def export_models_for_pymol_simple(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        '''
        Save autodock, flexaid, rdock output to separate files
        :return:
        '''
        self.info_type = 'UberDock'

        self.save_simple_files = {}

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type

        folder_utils.create_folder(save_directory)

        # This part is for saving autodock vina
        filename = "ligBindTraj_info_type:{0}_autodock.pdb".format(self.info_type)
        filename_to_write = save_directory + os.sep + filename

        output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

        # for mol in cluster:
        #     output.write(mol)
        #     output_sdf.write(mol)
        for sample in self.vina_samples_pybel:
            for mol in self.vina_samples_pybel[sample]:
                curr_mol = self.vina_samples_pybel[sample][mol]
                output.write(curr_mol)

        output.close()

        with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('END', 'ENDMDL'), end='')

        test = 1
        self.save_simple_files.update({'AutodockVina': filename_to_write})

        # Part for rDock
        filename = "ligBindTraj_info_type:{0}_rDock.pdb".format(self.info_type)
        filename_to_write = save_directory + os.sep + filename

        output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

        # for mol in cluster:
        #     output.write(mol)
        #     output_sdf.write(mol)
        for sample in self.rDock_samples_pybel:
            for mol in self.rDock_samples_pybel[sample]:
                curr_mol = self.rDock_samples_pybel[sample][mol]
                output.write(curr_mol)

        output.close()

        with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('END', 'ENDMDL'), end='')

        self.save_simple_files.update({'rDock': filename_to_write})

        test = 1

        # Part for LeDock
        filename = "ligBindTraj_info_type:{0}_LeDock.pdb".format(self.info_type)
        filename_to_write = save_directory + os.sep + filename

        output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

        # for mol in cluster:
        #     output.write(mol)
        #     output_sdf.write(mol)
        for sample in self.ledock_samples_pybel:
            for mol in self.ledock_samples_pybel[sample]:
                curr_mol = self.ledock_samples_pybel[sample][mol]
                output.write(curr_mol)

        output.close()

        with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('END', 'ENDMDL'), end='')

        self.save_simple_files.update({'LeDock': filename_to_write})

        test = 1

        # Part for FlexAid
        filename = "ligBindTraj_info_type:{0}_FlexAid.pdb".format(self.info_type)
        filename_to_write = save_directory + os.sep + filename

        output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

        # for mol in cluster:
        #     output.write(mol)
        #     output_sdf.write(mol)
        for sample in self.flexaid_samples_pybel:
            for mol in self.flexaid_samples_pybel[sample]:
                curr_mol = self.flexaid_samples_pybel[sample][mol]
                output.write(curr_mol)

        output.close()

        with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace('END', 'ENDMDL'), end='')

        self.save_simple_files.update({'FlexAid': filename_to_write})

        test = 1

        # mol.addh()
        # output.write(mol)
        # output_sdf.write(mol)

        # self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
        #                                                    'relativePathSDF':filename_to_write_sdf,
        #                                                    'relativePathPDB_fixed':None,
        #                                                    'filename': filename,
        #                                                    'colors': colors[clust],
        #                                                    'rgbColors': rgb_colors[clust],
        #                                                    'currModels': cluster,
        #                                                    'key': clust,
        #                                                    'percentage':percentage}})

        test = 1

    def generate_pymol_viz_uber_simple_export(self, label_atom='C12', folder_name='cluster_traj'):
        '''


        :param receptor: receptor file pdb
        :param exhaust_data:
        :param save_name:
        :param percentage:
        :param label_atom:
        :return:
        '''

        receptor = self.receptor_file_original
        simple_export_files = self.save_simple_files

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        folder_utils.create_folder(save_directory)

        # This part is for saving autodock vina
        filename = "ligBindTraj_simple_export.pse".format(self.info_type)
        filename_to_write = save_directory + os.sep + filename

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

        # self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
        #                                                    'filename': filename,
        #                                                    'colors': colors[clust],
        #                                                    'rgbColors': rgb_colors[clust],
        #                                                    'currModels': cluster,
        #                                                    'key': clust}})

        self.data_to_open = {}
        for prog_output in simple_export_files:
            # molecule = exhaust_data[exhaust_mol]
            # filepath = molecule.file_path
            # data_type = molecule.sample_info_type
            # num = molecule.sample_info_num

            test = 1

            curr_data = simple_export_files[prog_output]

            correct_file_pymol_name = prog_output

            correct_topol_filepath = curr_data
            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            # curr_color = 'exhaus_cluster_color_{0}'.format(curr_index+1)
            # pymol.cmd.set_color(curr_color, curr_data['colors'])
            # pymol.cmd.color(curr_color, correct_file_pymol_name)

            sleep(0.5)

        test = 1

        # in the future
        # pymol.cmd.cealign()
        # This works
        print('Finished Pymol for Uber simple model export  ---- >')
        # save_state_name = save_name
        save_state_name = filename_to_write
        pymol.cmd.save(save_state_name)

        test = 1
        # pymol.cmd.quit()

        # sleep(1)

    def export_ensemble_custom_cluster_models(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''

        self.info_type = 'UberDockEnsemble'

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)

        test = 1

        # if save_pickle_auto is True:
        #     self.save_analysed_data(save_directory + os.sep + '{0}_cluster_data.pickle'.format(self.molecule_name))

        if type == 'centroid':
            type_to_write = 'region'
        else:
            type_to_write = type

        # TODO this is very important

        # if len(self.save_extract_files_list) < 1:
        self.save_extract_files_list.update({type: {}})

        whole_size = len(self.analysis_dataframe)

        self.clust_percentage_data = {}


        self.ensemble_folder_write = namedtuple('Ensemble', 'name, save_directory, dict_data')

        self.ensemble_folder_write.save_directory = save_directory
        self.ensemble_folder_write.name = self.info_type

        self.ensemble_folder_write.dict_data = self.save_extract_files_list[type]

        test = 1

        for clust in self.ensemble_models:
            cluster =self.ensemble_models[clust]
            # print(cluster)
            # Make nomenclatur similar to plots

            clust_size = len(cluster)

            percentage = (clust_size * 100) / whole_size

            self.clust_percentage_data.update({int(clust): percentage})
            print('Clust {0} Percentage {1} \n'.format(clust, percentage))

            filename = "ligBindTraj_type_{0}_{1}_{2}.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename

            output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

            filename_sdf = "ligBindTraj_type_{0}_{1}_{2}.sdf".format(type_to_write, clust + 1, self.info_type)
            filename_to_write_sdf = save_directory + os.sep + filename_sdf
            self.filename_to_write_sdf = filename_to_write_sdf

            output_sdf = pybel.Outputfile("sdf", filename_to_write_sdf, overwrite=True)

            cluster_new = cluster.copy()

            # for mol in cluster:
            #     output.write(mol)
            #     output_sdf.write(mol)
            for mol in cluster_new:

                # TODO Adding hydrogen solves problem with mdtraj so far, need to be very carefulc 
                mol.addh()


                output.write(mol)
                output_sdf.write(mol)

            output.close()
            output_sdf.close()

            with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
                for line in file:
                    print(line.replace('END', 'ENDMDL'), end='')

            test = 1

            # percentage = self.clust_percentage_data[int(clust)]

            self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
                                                               'relativePathSDF': filename_to_write_sdf,
                                                               'relativePathPDB_fixed': None,
                                                               'filename': filename,
                                                               'currModels': cluster,
                                                               'key': clust,
                                                               'percentage': percentage}})

        test =1

        self.ensemble_folder_write.dict_data = self.save_extract_files_list[type]



    def extract_pybel_fixed_centroids_auto_custom(self):
        print('\n')
        print('----------------------------------------------------')
        print('Extracting Ultra analysis fixed pybel Centroids extraction Now')

        keys = list(self.save_extract_files_list['centroid'])

        # NEEED TO BE VERY CAREFUL HERE
        self.full_data_mdtraj_analysis = {}
        for key in keys:
            self.centroid_data = {}
            print('----->Type of data based on {0}  ---<\n'.format(key))
            indexes = list(range(len(self.save_extract_files_list['centroid'][key])))
            curr_data= self.save_extract_files_list['centroid'][key]

            test = 1


            self.curr_index = key
            print('Curr cluster traj index ', self.curr_index)

            # TODO pdb_file wrong
            self.centroid_data.update({self.curr_index: {}})
            data_to_work_with = curr_data
            # ligand_conf = '.' + os.sep + self.folder_path + os.sep + data_to_work_with
            self.find_centroid_custom(data_to_work_with)
            print('-------------------')
            test = 1

            self.full_data_mdtraj_analysis.update({key: self.centroid_data})
            test = 1

        test = 1


    def extract_centroids_auto_custom(self):
        print('\n')
        print('----------------------------------------------------')
        print('Extracting Ultra analysis Centroids Now')

        keys = list(self.ensemble_folder_write.dict_data.keys())

        # NEEED TO BE VERY CAREFUL HERE
        self.full_data_mdtraj_analysis = {}
        for key in keys:
            self.centroid_data = {}
            print('----->Type of data based on {0}  ---<\n'.format(key))
            indexes = list(range(len(self.ensemble_folder_write.dict_data[key])))
            curr_data= self.ensemble_folder_write.dict_data[key]

            test = 1


            self.curr_index = key
            print('Curr cluster traj index ', self.curr_index)

            # TODO pdb_file wrong
            self.centroid_data.update({self.curr_index: {}})
            data_to_work_with = curr_data
            # ligand_conf = '.' + os.sep + self.folder_path + os.sep + data_to_work_with
            self.find_centroid_custom(data_to_work_with)
            print('-------------------')
            test = 1

            self.full_data_mdtraj_analysis.update({key: self.centroid_data})
            test = 1

        test = 1

        # self.centroid_data = {}
        # for pdb_file, index in zip(self.pdb_files, indexes):
        #     print('Curr cluster traj index ', index)
        #     self.curr_index = index
        #
        #     self.centroid_data.update({index: {}})
        #     ligand_conf = '.' + os.sep + self.folder_path + os.sep + pdb_file
        #     self.find_centroid(ligand_conf, pdb_file)
        #     print('---------------------------------------------------------\n')

    def find_centroid_custom(self, data_to_work_with, overwrite_file=True):
        print('\n')
        #


        test = 1

        # ligand_conf = '.' + os.sep + self.folder_path + os.sep + data_to_work_with['filename']



        ligand_conf = data_to_work_with['relativePath']

        pre_process = data_to_work_with['relativePath'].split('.')[1]

        ligand_save_name = pre_process

        traj = md.load(ligand_conf)
        real_originalData = traj[:]

        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((traj.n_frames, traj.n_frames))
        for i in range(traj.n_frames):
            distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

        print('len(distances) ', len(distances))
        print('Max pairwise rmsd: %f nm' % np.max(distances))
        print('Min pairwise rmsd: %f nm' % np.min(distances))

        # This is the part for centroid


        # THIS works perfectly well
        # for beta in range(1,2):
        #     index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        #     centroid = traj[index]
        #     print(centroid)
        #
        #     # bindEner_curr = data_to_work_with['currModels'][index]['molDetail']['vina_info'][0]
        #     # print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))
        #
        #     # print('Color of represenative structure is {0}'.format(data_to_work_with['colors']))
        #     # print('RGB Color of represenative structure is {0}'.format(data_to_work_with['rgbColors']))
        #     #
        #     # actual_name, closest_name = plot_tools.get_cluster_color_name(data_to_work_with['rgbColors'])
        #     #
        #     # print('----------------------------------------------------\n')
        #
        #     # toWrite = real_originalData[index]
        #     toWrite = centroid
        #     print('toWrite checking is ', toWrite)
        #
        #     test = 1
        #
        #     centroid_file = '.' + ligand_save_name + '_beta:{0}_'.format(beta) + '_centroid.pdb'
        #     print('centroid file is ', centroid_file)
        #     toWrite.save(centroid_file, force_overwrite=overwrite_file)



        ################################################# Look different betas
        max_beta = 80000

        checked_indexes = []
        for beta in range(1,max_beta):
            index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()

            if index not in checked_indexes:
                print('Index is ', index)
                checked_indexes.append(index)
                centroid = traj[index]
                print(centroid)

                # bindEner_curr = data_to_work_with['currModels'][index]['molDetail']['vina_info'][0]
                # print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))

                # print('Color of represenative structure is {0}'.format(data_to_work_with['colors']))
                # print('RGB Color of represenative structure is {0}'.format(data_to_work_with['rgbColors']))
                #
                # actual_name, closest_name = plot_tools.get_cluster_color_name(data_to_work_with['rgbColors'])
                #
                # print('----------------------------------------------------\n')

                # toWrite = real_originalData[index]
                toWrite = centroid
                print('toWrite checking is ', toWrite)

                test = 1

                centroid_file = '.' + ligand_save_name + '_beta:{0}_'.format(beta) + '_centroid.pdb'
                print('centroid file is ', centroid_file)
                toWrite.save(centroid_file, force_overwrite=overwrite_file)






        # Very buggy
        # look_diff_beta = 3
        # beta = 0
        # index_prev = 0
        # while look_diff_beta != 0:
        #     beta +=1
        #     index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        #     #print('Index for representative structure ', index)
        #
        #     if index == index_prev:
        #         continue
        #     else:
        #         print('Index for representative structure ', index)
        #         index_prev = index
        #         look_diff_beta -=1
        #
        #         centroid = traj[index]
        #         print(centroid)
        #
        #         #bindEner_curr = data_to_work_with['currModels'][index]['molDetail']['vina_info'][0]
        #         #print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))
        #
        #         # print('Color of represenative structure is {0}'.format(data_to_work_with['colors']))
        #         # print('RGB Color of represenative structure is {0}'.format(data_to_work_with['rgbColors']))
        #         #
        #         # actual_name, closest_name = plot_tools.get_cluster_color_name(data_to_work_with['rgbColors'])
        #         #
        #         # print('----------------------------------------------------\n')
        #
        #         # toWrite = real_originalData[index]
        #         toWrite = centroid
        #         print('toWrite checking is ', toWrite)
        #
        #         test = 1
        #
        #         centroid_file = '.' + ligand_save_name + '_beta:{0}_'.format(beta) + '_centroid.pdb'
        #         print('centroid file is ', centroid_file)
        #         toWrite.save(centroid_file, force_overwrite=overwrite_file)



        test = 1




    def parallel_run(self, working_folder, centroid_data, centoid_data_index, correct_topology, parallel, num_threads):

        topology_to_fix = centroid_data[centoid_data_index]['centroid_file']

        topology_reconstruction_object = topology_reconstruction.TopologyReconstructionObject(
            working_folder=working_folder,
            topology_to_fix=topology_to_fix,
            correct_topology=correct_topology,
            parallel=parallel,
            num_threads=num_threads)

        # Write correct pdb file
        print('Writing correct topology ')
        topology_reconstruction_object.write_correct_topology()

        saved_topology_file = topology_reconstruction_object.get_correct_topology_filename()

        print('------->>>>>\n')

        # to_return = {topology_to_fix: {'reconstructionObject':topology_reconstruction,
        #                                'savedTopologyFile':saved_topology_file,
        #                                'topologyToFixName':topology_to_fix}}

        return [centoid_data_index, centroid_data, topology_to_fix, saved_topology_file, topology_reconstruction_object]
        # return to_return



    def reconstruct_centroids_topology_custom(self, working_folder_path,
                                              correct_topology_info,
                                              parallel=False,
                                              real_parallelism=True,
                                              num_of_threads=7):

        self.add_correct_topology(correct_topology_info)

        for key in self.full_data_mdtraj_analysis:
            centroid_data = self.full_data_mdtraj_analysis[key]
            if real_parallelism is True:
                pool = multiprocessing.Pool(num_of_threads)

                # range_n_clusters = list(range(1, 11))

                function_arguments_to_call = [[working_folder_path, centroid_data, centoid_data_index,
                                               self.correct_topology, False, 7] for
                                              centoid_data_index in centroid_data]

                test = 1

                # results = pool.starmap(parallel_md_silhouette_analysis_pca,self.range_n_clusters,self.reduced_cartesian,
                #                                                 self.pca_traj.time, algorithm )
                results = pool.starmap(self.parallel_run, function_arguments_to_call)

                test = 1
                # d = [x for x in results]
                # d = [list(x.keys())[0] for x in results]

                # d = {key: value for (key, value) in iterable}
                # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
                indexes = [x[0] for x in results]

                for i in indexes:
                    centroid_data[i].update({'correct_topol_object': results[i][4]})
                    centroid_data[i].update({'correct_topol_filepath': results[i][3]})

                # self.clusters_info = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}

                # This will free memory
                pool.close()


            elif real_parallelism is False:
                for centoid_data_index in centroid_data:
                    centroid_file_path = centroid_data[centoid_data_index]['centroid_file']
                    topology_reconstruction_object = topology_reconstruction.TopologyReconstructionObject(
                        working_folder=working_folder_path,
                        topology_to_fix=centroid_file_path,
                        correct_topology=self.correct_topology,
                        parallel=parallel)

                    # Write correct pdb file
                    print('Writing correct topology ')
                    topology_reconstruction_object.write_correct_topology()

                    saved_topology_file = topology_reconstruction_object.get_correct_topology_filename()

                    centroid_data[centoid_data_index].update({'correct_topol_object': topology_reconstruction_object})
                    centroid_data[centoid_data_index].update({'correct_topol_filepath': saved_topology_file})

                    test = 1



    def export_custom_cluster_models(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''
        data_to_use = self.ultra_clustering[type]

        if data_to_use['overrideClustNum'] is not None:
            n_clusters = data_to_use['overrideClustNum']
        else:
            n_clusters = data_to_use['clustNum']

        # n_clusters = data_to_use['clustNum']

        data_clustering = data_to_use['dataClustering']

        colors = data_clustering['colors']
        rgb_colors = data_clustering['rgbColors']

        self.info_type = 'UberDock'

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)

        test = 1

        if save_pickle_auto is True:
            self.save_analysed_data(save_directory + os.sep + '{0}_cluster_data.pickle'.format(self.molecule_name))

        if type == 'centroid':
            type_to_write = 'region'
        else:
            type_to_write = type

        # TODO this is very important
        clust_numbers = data_to_use['clusterModels']

        # if len(self.save_extract_files_list) < 1:
        self.save_extract_files_list.update({type: {}})

        whole_size = len(self.analysis_dataframe)

        self.clust_percentage_data = {}

        for clust in clust_numbers:
            cluster = clust_numbers[clust]
            # print(cluster)
            # Make nomenclatur similar to plots

            clust_size = len(cluster)

            percentage = (clust_size * 100) / whole_size

            self.clust_percentage_data.update({int(clust): percentage})
            print('Clust {0} Percentage {1} \n'.format(clust, percentage))

            filename = "ligBindTraj_type_{0}_{1}_{2}.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename

            output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

            filename_sdf = "ligBindTraj_type_{0}_{1}_{2}.sdf".format(type_to_write, clust + 1, self.info_type)
            filename_to_write_sdf = save_directory + os.sep + filename_sdf
            self.filename_to_write_sdf = filename_to_write_sdf

            output_sdf = pybel.Outputfile("sdf", filename_to_write_sdf, overwrite=True)

            cluster_new = cluster.copy()

            # for mol in cluster:
            #     output.write(mol)
            #     output_sdf.write(mol)
            for mol in cluster_new:
                mol.addh()
                output.write(mol)
                output_sdf.write(mol)

            output.close()
            output_sdf.close()

            with fileinput.FileInput(filename_to_write, inplace=True, backup='.bak') as file:
                for line in file:
                    print(line.replace('END', 'ENDMDL'), end='')

            test = 1

            # percentage = self.clust_percentage_data[int(clust)]

            self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
                                                               'relativePathSDF': filename_to_write_sdf,
                                                               'relativePathPDB_fixed': None,
                                                               'filename': filename,
                                                               'colors': colors[clust],
                                                               'rgbColors': rgb_colors[clust],
                                                               'currModels': cluster,
                                                               'key': clust,
                                                               'percentage': percentage}})

            # res_num = 1
            # for model in cluster:
            #     # curr_model = model
            #     curr_df = model['molDetail']['dataframe']
            #     pdb_tools.write_lig(curr_df, res_num, file_to_write)
            #     # self.write_model_to_file(curr_model, res_num, file_to_write)
            #     res_num += 1
            #     file_to_write.write('ENDMDL\n')
            # file_to_write.close()

        test = 1

    # TODO dont use h bonds for next version, reconstruct only non H atoms
    # TODO when use your script for Hydrogens
    def automorphism_ligand(self):
        ob = pybel.ob

        self.correct_ligand = next(pybel.readfile("pdb", self.ligand_file))

        obmol = self.correct_ligand.OBMol

        # TODO this works
        for res in ob.OBResidueIter(obmol):
            for atom in ob.OBResidueAtomIter(res):
                atName = res.GetAtomID(atom)
                print("Atom name:", atName)

                # Find automorphisms involving only non-H atoms
        self.correct_mappings = pybel.ob.vvpairUIntUInt()

        self.correct_bitvec = pybel.ob.OBBitVec()

        self.correct_lookup = []

        for i, atom in enumerate(self.correct_ligand):
            print(atom)
            # if not atom.OBAtom.IsHydrogen():
            self.correct_bitvec.SetBitOn(i + 1)
            self.correct_lookup.append(i)
        self.success = pybel.ob.FindAutomorphisms(self.correct_ligand.OBMol, self.correct_mappings, self.correct_bitvec)

        test = 1


    def fix_ensemble_topology_ultra_using_pybel(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        ob = pybel.ob
        print('pybel topology ultra fix')

        self.info_type = 'UberDockEnsembleFixed'

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)

        test = 1

        if save_pickle_auto is True:
            self.save_analysed_data(save_directory + os.sep + '{0}_cluster_data_ensembleFixed.pickle'.format(self.molecule_name))

        if type == 'centroid':
            type_to_write = 'region'
        else:
            type_to_write = type

        # TODO this is very important
        # if len(self.save_extract_files_list) < 1:
        self.save_extract_files_list.update({type: {}})

        whole_size = len(self.analysis_dataframe)

        self.clust_percentage_data = {}

        for clust in self.ensemble_models:
            cluster =self.ensemble_models[clust]
            # print(cluster)
            # Make nomenclatur similar to plots

            clust_size = len(cluster)

            percentage = (clust_size * 100) / whole_size

            self.clust_percentage_data.update({int(clust): percentage})
            print('Clust {0} Percentage {1} \n'.format(clust, percentage))

            filename = "ligBindTraj_type_{0}_{1}_{2}.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename

            output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

            # xtalcoords = [atom.coords for atom in self.correct_ligand if not atom.OBAtom.IsHydrogen()]
            xtalcoords = [atom.coords for atom in self.correct_ligand]

            # TODO TEST
            query = ob.CompileMoleculeQuery(self.correct_ligand.OBMol)
            mapper = ob.OBIsomorphismMapper.GetInstance(query)

            obmol_correct = self.correct_ligand.OBMol

            model_dicts = {}
            model_dicts_dataframe = {}
            mol_num = 1

            # TODO SMILES PATTERN NOT GOOD BUGGY MAYBE SMARTS
            for mol in cluster:
                mol.addh()

                isomorph = ob.vpairUIntUInt()
                mapper.MapFirst(mol.OBMol, isomorph)
                # print('part 1')
                # print(list(isomorph))

                # print('let see part 2')
                isomorphs = ob.vvpairUIntUInt()
                mapper.MapAll(mol.OBMol, isomorphs)
                for x in isomorphs:
                    # print(x)
                    isomorph_found = x

                temp_mol = []
                for index_pair in isomorph_found:
                    # print(index_pair)

                    correct_ind = index_pair[0]

                    to_look_ind = index_pair[1]

                    obmol_coords = mol.OBMol

                    coords_to_use = mol.atoms[to_look_ind].coords

                    for res in ob.OBResidueIter(obmol_correct):
                        for atom in ob.OBResidueAtomIter(res):
                            # print(atom)

                            atom_index = int(atom.GetIdx())
                            if atom_index - 1 == int(correct_ind):

                                atName = res.GetAtomID(atom)

                                atName_fixed = atName.split(' ')[1]
                                # print("Atom name:", atName_fixed)
                                test = 1

                                resname = res.GetName()

                                atom_type_orig = atom.GetType()
                                atom_type = atom_types[atom_type_orig]

                                # TODO this part needs to become pandas row
                                temp_local = ['ATOM', correct_ind + 1, atName_fixed, resname, 'Z', mol_num,
                                              round(coords_to_use[0], 3),
                                              round(coords_to_use[1], 3),
                                              round(coords_to_use[2], 3),
                                              '1.00',
                                              '0.00',
                                              atom_type
                                              ]

                                if correct_ind == 0:
                                    temp_dataframe = pd.DataFrame([temp_local])

                                    test = 1

                                    temp_dataframe.columns = ['ATOM', 'SerialNum', 'AtomName',
                                                              'ResidueName', 'ChainId', 'ChainNum',
                                                              'X', 'Y', 'Z', 'Occupancy',
                                                              'TempFactor', 'ElemSymbol']

                                    test = 1
                                else:
                                    test = 1
                                    temp_dataframe.loc[correct_ind] = temp_local
                                    test = 1

                                temp_mol.append(temp_local)
                                test = 1



                    test = 1

                temp_dataframe = temp_dataframe.sort_values('SerialNum')
                model_dicts_dataframe.update({mol_num: temp_dataframe})
                model_dicts.update({mol_num: temp_mol})
                mol_num += 1
            test = 1

            save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                             '_' + self.info_type
            self.save_cluster_models_dir = save_directory
            folder_utils.create_folder(save_directory)

            filename = "ligBindTraj_type_{0}_{1}_{2}_fixed.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename
            file_to_write = open(filename_to_write, 'w')

            res_num = 1
            for model in model_dicts_dataframe:
                # curr_model = model
                curr_df = model_dicts_dataframe[model]
                pdb_tools.write_lig(curr_df, res_num, file_to_write)
                # self.write_model_to_file(curr_model, res_num, file_to_write)
                res_num += 1
                file_to_write.write('ENDMDL\n')
            file_to_write.close()

            self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
                                                               'filename': filename,
                                                               'currModels': cluster,
                                                               'key': clust,
                                                               'percentage': percentage}})

            print('----------------------\n')
            test = 1

            # TODO convert correct residues and save and see if correct tomorrow

            # posecoords = [atom.coords for atom in mol]
            # for mapping in self.correct_mappings:
            #     automorph_coords = [None] * len(xtalcoords)
            #     for x, y in mapping:
            #         automorph_coords[self.correct_lookup.index(x)] = xtalcoords[self.correct_lookup.index(y)]
            #
            #     test = 1
            # mapping_rmsd = rmsd(posecoords, automorph_coords)

            test = 1



    def fix_topology_using_pybel(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        ob = pybel.ob

        data_to_use = self.ultra_clustering[type]

        if data_to_use['overrideClustNum'] is not None:
            n_clusters = data_to_use['overrideClustNum']
        else:
            n_clusters = data_to_use['clustNum']

        # n_clusters = data_to_use['clustNum']

        data_clustering = data_to_use['dataClustering']

        colors = data_clustering['colors']
        rgb_colors = data_clustering['rgbColors']

        self.info_type = 'UberDock'

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)

        test = 1

        if save_pickle_auto is True:
            self.save_analysed_data(save_directory + os.sep + '{0}_cluster_data.pickle'.format(self.molecule_name))

        if type == 'centroid':
            type_to_write = 'region'
        else:
            type_to_write = type

        # TODO this is very important
        clust_numbers = data_to_use['clusterModels'].copy()
        # if len(self.save_extract_files_list) < 1:
        self.save_extract_files_list.update({type: {}})

        whole_size = len(self.analysis_dataframe)

        self.clust_percentage_data = {}

        for clust in clust_numbers:
            cluster = clust_numbers[clust]
            # print(cluster)
            # Make nomenclatur similar to plots

            clust_size = len(cluster)

            percentage = (clust_size * 100) / whole_size

            self.clust_percentage_data.update({int(clust): percentage})
            print('Clust {0} Percentage {1} \n'.format(clust, percentage))

            filename = "ligBindTraj_type_{0}_{1}_{2}.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename

            output = pybel.Outputfile("pdb", filename_to_write, overwrite=True)

            # xtalcoords = [atom.coords for atom in self.correct_ligand if not atom.OBAtom.IsHydrogen()]
            xtalcoords = [atom.coords for atom in self.correct_ligand]

            # TODO TEST
            query = ob.CompileMoleculeQuery(self.correct_ligand.OBMol)
            mapper = ob.OBIsomorphismMapper.GetInstance(query)

            obmol_correct = self.correct_ligand.OBMol

            model_dicts = {}
            model_dicts_dataframe = {}
            mol_num = 1

            # TODO SMILES PATTERN NOT GOOD BUGGY MAYBE SMARTS
            for mol in cluster:
                mol.addh()

                isomorph = ob.vpairUIntUInt()
                mapper.MapFirst(mol.OBMol, isomorph)
                # print('part 1')
                # print(list(isomorph))

                # print('let see part 2')
                isomorphs = ob.vvpairUIntUInt()
                mapper.MapAll(mol.OBMol, isomorphs)
                for x in isomorphs:
                    # print(x)
                    isomorph_found = x

                temp_mol = []
                for index_pair in isomorph_found:
                    # print(index_pair)

                    correct_ind = index_pair[0]

                    to_look_ind = index_pair[1]

                    obmol_coords = mol.OBMol

                    coords_to_use = mol.atoms[to_look_ind].coords

                    for res in ob.OBResidueIter(obmol_correct):
                        for atom in ob.OBResidueAtomIter(res):
                            # print(atom)

                            atom_index = int(atom.GetIdx())
                            if atom_index - 1 == int(correct_ind):

                                atName = res.GetAtomID(atom)

                                atName_fixed = atName.split(' ')[1]
                                # print("Atom name:", atName_fixed)
                                test = 1

                                resname = res.GetName()

                                atom_type_orig = atom.GetType()
                                atom_type = atom_types[atom_type_orig]

                                # TODO this part needs to become pandas row
                                temp_local = ['ATOM', correct_ind + 1, atName_fixed, resname, 'Z', mol_num,
                                              round(coords_to_use[0], 3),
                                              round(coords_to_use[1], 3),
                                              round(coords_to_use[2], 3),
                                              '1.00',
                                              '0.00',
                                              atom_type
                                              ]

                                if correct_ind == 0:
                                    temp_dataframe = pd.DataFrame([temp_local])

                                    test = 1

                                    temp_dataframe.columns = ['ATOM', 'SerialNum', 'AtomName',
                                                              'ResidueName', 'ChainId', 'ChainNum',
                                                              'X', 'Y', 'Z', 'Occupancy',
                                                              'TempFactor', 'ElemSymbol']

                                    test = 1
                                else:
                                    test = 1
                                    temp_dataframe.loc[correct_ind] = temp_local
                                    test = 1

                                temp_mol.append(temp_local)
                                test = 1
                    test = 1
                model_dicts_dataframe.update({mol_num: temp_dataframe})
                model_dicts.update({mol_num: temp_mol})
                mol_num += 1
            test = 1

            save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                             '_' + self.info_type
            self.save_cluster_models_dir = save_directory
            folder_utils.create_folder(save_directory)

            filename = "ligBindTraj_type_{0}_{1}_{2}_fixed.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename
            file_to_write = open(filename_to_write, 'w')

            res_num = 1
            for model in model_dicts_dataframe:
                # curr_model = model
                curr_df = model_dicts_dataframe[model]
                pdb_tools.write_lig(curr_df, res_num, file_to_write)
                # self.write_model_to_file(curr_model, res_num, file_to_write)
                res_num += 1
                file_to_write.write('ENDMDL\n')
            file_to_write.close()

            # TEST

            # mol.atoms[4].coords
            # res.GetIdx()
            # atom.GetIdx()
            # atom.GetType()

            # GetName (void) const
            # unsigned int 	GetNum (void)
            # std::string 	GetNumString (void)
            # unsigned int 	GetNumAtoms () const
            # char 	GetChain (void) const
            # unsigned int 	GetChainNum (void) const
            # unsigned int 	GetIdx (void) const
            # unsigned int 	GetResKey (void) const

            print('----------------------\n')
            test = 1

            # TODO convert correct residues and save and see if correct tomorrow

            # posecoords = [atom.coords for atom in mol]
            # for mapping in self.correct_mappings:
            #     automorph_coords = [None] * len(xtalcoords)
            #     for x, y in mapping:
            #         automorph_coords[self.correct_lookup.index(x)] = xtalcoords[self.correct_lookup.index(y)]
            #
            #     test = 1
            # mapping_rmsd = rmsd(posecoords, automorph_coords)

            test = 1

    def fix_topology_using_rdkit(self, type='centroid', save_pickle_auto=False, folder_name='cluster_traj'):
        to_extract = self.save_extract_files_list[type]

        from rdkit.Chem import AllChem
        from rdkit import Chem

        correct_ligand = Chem.MolFromPDBFile(self.ligand_file)

        atoms = correct_ligand.GetAtoms()
        # dir(list(atoms)[0])
        # list(atoms)[0].GetSmarts()

        test = Chem.MolToSmiles(correct_ligand, isomericSmiles=True)
        print('corr_ligand ', test)

        for i in to_extract:
            relative_sdf = to_extract[i]['relativePathSDF']
            test = 1

            suppl = Chem.SDMolSupplier(relative_sdf)
            for mol_to_fix in suppl:
                test = 1

                # testus = fingerprint_tools.get_circular_similarity(correct_ligand, mol_to_fix)
                # testus1 = fingerprint_tools.get_atom_pair_similarity(correct_ligand, mol_to_fix)
                testus1 = fingerprint_tools.get_gobbi_similarity(correct_ligand, mol_to_fix)

                # print(mol.GetNumAtoms())

            test = set(Chem.MolToSmiles(item, isomericSmiles=True) for item in suppl)
            print(test)

            test = 1

        test = 1

        # query = ob.CompileMoleculeQuery(self.correct_ligand.OBMol)
        # mapper = ob.OBIsomorphismMapper.GetInstance(query)
        #
        # obmol_correct = self.correct_ligand.OBMol
        #
        #
        # model_dicts = {}
        # model_dicts_dataframe= {}
        # mol_num = 1
        #
        # # TODO SMILES PATTERN NOT GOOD BUGGY MAYBE SMARTS
        # for mol in cluster:
        #     mol.addh()
        #
        #     isomorph = ob.vpairUIntUInt()
        #     mapper.MapFirst(mol.OBMol, isomorph)
        #     # print('part 1')
        #     # print(list(isomorph))
        #
        #
        #     #print('let see part 2')
        #     isomorphs = ob.vvpairUIntUInt()
        #     mapper.MapAll(mol.OBMol, isomorphs)
        #     for x in isomorphs:
        #         #print(x)
        #         isomorph_found = x
        #
        #
        #     temp_mol = []
        #     for index_pair in isomorph_found:
        #         # print(index_pair)
        #
        #         correct_ind = index_pair[0]
        #
        #         to_look_ind = index_pair[1]
        #
        #         obmol_coords = mol.OBMol
        #
        #         coords_to_use = mol.atoms[to_look_ind].coords
        #
        #         for res in ob.OBResidueIter(obmol_correct):
        #             for atom in ob.OBResidueAtomIter(res):
        #                 #print(atom)
        #
        #                 atom_index = int(atom.GetIdx())
        #                 if atom_index-1 == int(correct_ind):
        #
        #                     atName = res.GetAtomID(atom)
        #
        #                     atName_fixed =atName.split(' ')[1]
        #                     #print("Atom name:", atName_fixed)
        #                     test = 1
        #
        #                     resname = res.GetName()
        #
        #                     atom_type_orig  = atom.GetType()
        #                     atom_type = atom_types[atom_type_orig]
        #
        #
        #                     # TODO this part needs to become pandas row
        #                     temp_local = ['ATOM', correct_ind+1, atName_fixed, resname, 'Z', mol_num,
        #                             round(coords_to_use[0],3)  ,
        #                             round(coords_to_use[1],3) ,
        #                             round(coords_to_use[2],3) ,
        #                             '1.00' ,
        #                             '0.00',
        #                             atom_type
        #                             ]
        #
        #                     if correct_ind == 0:
        #                         temp_dataframe = pd.DataFrame([temp_local])
        #
        #                         test  = 1
        #
        #                         temp_dataframe.columns = ['ATOM', 'SerialNum', 'AtomName',
        #                                                   'ResidueName', 'ChainId', 'ChainNum',
        #                                                   'X', 'Y','Z', 'Occupancy',
        #                                                   'TempFactor', 'ElemSymbol']
        #
        #                         test = 1
        #                     else:
        #                         test = 1
        #                         temp_dataframe.loc[correct_ind] =  temp_local
        #                         test = 1
        #
        #
        #                     temp_mol.append(temp_local)
        #                     test = 1
        #         test = 1
        #     model_dicts_dataframe.update({mol_num:temp_dataframe})
        #     model_dicts.update({mol_num: temp_mol})
        #     mol_num += 1
        # test = 1
        #
        # save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
        #                  '_' + self.info_type
        # self.save_cluster_models_dir = save_directory
        # folder_utils.create_folder(save_directory)
        #
        # filename = "ligBindTraj_type_{0}_{1}_{2}_fixed.pdb".format(type_to_write, clust + 1, self.info_type)
        # filename_to_write = save_directory + os.sep + filename
        # file_to_write = open(filename_to_write, 'w')
        #
        #
        # res_num = 1
        # for model in model_dicts_dataframe:
        #     # curr_model = model
        #     curr_df = model_dicts_dataframe[model]
        #     pdb_tools.write_lig(curr_df, res_num, file_to_write)
        #     # self.write_model_to_file(curr_model, res_num, file_to_write)
        #     res_num += 1
        #     file_to_write.write('ENDMDL\n')
        # file_to_write.close()
        #
        #
        #

    @hlp.timeit
    def show_uber_cluster_quality_analysis_plots(self, data_type='centroid', custom_dpi=1200, show_plot=False,
                                                 lang='eng',
                                                 trasparent_alpha=False):
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                     2)  # sharex='col', sharey='row') TODO this can be used for shared columns

        fig.set_size_inches(plot_tools.cm2inch(17, 17))

        sns.set(font_scale=0.9)

        main_data = self.ultra_clustering[data_type]
        test = 1

        data_clustering = main_data['dataClustering']
        cluster_analysis_info = data_clustering['clusterAnalysisInfo']

        cluster_range = self.range_n_clusters
        score = cluster_analysis_info['dbiBook']
        if lang == 'rus':
            criteria_name = 'Индекс Дэвиса-Болдина'
            score_text = 'Оптимальным решением кластеризации\n будет минимальное значение индекса Дэвиса-Болдина'
        else:
            criteria_name = 'Davis-Bouldain Index'
            score_text = 'The optimal clustering solution\n has the smallest Davies-Bouldin index value.'
        ax1.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax1.plot(cluster_range, score, ':k', linewidth=3.0)

        ax1.set_xlim(cluster_range[0], cluster_range[-1])

        ax1.set_title(score_text)
        # ax1.set_xlabel('n of clusters')
        ax1.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = cluster_analysis_info['dunn']
        if lang == 'rus':
            criteria_name = 'Индекс Данна'
            score_text = 'Оптимальным решением кластеризации\n будет максимальное значение индекса Данна'
        else:
            criteria_name = "Dunn's Index"
            score_text = "Maximum value of the index\n represents the right partitioning given the index"
        ax2.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax2.plot(cluster_range, score, ':k', linewidth=3.0)

        ax2.set_xlim(cluster_range[0], cluster_range[-1])
        ax2.set_title(score_text)
        # ax2.set_xlabel('n of clusters')
        ax2.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = cluster_analysis_info['silhouette']
        if lang == 'rus':
            criteria_name = 'Среднее значение коэффициента\n силуэта для всех образцов'
            score_text = 'Объекты с высоким значением силуэта\n считаются хорошо сгруппированными.'
        else:
            criteria_name = 'Mean Silhouette Coefficient for all samples'
            score_text = 'Objects with a high silhouette value\n are considered well clustered'
        ax3.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax3.plot(cluster_range, score, ':k', linewidth=3.0)

        ax3.set_xlim(cluster_range[0], cluster_range[-1])
        ax3.set_title(score_text)
        ax3.set_xlabel('n of clusters')
        ax3.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = cluster_analysis_info['calinski']
        if lang == 'rus':
            criteria_name = 'Оценка Калински-Харабаз '
            score_text = 'Объекты с высоким значением оценки Калински-Харабаз\n считаются хорошо сгруппированными'
        else:
            criteria_name = 'Calinski-Harabaz score'
            score_text = 'Objects with a high Calinski-Harabaz score\n value are considered well clustered'
        ax4.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax4.plot(cluster_range, score, ':k', linewidth=3.0)
        ax4.set_xlim(cluster_range[0], cluster_range[-1])
        ax4.set_title(score_text)
        ax4.set_xlabel('n of clusters')
        ax4.set_ylabel(criteria_name)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.90)
        if lang == 'rus':
            suptitle_text = 'Определение оптимального количества кластеров'
        else:
            suptitle_text = "Determination of optimal number of Clusters"

        # fig.tight_layout()

        plt.suptitle((suptitle_text),
                     fontsize=14, fontweight='bold')

        fig.savefig(self.simulation_name + '_{0}_'.format(lang) +
                    'dataType:{0}_cluster_determination.png'.format(data_type), dpi=custom_dpi,
                    transparent=trasparent_alpha)

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_uber_docking_pca_simple_results_v2(self, custom_dpi=1200, show_plot=False, lang='eng',
                                             trasparent_alpha=False, down_region=30,
                                             load_data=['LeDock', 'rDock', 'FlexAid', 'Vina']):

        # self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid,
        #                                       'dataOriginal':data_centroid,
        #                                       'clusterModels':centroid_cluster_models,
        #                                       'dataClustering': centroid_cluster_analysis,
        #                                       'clustNum':clust_num_centroid},
        #                          'reshape': {'data': temp_reshape,
        #                                      'dataOriginal': data_reshape,
        #                                      'clusterModels': reshape_cluster_models,
        #                                      'dataClustering': reshape_cluster_analysis,
        #                                      'clustNum': clust_num_reshape}}

        # import seaborn as sns
        # sns.set(style="white", context='paper', font_scale=1)



        X = self.pca_data_all

        # New Version

        test = 1
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        if lang == 'eng':
            plot1_title = "The silhouette plot for the various clusters."
            plot1_xlabel = "The silhouette coefficient values"
            plot1_ylabel = "Cluster label"

            plot2_title = "The visualization of the clustered data."
            # plot2_xlabel = "Feature space for the 1st feature"
            # plot2_ylabel = "Feature space for the 2nd feature"

            plot2_xlabel = "PC1"
            plot2_ylabel = "PC2"

            # plt.set_xlabel("PC1 (Å)")
            # plt.set_ylabel("PC2 (Å)")

            # plot_whole_titile = "Silhouette analysis for KMeans clustering on docking data with n_clusters = %d" % n_clusters
            plot_whole_titile = "Silhouette analysis for k-means clustering on docking data"
        else:
            plot1_title = "Силуэтный график для различных кластеров"
            plot1_xlabel = "Значение коэффициента силуэта"
            plot1_ylabel = "Маркировка кластеров"

            plot2_title = "Визуализация данных кластеризации"
            plot2_xlabel = "Пространство для 1-го признака"
            plot2_ylabel = " Пространство для 2-го признака"

            plot_whole_titile = "Силуэтный анализ на основе данных докинга, используя алгоритм k-средних = %d" % n_clusters

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)

        # This is part i need to modify
        # temp_x1 = X[self.analysis_dataframe]
        markers = list(self.analysis_dataframe[['Marker']].values.flatten())
        labels = list(self.analysis_dataframe[['Program']].values.flatten())
        marker_colors = list(self.analysis_dataframe[['MarkerColor']].values.flatten())
        # object_centroid = ax.scatter(X['component1'], X['component2'], marker=markers, s=80, lw=0, alpha=0.7,
        #                              c=colors)
        # object_centroid.set_zorder(1)

        # Plot individual docking program
        # for xp, yp, m, color, label in zip(X['component1'], X['component2'], markers, marker_colors, labels):
        #     object_centroid = ax.scatter(xp, yp, marker=m, s=80, lw=0, alpha=0.7,
        #                                  c=color, label=label)  # ,
        #     object_centroid.set_zorder(1)

        # plot rDock
        test = 1

        # X[self.analysis_dataframe[['Program']]=='rDock']
        if 'rDock' in load_data:
            rDock_index = self.analysis_dataframe['Program'] == 'rDock'
            final_indexes_rdock = rDock_index[rDock_index == True]
            dataframe = X.ix[final_indexes_rdock.index]

            rDock = ax.scatter(dataframe['component1'], dataframe['component2'], marker='v', s=50, lw=0, alpha=0.7,
                               c='r', label='rDock')
            rDock.set_zorder(1)

        if 'FlexAid' in load_data:
            flexAid_index = self.analysis_dataframe['Program'] == 'FlexAid'
            final_indexes_flexaid = flexAid_index[flexAid_index == True]
            dataframe = X.ix[final_indexes_flexaid.index]

            flexAid = ax.scatter(dataframe['component1'], dataframe['component2'], marker='s', s=50, lw=0, alpha=0.7,
                                 c='g', label='FlexAid')
            flexAid.set_zorder(2)

        if 'Vina' in load_data:
            vina_index = self.analysis_dataframe['Program'] == 'Vina'
            final_indexes_vina = flexAid_index[vina_index == True]
            dataframe = X.ix[final_indexes_vina.index]

            vina = ax.scatter(dataframe['component1'], dataframe['component2'], marker='o', s=50, lw=0, alpha=0.7,
                              c='b', label='Autodock Vina')
            vina.set_zorder(3)

        # TODO add if options to choose which to visualize
        if 'LeDock' in load_data:
            ledock_index = self.analysis_dataframe['Program'] == 'LeDock'
            final_indexes_ledock = ledock_index[ledock_index == True]
            dataframe = X.ix[final_indexes_ledock.index]

            vina = ax.scatter(dataframe['component1'], dataframe['component2'], marker='*', s=50, lw=0, alpha=0.7,
                              c='y', label='LeDock')
            vina.set_zorder(4)

        test = 1

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        # object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
        #                                      marker='o', c="white", alpha=1, s=50)
        # object_centroid_circles.set_zorder(2)
        #
        # for i, c in enumerate(centers):
        #     roman_number = extra_tools.write_roman(int(i + 1))
        #     object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=350,
        #                                               c='b')
        #     object_centroid_circles_text.set_zorder(3)

        # ax.set_title(plot2_title)

        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # LEGEND part

        # legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
        #           ncol=4,  shadow=True, fontsize = 'small',  prop={'size': 6})

        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
                  ncol=4,  shadow=True,   prop={'size': 14}, handletextpad=0.01)

        # frame = legend.get_frame()
        #
        # frame.set_linewidth(2)
        # frame.set_edgecolor("black")


        # for label in legend.get_texts():
        #     label.set_fontsize('small')

        # legend = ax.legend(loc='upper right', shadow=True,  fontsize = 'medium')
        #
        # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        #
        #
        #
        # handles, labels = ax.get_legend_handles_labels()
        #
        # def flip(items, ncol):
        #     return itertools.chain(*[items[i::ncol] for i in range(ncol)])
        #
        # plt.legend(flip(handles, 3), flip(labels, 3), loc=4, ncol=4)
        #
        #
        # frame = legend.get_frame()
        #
        # frame.set_linewidth(2)
        # frame.set_edgecolor("black")


        # frame.set_facecolor('0.90')

        # # # Set the fontsize

        #
        # for label in legend.get_lines():
        #     label.set_linewidth(1.5)  # the legend line width

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')



        # plt.rcParams['legend.numpoints'] = 1
        # legend = ax.legend(loc="lower left", markerscale=0.7, numpoints=1, fontsize=10)
        # TODO fix legend
        # red_line = matplotlib.lines.Line2D([], [], color='red', markersize=100, label='Blue line')
        #
        # blue_line = matplotlib.lines.Line2D([], [], color='blue', markersize=100, label='Green line')
        # purple_line = matplotlib.lines.Line2D([], [], color='purple', markersize=100, label='Green line')
        #
        # handles = [blue_line, red_line, purple_line]
        # labels = [h.get_label() for h in handles]
        #
        # ax.legend(handles=handles, labels=labels)

        # ax.grid('on')
        #
        # sns.set(style="white", context='paper', font_scale=1)

        plt.tight_layout()
        self.simulation_name = 'UBERDOCK_simple'

        fig.savefig(self.simulation_name + '_best_ultra_docking_PCA_analysis_v2' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_uber_docking_pca_simple_results(self, custom_dpi=1200, show_plot=False, lang='eng',
                                             trasparent_alpha=False, down_region=30,
                                             load_data=['LeDock', 'rDock', 'FlexAid', 'Vina']):

        # self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid,
        #                                       'dataOriginal':data_centroid,
        #                                       'clusterModels':centroid_cluster_models,
        #                                       'dataClustering': centroid_cluster_analysis,
        #                                       'clustNum':clust_num_centroid},
        #                          'reshape': {'data': temp_reshape,
        #                                      'dataOriginal': data_reshape,
        #                                      'clusterModels': reshape_cluster_models,
        #                                      'dataClustering': reshape_cluster_analysis,
        #                                      'clustNum': clust_num_reshape}}

        # import seaborn as sns
        sns.set(style="white", context='paper', font_scale=1)

        centroid_data = self.ultra_clustering['centroid']

        if centroid_data['overrideClustNum'] is not None:
            n_clusters = centroid_data['overrideClustNum']
        else:
            n_clusters = centroid_data['clustNum']

        cluster_labels = centroid_data['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old version
        centers = centroid_data['dataClustering']['clusterInfo'][n_clusters]['centers']

        # colors = centroid_data['dataClustering']['colorData']
        # self.colors_ = colors
        #
        # self.cluster_colors = centroid_data['dataClustering']['rgbColors']

        colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['colorData']

        self.colors_ = colors

        self.cluster_colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['rgbColors']

        X = centroid_data['dataPCA']

        # New Version
        centers = calculate_cluster_center_docking(X, cluster_labels)

        test = 1
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots()
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))
        # fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        if lang == 'eng':
            plot1_title = "The silhouette plot for the various clusters."
            plot1_xlabel = "The silhouette coefficient values"
            plot1_ylabel = "Cluster label"

            plot2_title = "The visualization of the clustered data."
            # plot2_xlabel = "Feature space for the 1st feature"
            # plot2_ylabel = "Feature space for the 2nd feature"

            plot2_xlabel = "PC1"
            plot2_ylabel = "PC2"

            # plt.set_xlabel("PC1 (Å)")
            # plt.set_ylabel("PC2 (Å)")

            # plot_whole_titile = "Silhouette analysis for KMeans clustering on docking data with n_clusters = %d" % n_clusters
            plot_whole_titile = "Silhouette analysis for k-means clustering on docking data"
        else:
            plot1_title = "Силуэтный график для различных кластеров"
            plot1_xlabel = "Значение коэффициента силуэта"
            plot1_ylabel = "Маркировка кластеров"

            plot2_title = "Визуализация данных кластеризации"
            plot2_xlabel = "Пространство для 1-го признака"
            plot2_ylabel = " Пространство для 2-го признака"

            plot_whole_titile = "Силуэтный анализ на основе данных докинга, используя алгоритм k-средних = %d" % n_clusters

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)

        # This is part i need to modify
        # temp_x1 = X[self.analysis_dataframe]
        markers = list(self.analysis_dataframe[['Marker']].values.flatten())
        labels = list(self.analysis_dataframe[['Program']].values.flatten())
        marker_colors = list(self.analysis_dataframe[['MarkerColor']].values.flatten())
        # object_centroid = ax.scatter(X['component1'], X['component2'], marker=markers, s=80, lw=0, alpha=0.7,
        #                              c=colors)
        # object_centroid.set_zorder(1)

        # Plot individual docking program
        # for xp, yp, m, color, label in zip(X['component1'], X['component2'], markers, marker_colors, labels):
        #     object_centroid = ax.scatter(xp, yp, marker=m, s=80, lw=0, alpha=0.7,
        #                                  c=color, label=label)  # ,
        #     object_centroid.set_zorder(1)

        # plot rDock
        test = 1

        # X[self.analysis_dataframe[['Program']]=='rDock']
        if 'rDock' in load_data:
            rDock_index = self.analysis_dataframe['Program'] == 'rDock'
            final_indexes_rdock = rDock_index[rDock_index == True]
            dataframe = X.ix[final_indexes_rdock.index]

            rDock = ax.scatter(dataframe['component1'], dataframe['component2'], marker='v', s=80, lw=0, alpha=0.7,
                               c='r', label='rDock')
            rDock.set_zorder(1)

        if 'FlexAid' in load_data:
            flexAid_index = self.analysis_dataframe['Program'] == 'FlexAid'
            final_indexes_flexaid = flexAid_index[flexAid_index == True]
            dataframe = X.ix[final_indexes_flexaid.index]

            flexAid = ax.scatter(dataframe['component1'], dataframe['component2'], marker='s', s=80, lw=0, alpha=0.7,
                                 c='g', label='FlexAid')
            flexAid.set_zorder(2)

        if 'Vina' in load_data:
            vina_index = self.analysis_dataframe['Program'] == 'Vina'
            final_indexes_vina = flexAid_index[vina_index == True]
            dataframe = X.ix[final_indexes_vina.index]

            vina = ax.scatter(dataframe['component1'], dataframe['component2'], marker='o', s=80, lw=0, alpha=0.7,
                              c='b', label='Autodock Vina')
            vina.set_zorder(3)

        # TODO add if options to choose which to visualize
        if 'LeDock' in load_data:
            ledock_index = self.analysis_dataframe['Program'] == 'LeDock'
            final_indexes_ledock = ledock_index[ledock_index == True]
            dataframe = X.ix[final_indexes_ledock.index]

            vina = ax.scatter(dataframe['component1'], dataframe['component2'], marker='*', s=80, lw=0, alpha=0.7,
                              c='y', label='LeDock')
            vina.set_zorder(4)

        test = 1

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        # object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
        #                                      marker='o', c="white", alpha=1, s=50)
        # object_centroid_circles.set_zorder(2)
        #
        # for i, c in enumerate(centers):
        #     roman_number = extra_tools.write_roman(int(i + 1))
        #     object_centroid_circles_text = ax.scatter(c[0], c[1], marker='$%s$' % roman_number, alpha=1, s=350,
        #                                               c='b')
        #     object_centroid_circles_text.set_zorder(3)

        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # LEGEND part
        legend = ax.legend(loc='lower left', shadow=True)

        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        fig.tight_layout()

        # plt.rcParams['legend.numpoints'] = 1
        # legend = ax.legend(loc="lower left", markerscale=0.7, numpoints=1, fontsize=10)
        # TODO fix legend
        # red_line = matplotlib.lines.Line2D([], [], color='red', markersize=100, label='Blue line')
        #
        # blue_line = matplotlib.lines.Line2D([], [], color='blue', markersize=100, label='Green line')
        # purple_line = matplotlib.lines.Line2D([], [], color='purple', markersize=100, label='Green line')
        #
        # handles = [blue_line, red_line, purple_line]
        # labels = [h.get_label() for h in handles]
        #
        # ax.legend(handles=handles, labels=labels)

        ax.grid('on')

        sns.set(style="white", context='paper', font_scale=1)

        self.simulation_name = 'LASR_HSL_UBERDOCK_simple'

        fig.savefig(self.simulation_name + '_best_ultra_docking_PCA_analysis' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()
















    def override_clust_num(self, clust_num, selection='centroid'):
        print('-------------------------------------------------------------------------------\n')
        # Set custom number of cluster for selection
        self.ultra_clustering[selection]['overrideClustNum'] = clust_num
        print('Set custom number of clusters {0} for selection {1}'.format(clust_num, selection))
        print('-------------------------------------------------------------------------------\n')

    @hlp.timeit
    def cluster_analysis_custom(self, custom, type, show=False, algorithm='kmeans', parallel=False, num_of_threads=7):

        test = 1
        if parallel is True:
            self.parallel_cluster_proc = []

            # self.simultaneous_run = list(range(0, num_of_threads))

            pool = multiprocessing.Pool(num_of_threads)

            # range_n_clusters = list(range(1, 11))
            k_neighb = 25

            function_arguments_to_call = [
                [x, custom, None, algorithm, k_neighb, 'docking'] for x in
                self.range_n_clusters]

            test = 1

            # results = pool.starmap(parallel_md_silhouette_analysis_pca,self.range_n_clusters,self.reduced_cartesian,
            #                                                 self.pca_traj.time, algorithm )
            results = pool.starmap(parallel_data_cluster_analysis, function_arguments_to_call)

            # d = [x for x in results]
            # d = [list(x.keys())[0] for x in results]

            # d = {key: value for (key, value) in iterable}
            # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            clusters_info = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            test = 1

            pool.close()

            # self.parallel_cluster_proc.append(proc)
            # proc.start()# start now
        elif parallel is False:

            clusters_info = md_silhouette_analysis_pca(custom,
                                                       None,
                                                       range_n_clusters=self.range_n_clusters,
                                                       show_plots=show,
                                                       algorithm=algorithm,
                                                       data_type='docking',
                                                       type=type)

        # sil_pca = self.extract_info_cluster_data(clusters_info, 'silhouette')
        # calinski_pca = self.extract_info_cluster_data(clusters_info, 'calinski')
        # dunn_pca = self.extract_info_cluster_data(clusters_info, 'dunn')
        # dbi_pca = self.extract_info_cluster_data(clusters_info, 'dbi')
        #
        # book_dbi_pca = self.extract_info_cluster_data(clusters_info, 'book_dbi')
        # self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()

        # clust_num = self.select_number_of_clusters(cluster_info)
        # Version 2
        # clust_num, clust_analysis_info = select_number_of_clusters_v2(clusters_info,
        #                                                               self.range_n_clusters)

        # Version 3

        clust_num, clust_analysis_info = select_number_of_clusters_v2_docking(clusters_info, type,
                                                                              self.range_n_clusters)

        test = 1

        # cluster_models = self.collect_cluster_info()

        # Backup

        # TODO color part for ultra docking analysis

        if type == 'centroid':
            colors = sns.cubehelix_palette(n_colors=clust_num, rot=.5, dark=0, light=0.85)
        elif type == 'reshape':
            colors = sns.cubehelix_palette(n_colors=clust_num, start=2.8, rot=.1)

        colors_rgb = converters.convert_seaborn_color_to_rgb(colors)

        colors_ = colors
        cluster_labels = clusters_info[clust_num]['labels']
        colors_data = converters.convert_to_colordata(cluster_labels, colors)
        cluster_colors = colors_data

        data_to_return = {
            'colors': colors,
            'rgbColors': colors_rgb,
            'labels': cluster_labels,
            'colorData': cluster_colors,
            'clusterInfo': clusters_info,
            'clustNum': clust_num,
            'clusterAnalysisInfo': clust_analysis_info,
            'overrideClustNum': None}

        test = 1

        import gc
        gc.collect()

        return data_to_return

    @hlp.timeit
    def cluster_analysis_hdbscan(self, custom, type, show=False, algorithm='kmeans', parallel=False, num_of_threads=7):

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # TODO TSNE random not reproducible enough but hdbscan works
        print('HDB SCAN TIME ')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=60).fit(custom)
        color_palette = sns.color_palette('Paired', 12)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]

        projection = TSNE().fit_transform(custom)
        plt.scatter(*projection.T, s=100, linewidth=0, c=cluster_member_colors, alpha=0.25)

        # plt.show()
        # clust_num, clust_analysis_info = select_number_of_clusters_v2_docking(clusters_info, type,
        #                                                                       self.range_n_clusters)
        #
        # test = 1
        #
        # # cluster_models = self.collect_cluster_info()
        #
        # # Backup
        #
        # # TODO color part for ultra docking analysis
        #
        # if type == 'centroid':
        #     colors = sns.cubehelix_palette(n_colors=clust_num, rot=.5, dark=0, light=0.85)
        # elif type == 'reshape':
        #     colors = sns.cubehelix_palette(n_colors=clust_num, start=2.8, rot=.1)
        #
        # colors_rgb = converters.convert_seaborn_color_to_rgb(colors)
        #
        # colors_ = colors
        # cluster_labels = clusters_info[clust_num]['labels']
        # colors_data = converters.convert_to_colordata(cluster_labels, colors)
        # cluster_colors = colors_data
        #
        # data_to_return = {
        #     'colors': colors,
        #     'rgbColors': colors_rgb,
        #     'labels': cluster_labels,
        #     'colorData': cluster_colors,
        #     'clusterInfo': clusters_info,
        #     'clustNum': clust_num,
        #     'clusterAnalysisInfo': clust_analysis_info,
        #     'overrideClustNum': None}
        #
        # test = 1
        #
        # import gc
        # gc.collect()
        #
        # return data_to_return

    # TODO fix parallel implementation
    @hlp.timeit
    def cluster_analysis_pca(self, show=False, algorithm='kmeans', parallel=False, num_of_threads=7):

        test = 1
        if parallel is True:
            self.parallel_cluster_proc = []

            # self.simultaneous_run = list(range(0, num_of_threads))

            pool = multiprocessing.Pool(num_of_threads)

            # range_n_clusters = list(range(1, 11))
            k_neighb = 25

            function_arguments_to_call = [
                [x, self.pca_data, None, algorithm, k_neighb, 'docking'] for x in
                self.range_n_clusters]

            test = 1

            # results = pool.starmap(parallel_md_silhouette_analysis_pca,self.range_n_clusters,self.reduced_cartesian,
            #                                                 self.pca_traj.time, algorithm )
            results = pool.starmap(parallel_data_cluster_analysis, function_arguments_to_call)

            # d = [x for x in results]
            # d = [list(x.keys())[0] for x in results]

            # d = {key: value for (key, value) in iterable}
            # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            self.clusters_info = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            test = 1

            # self.parallel_cluster_proc.append(proc)
            # proc.start()# start now
        elif parallel is False:

            self.clusters_info = md_silhouette_analysis_pca(self.pca_data,
                                                            None,
                                                            range_n_clusters=self.range_n_clusters,
                                                            show_plots=show,
                                                            algorithm=algorithm,
                                                            data_type='docking')

        self.sil_pca = self.extract_info_cluster_data(self.clusters_info, 'silhouette')
        self.calinski_pca = self.extract_info_cluster_data(self.clusters_info, 'calinski')
        self.dunn_pca = self.extract_info_cluster_data(self.clusters_info, 'dunn')
        self.dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'dbi')

        self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')
        # self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()

        # Version 1
        # self.clust_num = self.select_number_of_clusters(self.clusters_info)

        # Version 2
        # self.clust_num, self.clust_analysis_info = select_number_of_clusters_v2(self.clusters_info,
        #                                                               self.range_n_clusters)

        # Version 3
        test = 1
        type = 'centroid'

        self.clust_num, self.clust_analysis_info = select_number_of_clusters_v2_docking(self.clusters_info, type,
                                                                                        self.range_n_clusters)

        self.cluster_models = self.collect_cluster_info()

        # Backup
        colors = sns.cubehelix_palette(n_colors=self.clust_num, rot=.7, dark=0, light=0.85)
        self.colors_ = colors
        cluster_labels = self.clusters_info[self.clust_num]['labels']
        colors_data = converters.convert_to_colordata(cluster_labels, colors)
        self.cluster_colors = colors_data

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
