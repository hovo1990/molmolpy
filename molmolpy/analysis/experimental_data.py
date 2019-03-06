# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    experimental_data
# @brief   Experimental Data Analysis class
# @author  Hovakim Grabski
#
# <!--------------------------------------------------------------------------
#
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

import pickle
import json
import subprocess as sub

# from multiprocessing import Pool
import multiprocessing

import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from sklearn import mixture

import mdtraj as md

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils

from molmolpy.utils import run_tools
from molmolpy.utils import helper as hlp
from molmolpy.utils import filter_items

from scipy import linalg
from pandas import HDFStore, DataFrame
import matplotlib as mpl

import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from sklearn.decomposition import PCA

from sklearn import mixture
from multiprocessing import Pool

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import protein_analysis
from molmolpy.utils import nucleic_analysis

from molmolpy.utils import helper as hlp

from itertools import combinations

import seaborn as sns

import numba

matplotlib.rcParams.update({'font.size': 12})
# matplotlib.style.use('ggplot')
sns.set(style="white", context='paper')


class ExperimentalDataAnalysisObject(object):
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
                 load_pickle_file=None,
                 experimental_data=None,
                 experimantal_data_name=None,
                 folder_path='.',
                 jobs=100,
                 k_clust=10):

        self.load_pickle_file = load_pickle_file

        if load_pickle_file is not None:
            self.load_rosetta_pickle_data(self.load_pickle_file)
        elif experimantal_data_name is not None:
            print('ExperimentalDataAnalysisObject has been created')

            sns.set(font_scale=2)

            # self.rosetta_dock_folder = rosetta_dock_folder
            #
            # self.directories = [self.rosetta_dock_folder]
            #
            # self.folder_path = folder_path

            self.experimantal_data_name = experimantal_data_name

            # self.state_data = {}
            #
            # self.state_data_samples = {}
            #
            # self.samples_fasc_files = sorted(self.obtain_samples())
            #
            # self.fasc_file_data = self.load_fasc_samples()

            self.mdtraj= md.load(experimental_data)

            self.full_traj = self.mdtraj

            # self.joined_md_trajs = self.join_mdtraj_traj()

            self.k_clust = k_clust

            self.range_n_clusters = list(range(2, k_clust + 1))

            self.experimental_data_analysis_name = 'experimental_data_analysis' + '_' + experimantal_data_name


            # self.save_rosetta_pickle_data()

            test = 1
        else:
            print('Oh come on buddy give me something !!!!!!!!!!!!!')

            # original data before transformation

            ####################################################################################################################

            # need to find a way to extract models correctrly

    def save_rosetta_pickle_data(self):
        print('Saving Rosetta Pickle File!!!!!!!\n')
        save_data = self.simulation_name + '.pickle'
        self.rosetta_analysis_object.update({'rosettaPickleName': save_data})

        filehandler = open("{0}".format(save_data), "wb")
        pickle.dump(self.rosetta_analysis_object, filehandler)
        filehandler.close()

        print('Rosetta pickle saved for faster load next time !!!!')
        print('---------------------------------------------------\n')

        # # TODO save to HDF format
        #
        # hdf = HDFStore("{0}_hdf.h5".format(save_data))
        #
        # hdf.put('sparta', self.nmr_shifts_sparta)
        # hdf.put('shiftx2', self.nmr_shifts_shift2x)
        #
        # print('Sparta')
        # print(hdf['sparta'].shape)
        # print('Shiftx2')
        # print(hdf['shiftx2'].shape)
        # hdf.close()  # closes the file

    # TODO should I add json saving of information or not?
    def load_rosetta_pickle_data(self, filename):
        '''

        :param filename: load json state data
        :return:
        '''
        print('Loading pickle file, this should be a lot faster !!!!!!!!!!!')
        print('----------------------------------------------------------\n')

        # self.absolute_path = os.path.abspath(filename)
        self.load_state_called = True

        print(os.path.abspath(__file__))

        file = open(filename, 'rb')
        self.rosetta_analysis_object = pickle.load(file)
        file.close()

        self.rosetta_dock_folder = self.rosetta_analysis_object['rosettaFolder']
        self.folder_path = self.rosetta_analysis_object['folderPath']

        self.rosetta_sim_name = self.rosetta_analysis_object['rosettaSimName']
        self.simulation_name = self.rosetta_analysis_object['simulationName']
        self.range_n_clusters = self.rosetta_analysis_object['clusterQuantity']
        self.samples_fasc_files = self.rosetta_analysis_object['fascFiles']

        self.fasc_file_data = self.rosetta_analysis_object['fascData']

        self.full_traj = self.rosetta_analysis_object['rosettaConformations']

        self.full_score = self.rosetta_analysis_object['fullScore']
        self.full_rmsd = self.rosetta_analysis_object['fullRMSD']
        self.full_fa_atr = self.rosetta_analysis_object['fa_atr']
        self.full_fa_rep = self.rosetta_analysis_object['fa_rep']
        self.full_fa_sol = self.rosetta_analysis_object['fa_sol']
        self.full_fa_elec = self.rosetta_analysis_object['fa_elec']
        self.full_fa_pair = self.rosetta_analysis_object['fa_pair']
        self.full_hbond_sr_bb = self.rosetta_analysis_object['hbond_sr_bb']
        self.full_hbond_lr_bb = self.rosetta_analysis_object['hbond_lr_bb']
        self.full_hbond_bb_sc = self.rosetta_analysis_object['hbond_bb_sc']
        self.full_hbond_sc = self.rosetta_analysis_object['hbond_sc']
        self.full_dslf_ss_dst = self.rosetta_analysis_object['dslf_ss_dst']
        self.full_dslf_cs_ang = self.rosetta_analysis_object['dslf_cs_ang']
        self.full_dslf_ss_dih = self.rosetta_analysis_object['dslf_ss_dih']
        self.full_dslf_ca_dih = self.rosetta_analysis_object['dslf_ca_dih']
        self.full_fa_dun = self.rosetta_analysis_object['fa_dun']
        self.full_samples = self.rosetta_analysis_object['fullSamples']

        print('Finished loading Rosetta pickle data\n')

        test = 1



        # TODO this part needs to be thought out

    @hlp.timeit
    def export_cluster_models(self,
                              selection_obj='protein',
                              select_lig=None,
                              save_data=False, nth_frame=1):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''

        print('Exporting cluster models\n')
        n_clusters = self.select_number_of_clusters()

        cluster_labels = self.clusters_info[n_clusters]['labels']
        labels = cluster_labels

        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        unique_labels = list(set(cluster_labels))
        print('Unique labels ', unique_labels)

        original_data = self.full_traj

        self.clusterized_data = {}

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                cluster_data = []
                xyz = original_data[labels == k]

                # sel_traj = xyz[:]

                topology = xyz.topology

                selection_name = selection_obj
                selection_final_name = selection_obj
                selection = topology.select(selection_obj)
                selection_final = selection

                if select_lig is not None:
                    # selection1 = topology.select(select_lig)
                    # selection_final  = np.concatenate((selection, selection1))
                    # selection_name  = selection_name + ' and ' + select_lig
                    #
                    # selection_final = list(topology.select(selection_obj)) + list(topology.select(select_lig))
                    selection_final_name = selection_obj + '+' + select_lig
                    selection_final = topology.select(selection_obj + ' or ' + select_lig)

                # list(topology.select(selection_obj)) + list(topology.select(select_lig))

                sel_traj = xyz.atom_slice(atom_indices=selection_final)
                # sel_traj.restrict_atoms(selection_final)
                clust_num = int(k) + 1
                if save_data is True:
                    temp_data = sel_traj[::nth_frame]
                    temp_data[0].save(self.simulation_name + '_' + 'cluster_' + str(
                        clust_num) + '_' + selection_final_name + '_frame_0.pdb')
                    temp_data.save(
                        self.simulation_name + '_' + 'cluster_' + str(clust_num) + '_' + selection_final_name + '.pdb')
                self.clusterized_data.update({k: sel_traj})

        self.save_pdb_hbond = True
        print('---------------------------------------------')

    @hlp.timeit
    def find_clusters_centroid(self):

        print('Find Clusters centroids is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_centroid = True

        self.clusters_centroids = []
        self.clusters_centroids_dict = {}

        for k in self.clusterized_data:
            print('Finding centroid for cluster {0}'.format(k))
            clust_temp_data = self.clusterized_data[k]
            print('Curr cluster temp ', clust_temp_data)

            atom_indices = [a.index for a in clust_temp_data.topology.atoms if a.element.symbol != 'H']
            distances = np.empty((clust_temp_data.n_frames, clust_temp_data.n_frames))
            for i in range(clust_temp_data.n_frames):
                distances[i] = md.rmsd(clust_temp_data, clust_temp_data, i, atom_indices=atom_indices)

            beta = 1
            index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
            print(index)

            curr_score = 'Centroid score is {0}'.format(self.transformed_data_pandas['score'].iloc[index])
            print(curr_score)

            centroid = clust_temp_data[index]
            # self.centroid_conf = centroid
            # print(centroid)

            # self.centroid_conf = centroid
            self.clusters_centroids.append(centroid)

            self.clusters_centroids_dict.update({str(k): centroid})

            centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(k))

        print('-----------------------------------\n')

    # count per residue
    def count_inter_prot_hbond(self, t, hbonds, residue):
        label = lambda hbond: '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))

        # res.residue.chain.index

        hbond_atoms = []
        hbond_indexes_sel = []
        hbond_count = 0
        for hbond in hbonds:
            res = label(hbond)

            atom1 = t.topology.atom(hbond[0])
            atom2 = t.topology.atom(hbond[2])

            atom1_chain_index = atom1.residue.chain.index
            atom2_chain_index = atom2.residue.chain.index

            if atom1_chain_index != atom2_chain_index:
                print('res ', res)
                hbond_atoms.append(res)
                hbond_indexes_sel.append(hbond)
                hbond_count += 1
                test = 1
        # print('------------------------------------------------')
        test = 1
        return hbond_atoms, hbond_count, hbond_indexes_sel

    @hlp.timeit
    def find_clusters_centroid_hbonds(self,
                                      receptor_name='LasR',
                                      residue_name='HSL',
                                      title='Simulation',
                                      xlabel=r"Time $t$ (ns)",
                                      ylabel=r"Number of Hydrogen Bonds",
                                      lang='en',
                                      hbond_length=0.4,
                                      custom_dpi=600):

        print('Find Clusters centroids H-Bonds is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        for k in self.clusters_centroids_dict:
            print('Finding H-Bonds for cluster {0}'.format(k))
            clust_temp_data = self.clusters_centroids_dict[k]

            # self.sim_time = self.full_traj.time / 1000

            # paral = Pool(processes=16)
            # data_count = list(map(self.hbond_frame_calc, self.full_traj))
            #
            # print('data count ',data_count)

            # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
            # print('count of hbonds is ', len(hbonds))


            title = 'Frequency of H-Bonds between {0}-{1} in Cluster {2}'.format(receptor_name, residue_name, k)

        # self.hbond_count.append(len(hbonds))
        # hbonds_frames = md.wernet_nilsson(clust_temp_data, exclude_water=True, periodic=False)
        hbonds_frames = md.baker_hubbard(clust_temp_data, exclude_water=True, periodic=False)

        sim_hbond_atoms = []
        sim_hbond_count = []

        sim_hbond_sel = []

        # for hbonds in hbonds_frames:
        #     hbond_atoms, hbond_count, hbond_indexes_sel =  self.count_inter_prot_hbond(clust_temp_data, hbonds, residue_name)
        #
        #     sim_hbond_atoms.append(hbond_atoms)
        #     sim_hbond_count.append(hbond_count)
        #
        #     if len(hbond_indexes_sel) > 0:
        #         sim_hbond_sel += hbond_indexes_sel


        hbond_atoms, hbond_count, hbond_indexes_sel = self.count_inter_prot_hbond(clust_temp_data, hbonds_frames,
                                                                                  residue_name)

        sim_hbond_atoms.append(hbond_atoms)
        sim_hbond_count.append(hbond_count)

        if len(hbond_indexes_sel) > 0:
            sim_hbond_sel += hbond_indexes_sel

        sim_hbound_np = np.array(sim_hbond_count)

        # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

        sim_hbound_sel_np = np.array(sim_hbond_sel)
        # sim_hbound_sel_np = np.array(updated_indexes)


        # TODO this part is very important, for removing duplicates
        if len(sim_hbound_sel_np) > 0:
            unique_bonds_list = sim_hbound_sel_np[:, [0, 2]]
            unique_bonds_analysis = filter_items.filter_similar_lists(unique_bonds_list)

            unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)


        print('-----------------------------------\n')

    @hlp.timeit
    def compute_contacts(self,
                       receptor_name='LasR',
                       residue_name='HSL',
                       title='Simulation_contactMaps',
                       xlabel=r"Time $t$ (ns)",
                       ylabel=r"Number of Hydrogen Bonds",
                       lang='en',
                       distance_length=0.4,
                       distance_length_repulsion=2,
                       custom_dpi=600):

        print('Find Clusters centroids H-Bonds is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # TODO for next time

        self.contact_maps_dict = {}

        print('Compute residue distances')
        clust_temp_data = self.full_traj

        # self.sim_time = self.full_traj.time / 1000

        # paral = Pool(processes=16)
        # data_count = list(map(self.hbond_frame_calc, self.full_traj))
        #
        # print('data count ',data_count)

        # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
        # print('count of hbonds is ', len(hbonds))

        group_1 = list(range(0, 163))
        group_2 = list(range(163, 326))
        pairs = list(itertools.product(group_1, group_2))
        # print(pairs)

        # select from topology atoms
        # test_top.select('chainid 1')

        # clust_temp_data.topology.chains
        # list(clust_temp_data.topology.chains)[0]

        # list(clust_temp_data.topology.residues)[0:239]

        # list(clust_temp_data.topology.residues)

        # list(clust_temp_data.topology.residues)
        test = 1

        # This is quite fast
        contacts = md.compute_contacts(clust_temp_data, pairs, scheme='closest-heavy')

        # test = distances[0][distances[0]<0.4]



        distances = contacts[0]
        indexes = contacts[1]

        distances_final = distances[0][distances[0] < distance_length]
        indexes_final = contacts[1][distances[0] < distance_length]

        distances_final_repulsion = distances[0][distances[0] > distance_length_repulsion]
        indexes_final_repulsion = contacts[1][distances[0] > distance_length_repulsion]


        # distances_final = distances[0][distances[0] > 0.0]
        # indexes_final = contacts[1][distances[0] > 0.0]

        contact_maps = md.geometry.squareform(contacts[0], contacts[1])

        self.contact_maps_dict.update({'0': {'contacts': contacts,
                                           'distancesFinal': distances_final,
                                           'indexes_final': indexes_final,
                                           'distanceCutoff': distance_length,
                                           'calcPairs': pairs,
                                           'group1': group_1,
                                           'group2': group_2}})

        test = 1
        # plt.clf()
        # sns_heatmap = sns.heatmap(contact_maps[0])
        #
        # sns_heatmap.figure.savefig(self.simulation_name + '_' + title + '_' + 'cluster:{0}'.format(k) + '.png',
        #                            dpi=custom_dpi)
        # sns.plt.show()

        label = lambda comp_dist: '%s -- %s ' % (
            clust_temp_data.topology.residue(comp_dist[0]), clust_temp_data.topology.residue(comp_dist[1]))


        print('Distances attraction')
        for i in range(len(indexes_final)):
            print(indexes_final[i])
            res = label(indexes_final[i])
            print(res + ' dist: {0}'.format(distances_final[i]))
            print('---------------------------------------------')

        print('-------------------\n')

        set1_indexes = indexes_final[:,0]


        set2_indexes = indexes_final[:,1]


        print('__________RECEPTOR______________________________')
        set1 = set(set1_indexes)
        residues_set1 = sorted(self.get_residue_numbers(set1))
        residue_set1_text = self.write_cluspro(residues_set1)
        residue_set1_text = self.write_patchdock(residues_set1)

        print('_____LIGAND____________________________________')

        set2 = set(set2_indexes)
        residues_set2 = sorted(self.get_residue_numbers(set2))
        residue_set2_text = self.write_cluspro(residues_set2)
        residue_set2_text = self.write_patchdock(residues_set2)

        test = 1






        print('Residue Set 1 ')
        print(residue_set1_text)
        print('------\n')


        print('Residue Set 2 ')
        print(residue_set2_text)
        print('------\n')


        print('Common elements in set1 and set2')

        common_elements =  list(set(residues_set1).intersection(residues_set2))
        residue_common_text = self.write_cluspro(common_elements)
        print(residue_common_text)

        residue_common_text= self.write_patchdock(common_elements)

        test=1

        print('========================Repulsion==============================')

        # print('Distances repulsion')
        # for i in range(len(indexes_final_repulsion)):
        #     print(indexes_final_repulsion[i])
        #     res = label(indexes_final_repulsion[i])
        #     print(res + ' dist: {0}'.format(indexes_final_repulsion[i]))
        #     print('---------------------------------------------')
        #
        # print('-------------------\n')


        set1_indexes = indexes_final_repulsion[:,0]


        set2_indexes = indexes_final_repulsion[:,1]

        set1 = set(set1_indexes)
        residues_set1 = sorted(self.get_residue_numbers(set1))
        residue_set1_text = self.write_cluspro(residues_set1)
        residue_set1_text = self.write_patchdock(residues_set1)

        set2 = set(set2_indexes)
        residues_set2 = sorted(self.get_residue_numbers(set2))
        residue_set2_text = self.write_cluspro(residues_set2)
        residue_set2_text = self.write_patchdock(residues_set2)

        test = 1


        print('Residue Set 1 ')
        print(residue_set1_text)
        print('------\n')


        print('Residue Set 2 ')
        print(residue_set2_text)
        print('------\n')


        test = 1

        test = 1
        return self.contact_maps_dict
        print('-----------------------------------\n')

    def get_residue_numbers(self, index_set):
        topology = self.full_traj.topology

        index_list = list(index_set)

        full_residue_names = []

        for index in index_list:
            residue = topology.residue(index)
            full_residue_names.append(residue.resSeq)
            test = 1

        return full_residue_names

    def write_cluspro(self, residues):
        string_text = ''
        for res in residues:
            string_text += 'A-{0} '.format(res)
        return string_text


    def write_patchdock(self, residues):
        string_text = ''
        for res in residues:
            string_text += '{0} A\n'.format(res)
        return string_text


    @hlp.timeit
    def contact_maps_analysis(self, show=False, algorithm='kmeans', parallel=True, num_of_threads=7, k_clust_count=20):

        self.max_cluster = -1
        self.max_cluster_index = 0

        print('Max Cluster index is ', self.max_cluster_index)
        test = 1

        self.range_n_clusters_contact = list(range(2, k_clust_count + 1))

        object_to_work = self.contact_maps_dict[str(self.max_cluster_index)]
        self.object_to_work = object_to_work

        indexes = object_to_work['indexes_final']
        distances = object_to_work['distancesFinal']

        if parallel is True:
            # self.simultaneous_run = list(range(0, num_of_threads))

            pool = multiprocessing.Pool(num_of_threads)

            # range_n_clusters = list(range(1, 11))
            k_neighb = 25

            function_arguments_to_call = [[x, indexes, None, algorithm, k_neighb] for x in
                                          self.range_n_clusters_contact]

            test = 1

            # results = pool.starmap(parallel_md_silhouette_analysis_pca,self.range_n_clusters,self.reduced_cartesian,
            #                                                 self.pca_traj.time, algorithm )
            results = pool.starmap(parallel_data_cluster_analysis, function_arguments_to_call)

            # d = [x for x in results]
            # d = [list(x.keys())[0] for x in results]

            # d = {key: value for (key, value) in iterable}
            # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            self.clusters_info_contact = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            test = 1

        self.contact_maps_clust_num, self.contact_maps_clust_info = select_number_of_clusters(
            self.clusters_info_contact, self.range_n_clusters_contact)

        test = 1
        # self.cluster_list = self.collect_cluster_info()

    @hlp.timeit
    def indexes_contact_maps_plot(self, cust_clust=None):
        title = self.simulation_name + '_' + 'contact_maps_indexes_analysis_md_{0}'.format(str(cust_clust)) + '.png'

        plot_tools.show_best_data_analysis(self.object_to_work,
                                           self.clusters_info_contact,
                                           self.contact_maps_clust_info,
                                           cust_clust=cust_clust,
                                           show_plot=False,
                                           title=title)

        test = 1

    @hlp.timeit
    def indexes_cluster_quality_plot(self):
        title = self.simulation_name + '_' + 'contact_maps_indexes_cluster_quality_analysis_md' + '.png'

        plot_tools.show_data_cluster_analysis_plots(
                                           self.contact_maps_clust_info,
                                           self.range_n_clusters_contact,
                                           show_plot=False,
                                           title=title)

        test = 1

    @hlp.timeit
    def plot_cluster_maps(self, k_clust=None, title='LasR Dimer Contact Map', custom_dpi=300):
        object_to_work = self.contact_maps_dict[str(self.max_cluster_index)]
        # self.object_to_work = object_to_work

        if k_clust is None:
            k_clust = self.contact_maps_clust_num
        # else:
        #     k_clust = k_clust

        centroid = self.clusters_centroids_dict[str(self.max_cluster_index)]

        indexes = object_to_work['indexes_final']
        print('Total pairs of contacts less than {0}'.format(len(indexes)))


        # TODO BUG is located here
        distances = object_to_work['distancesFinal']
        contacts = object_to_work['contacts']
        contact_maps = md.geometry.squareform(contacts[0], contacts[1])

        contact_maps_reshape = contact_maps.reshape(contact_maps.shape[1:])
        contact_maps_pandas = pd.DataFrame(contact_maps_reshape)

        cluster_info = self.clusters_info_contact[k_clust]

        test = 1

        cluster_labels = self.clusters_info_contact[k_clust]['labels']
        labels = cluster_labels

        sample_silhouette_values = self.clusters_info_contact[k_clust]['silhouette_values']
        silhouette_avg = self.clusters_info_contact[k_clust]['silhouette']

        centers = self.clusters_info_contact[k_clust]['centers']

        unique_labels = list(set(cluster_labels))
        print('Unique labels ', unique_labels)

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                cluster_data = []
                index_to_work = indexes[labels == k]
                distances_to_work = distances[labels == k]

                #
                #
                # max(index_to_work[:,1])
                # min(index_to_work[:,1])

                min_index0 =  min(index_to_work[:,0])
                max_index0 = max(index_to_work[:,0])
                dif_index0 = max_index0- min_index0

                min_index1 =  min(index_to_work[:,1])
                max_index1 = max(index_to_work[:,1])
                dif_index1 = max_index1- min_index1

                max_dif = max(index_to_work[:,0])-min(index_to_work[:,0])

                start_index0 = min_index0-1
                end_index0 = min_index0 + max_dif + 1

                start_index1 = min_index1-1
                end_index1 = min_index1 + max_dif + 1

                contact_maps_slice = contact_maps[start_index0:end_index0, start_index1:end_index1]


                # a[start_index0:end_index0, start_index1:end_index1]

                print('Cluster contact maps {0}'.format(k))
                print('Number of contacts {0}'.format(str(len(index_to_work))))

                label = lambda comp_dist: '%s -- %s ' % (
                    centroid.topology.residue(comp_dist[0]), centroid.topology.residue(comp_dist[1]))


                # TODO HERE was the bug
                for i in range(len(index_to_work)):
                    print(index_to_work[i])
                    res = label(index_to_work[i])
                    print(res + ' dist: {0}'.format(distances_to_work[i]))

                print('---------------------------------------------')



                indexes0 = []
                for i in range(start_index0,end_index0+1):
                    indexes0.append(centroid.topology.residue(i))


                indexes1 = []
                for i in range(start_index1,end_index1+1):
                    indexes1.append(centroid.topology.residue(i))

                # TODO huge bug here
                if len(index_to_work)>10:
                    plt.clf()
                    submatrix = contact_maps_pandas.ix[start_index0:end_index0, start_index1:end_index1]

                    submatrix_reverse = submatrix.iloc[::-1]

                    a4_dims = (40, 40)
                    fig, ax = plt.subplots(figsize=a4_dims)

                    # sns_heatmap = sns.heatmap(submatrix, ax=ax,annot=True, linewidths=.5)
                    sns_heatmap = sns.heatmap(submatrix_reverse, ax=ax,annot=True, linewidths=.5)

                    sns_heatmap.set(xlabel='Monomer B', ylabel='Monomer A')

                    # sns_heatmap.set_title('Number of contacts: {0}'.format(str(len(index_to_work))), fontsize=60)
                    sns_heatmap.set_title(title, fontsize=48)
                    # TODO very important bug don't forget it
                    sns_heatmap.set(xticklabels=indexes1)
                    sns_heatmap.set(yticklabels=indexes0)

                    sns_heatmap.figure.savefig(self.simulation_name + '_'  + 'contact_maps_cluster:{0}'.format(k) + '.png',
                                               dpi=custom_dpi, bbox_inches='tight')




                # sns.plt.show()


                # sel_traj = xyz[:]

                test = 1






    @hlp.timeit
    def show_all_cluster_analysis_plots(self,
                                        show_plot=False,
                                        custom_dpi=600):
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                     2)  # sharex='col', sharey='row') TODO this can be used for shared columns
        fig.set_size_inches(20, 20)

        cluster_range = self.range_n_clusters
        score = self.book_dbi_pca
        criteria_name = 'Davis-Bouldain Index'
        score_text = 'The optimal clustering solution\n' \
                     ' has the smallest Davies-Bouldin index value.'
        ax1.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax1.plot(cluster_range, score, ':k', linewidth=3.0)

        ax1.set_xlim(cluster_range[0], cluster_range[-1])

        ax1.set_title(score_text)
        ax1.set_xlabel('n of clusters')
        ax1.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = self.dunn_pca
        criteria_name = "Dunn's Index"
        score_text = "Maximum value of the index\n" \
                     "represents the right partitioning given the index"
        ax2.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax2.plot(cluster_range, score, ':k', linewidth=3.0)

        ax2.set_xlim(cluster_range[0], cluster_range[-1])
        ax2.set_title(score_text)
        ax2.set_xlabel('n of clusters')
        ax2.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = self.sil_pca
        criteria_name = 'Mean Silhouette Coefficient for all samples'
        score_text = 'Objects with a high silhouette\n' \
                     'value are considered well clustered'
        ax3.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax3.plot(cluster_range, score, ':k', linewidth=3.0)

        ax3.set_xlim(cluster_range[0], cluster_range[-1])
        ax3.set_title(score_text)
        ax3.set_xlabel('n of clusters')
        ax3.set_ylabel(criteria_name)

        cluster_range = self.range_n_clusters
        score = self.calinski_pca
        criteria_name = 'Calinski-Harabaz score'
        score_text = 'Objects with a high Calinski-Harabaz\n' \
                     'score value are considered well clustered'
        ax4.scatter(cluster_range, score, marker='o', c='b', s=200)
        ax4.plot(cluster_range, score, ':k', linewidth=3.0)
        ax4.set_xlim(cluster_range[0], cluster_range[-1])
        ax4.set_title(score_text)
        ax4.set_xlabel('n of clusters')
        ax4.set_ylabel(criteria_name)

        plt.tight_layout()
        # plt.suptitle(("Docking Number of Cluster Determination"),
        #              fontsize=14, fontweight='bold')

        if show_plot is True:
            plt.show()

        fig.savefig(self.simulation_name + '_' + 'MD_cluster_scores.png', dpi=custom_dpi, bbox_inches='tight')

    @hlp.timeit
    def show_silhouette_analysis_pca_best(self, show_plot=False, custom_dpi=600):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        n_clusters = self.select_number_of_clusters()

        cluster_labels = self.clusters_info[n_clusters]['labels']
        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        X = self.reduced_cartesian

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        sns.set(font_scale=2)

        # sns.axes_style()


        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10

        # TODO a new try
        colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        self.colors_ = colors

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
                    c=colors)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=100)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=100)

        ax2.set_title("The visualization of the clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        fig.savefig(self.simulation_name + '_' + 'Best_cluster_analysis_md_' + '.png', dpi=custom_dpi,
                    bbox_inches='tight')
        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_cluster_analysis_pca_best(self, show_plot=False, custom_dpi=600):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        n_clusters = self.select_number_of_clusters()

        cluster_labels = self.clusters_info[n_clusters]['labels']
        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        X = self.reduced_cartesian

        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(figsize=(10, 10))
        # fig.set_size_inches(18, 7)

        sns.set(font_scale=2)

        # TODO a new try
        colors = self.colors_

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        plt.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
                    c=colors)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # Draw white circles at cluster centers
        plt.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=800)

        for i, c in enumerate(centers):
            clust_num = i + 1
            plt.scatter(c[0], c[1], marker='$%d$' % clust_num, alpha=1, s=800)

        plt.title("The visualization of the clustered data")
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")

        # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

        fig.savefig(self.simulation_name + '_' + 'Best_cluster_analysis_simple_md_VIP' + '.png', dpi=custom_dpi,
                    bbox_inches='tight')
        if show_plot is True:
            plt.show()

    # TODO do PCA transformation of MD simulation
    @hlp.timeit
    def rosetta_pca_analysis(self, selection='protein'):

        self.called_md_pca_analysis = True
        print('PCA analysis has been called\n')
        print('-------------------------------\n')

        pca1 = PCA(n_components=2)

        # this is for keeping selection from trajectory

        # self.pca_traj = self.full_traj[:]
        #
        # self.topology = self.pca_traj.topology
        #
        # self.selection = self.topology.select(selection)
        #
        # # self.selection = self.topology.select(selection)
        # # print('selection is ', self.selection)
        #
        # self.pca_traj.restrict_atoms(self.selection)
        # self.full_traj.save(selection +'.pdb')

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        self.pca_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        self.pca_traj.superpose(self.pca_traj, 0)

        self.reduced_cartesian = pca1.fit_transform(
            self.pca_traj.xyz.reshape(self.pca_traj.n_frames, self.pca_traj.n_atoms * 3))
        print(self.reduced_cartesian.shape)
        print("PCA transformation finished successfully")
        print('-----------------------------------\n')



        # TODO do PCA transformation of MD simulation

    @hlp.timeit
    def rosetta_pca_analysis_energy(self, selection='protein', filter=True):

        self.called_md_pca_analysis = True
        print('PCA analysis has been called\n')
        print('-------------------------------\n')

        pca1 = PCA(n_components=2)

        # this is for keeping selection from trajectory

        # self.pca_traj = self.full_traj[:]
        #
        # self.topology = self.pca_traj.topology
        #
        # self.selection = self.topology.select(selection)
        #
        # # self.selection = self.topology.select(selection)
        # # print('selection is ', self.selection)
        #
        # self.pca_traj.restrict_atoms(self.selection)
        # self.full_traj.save(selection +'.pdb')

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        self.pca_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        self.pca_traj.superpose(self.pca_traj, 0)

        self.transformed_data = self.pca_traj.xyz.reshape(self.pca_traj.n_frames, self.pca_traj.n_atoms * 3)

        self.original_transformed_data_pandas = pd.DataFrame(self.transformed_data)

        # df1['e'] = Series(np.random.randn(sLength), index=df1.index)

        # self.full_score = []
        # self.full_rmsd = []

        # self.transformed_data_pandas['score'] = Series(self.full_score, index=self.transformed_data_pandas.index)
        self.original_transformed_data_pandas['score'] = pd.Series(self.full_score)

        # Maybe without rmsd
        self.original_transformed_data_pandas['rmsd'] = pd.Series(self.full_rmsd)

        self.transformed_data_pandas = self.original_transformed_data_pandas[:]

        if filter is True:
            self.transformed_data_pandas = self.transformed_data_pandas.loc[self.transformed_data_pandas['score'] < 0]
            test = 1

        # DROP both score and rmsd
        # self.to_analyze =  self.transformed_data_pandas.drop(['score', 'rmsd'], axis=1)

        # Drop score
        self.to_analyze = self.transformed_data_pandas.drop(['score'], axis=1)

        self.reduced_cartesian = pca1.fit_transform(self.to_analyze)
        print(self.reduced_cartesian.shape)
        print("PCA transformation finished successfully")
        print('-----------------------------------\n')

    @hlp.timeit
    def extract_info_cluster_data(self, cluster_data, key):
        temp_data = []
        for clust_num in self.range_n_clusters:
            temp_data.append(cluster_data[clust_num][key])
        return temp_data

    @hlp.timeit
    def rosetta_pca_full_analysis(self, show=False, algorithm='kmeans', parallel=True, num_of_threads=7):
        if parallel is True:
            self.parallel_cluster_proc = []

            # self.simultaneous_run = list(range(0, num_of_threads))

            pool = multiprocessing.Pool(num_of_threads)

            # range_n_clusters = list(range(1, 11))
            k_neighb = 25

            function_arguments_to_call = [[x, self.reduced_cartesian, self.pca_traj.time, algorithm, k_neighb] for x in
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

            self.clusters_info = md_silhouette_analysis_pca(self.reduced_cartesian,
                                                            self.pca_traj.time,
                                                            range_n_clusters=self.range_n_clusters,
                                                            show_plots=show,
                                                            algorithm=algorithm)

        self.sil_pca = self.extract_info_cluster_data(self.clusters_info, 'silhouette')
        self.calinski_pca = self.extract_info_cluster_data(self.clusters_info, 'calinski')
        self.dunn_pca = self.extract_info_cluster_data(self.clusters_info, 'dunn')
        self.dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'dbi')

        self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')
        # self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()

        self.clust_num = self.select_number_of_clusters()

        # self.cluster_list = self.collect_cluster_info()

    @hlp.timeit
    def select_number_of_clusters(self):
        # ["foo", "bar", "baz"].index("bar")
        max_silhouette = max(self.sil_pca)
        max_dunn = max(self.dunn_pca)
        min_dbi = min(self.dbi_pca)

        sil_index = self.sil_pca.index(max_silhouette)
        dunn_index = self.dunn_pca.index(max_dunn)
        dbi_index = self.dbi_pca.index(min_dbi)

        cluster_quantity = []
        cluster_quantity.append(self.range_n_clusters[sil_index])
        cluster_quantity.append(self.range_n_clusters[dunn_index])
        cluster_quantity.append(self.range_n_clusters[dbi_index])

        print('------------------------------------------------')
        print('verify yolo', cluster_quantity)

        cluster_set = set(cluster_quantity)

        cluster_dict = {}
        for n_set in cluster_set:
            count = cluster_quantity.count(n_set)
            cluster_dict.update({n_set: count})

        print('verify yolo ', cluster_dict)

        import operator
        # clust_num = max(cluster_dict.items(), key=operator.itemgetter(1))[0]


        whole_stuff = max(cluster_dict.items(), key=operator.itemgetter(1))
        # clust_num = max(cluster_dict.iterkeys(), key=lambda k: cluster_dict[k])
        clust_num_pre = [key for key, val in cluster_dict.items() if val == max(cluster_dict.values())]

        import numpy
        def median(lst):
            return numpy.median(numpy.array(lst))

        clust_num = sorted(clust_num_pre)[len(clust_num_pre) // 2]

        print("number of clusters is ", clust_num)

        return clust_num

    # This part is for helper functions
    def cleanVal(self, val):
        tempus = val.split(' ')
        # print tempus
        for i in tempus:
            if i == '':
                pass
            else:
                return i

    def extractInfo(self, line):
        lineSplit = line.split(' ')
        # print('lineSplit ',lineSplit)
        cleanedVal = []
        for i in lineSplit:
            if i == '':
                pass
            else:
                cleanedVal.append(i)
        return cleanedVal

    def cleanArray(self, array_data):
        clean_array = []
        for i in array_data:
            if i == '':
                pass
            else:
                clean_array.append(i)
        return clean_array


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

    def hold_nSec(self, n):
        for i in range(1, n + 1):
            print(i)
            time.sleep(1)  # Delay for 1 sec
        print('Ok %s secs have pass' % (n))

    def get_molecule_name(self):
        return self.molecule_name

    def get_receptor_name(self):
        return self.receptor_name

    def set_molecule_name(self, mol_name):
        self.molecule_name = mol_name

    def set_receptor_name(self, receptor_name):
        self.receptor_name = receptor_name

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
