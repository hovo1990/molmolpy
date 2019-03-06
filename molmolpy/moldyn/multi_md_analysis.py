# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    multi_md_analysis.py
# @brief   multi_md_analysis object
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
import time

import pylab as plt

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


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)



class MultiMDAnalysisObject(object):
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
    Convert gro to PDB so mdtraj recognises topology
    YEAH

    gmx editconf -f npt.gro -o npt.pdb


    """

    def __init__(self, file_list=None):

        self.simulation_data = {}

        self.sim_indexes = []

        if file_list is not None:
            if len(file_list) > 0:
                for i in range(len(file_list)):
                    self.add_simulation_pickle_data(i + 1, file_list[i])
                    self.sim_indexes.append(i + 1)

            colors = sns.cubehelix_palette(n_colors=len(file_list), rot=.7, dark=0, light=0.85)
            self.colors_ = colors

        test = 1

    def add_simulation_pickle_data(self, index, filename):
        temp_data = pickle.load(open(filename, "rb"))
        self.simulation_data.update({str(index): temp_data})

    @hlp.timeit
    def plot_rmsd_multi(self, selection,
                        title='Simulation',
                        xlabel=r"Time $t$ (ns)",
                        ylabel=r"RMSD(nm)",
                        custom_dpi=1200,
                        custom_labels=None,
                        position='best',
                        noTitle=True,
                        size_x=8.4,
                        size_y=7):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=0.8)

        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        title = 'Cluster Simulation {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

        # fig = plt.figure(figsize=(10, 7))

        # fig.suptitle(title, fontsize=16)
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')


        # fig = plt.figure(figsize=(10, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(size_x, size_y))

        # fig.suptitle(title, fontsize=16)
        if noTitle is False:
            fig.suptitle(title)

        for i in self.sim_indexes:
            self.sim_time = self.simulation_data[str(i)]['time']
            traj_rmsd = self.simulation_data[str(i)]['rmsd'][selection]
            if custom_labels is None:
                curr_label = 'Simulation {0}'.format(i)
            else:
                curr_label = '{0}'.format(custom_labels[i-1])

            curr_color = self.colors_[i - 1]
            plt.plot(self.sim_time, traj_rmsd, color=curr_color,
                     linewidth=0.52, label=curr_label)

        # plt.legend(loc="best", prop={'size': 8})
        # plt.xlabel(xlabel, fontsize=16)
        # plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm

        # leg = plt.legend(loc='best', shadow=True, prop={'size': 16})
        leg = plt.legend(loc=position, shadow=True, ncol=2)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(6.0)

        # remove part of ticks
        sns.despine()

        fig.savefig('Multi_Plot_RMSD_' + '_' + title + '_' + selection + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rg_multi(self,
                      selection,
                      title='LasR Rg',
                      xlabel=r"time $t$ (ns)",
                      ylabel=r"C$_\alpha$ Rg from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
                      custom_dpi=600):
        import pylab as plt

        sns.set(style="ticks", context='paper')
        # sns.set(font_scale=2)

        # In[27]:

        fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        title = 'Cluster Simulation {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

        # fig = plt.figure(figsize=(10, 7))

        fig.suptitle(title, fontsize=16)
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')


        for i in self.sim_indexes:
            self.sim_time = self.simulation_data[str(i)]['time']
            traj_rmsd = self.simulation_data[str(i)]['Rg'][selection]
            curr_label = 'Simulation {0}'.format(i)

            curr_color = self.colors_[i - 1]
            plt.plot(self.sim_time, traj_rmsd, color=curr_color,
                     linewidth=0.6, label=curr_label)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm

        leg = plt.legend(loc='best', shadow=True, prop={'size': 16})

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(9.0)

        # remove part of ticks
        sns.despine()
        # In[28]:

        fig.savefig('Multi_Plot_Rg_' + '_' + title + '_' + selection + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('Rg plot created')
        print('-----------------------------------\n')


    # TODO calculate confidence intervals
    @hlp.timeit
    def plot_rmsf_plus_confidence_multi(self, selection,
                        title='LasR RMSF',
                        xlabel=r"Residue",
                        ylabel=r"RMSF(nm)",
                        custom_dpi=600):

        '''

        ylabel=r"C$_\alpha$ RMSF from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        :param title:
        :param xlabel:
        :param ylabel:
        :param custom_dpi:
        :return:
        '''
        sns.set(style="ticks", context='paper')
        # sns.set(font_scale=2)

        fig = plt.figure(figsize=(14, 7))

        title = 'Cluster Simulation {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

        # fig = plt.figure(figsize=(10, 7))

        fig.suptitle(title, fontsize=16)

        for i in self.sim_indexes:
            self.sim_time = self.simulation_data[str(i)]['time']
            curr_label = 'Simulation {0}'.format(i)

            traj_rmsf = self.simulation_data[str(i)]['rmsf'][selection]['rmsf']
            atom_indices_rmsf = self.simulation_data[str(i)]['rmsf'][selection]['ref_atom_indices']
            curr_color = self.colors_[i - 1]

            conv_data = converters.convert_data_to_pandas(atom_indices_rmsf, traj_rmsf, x_axis_name='Residue',
                                                          y_axis_name='RMSF')

            conv_data['Residue'] += 1

            confidence = hlp.mean_confidence_interval(conv_data['RMSF'])

            # plt.plot(self.sim_time, traj_rmsd, color=curr_color,
            #          linewidth=0.6, label=curr_label)

            # Plot the response with standard error
            sns.tsplot(data=conv_data, ci=[95], color="m")

            # plt.plot(conv_data['x'], conv_data['y'], color=curr_color,
            #          linewidth=0.6, label=curr_label)

            # plt.xlim(min(conv_data['x']) - 100, max(conv_data['x']) + 100)

        # traj_rmsf = self.rmsf_analysis_data[selection]['rmsf']
        # atom_indices_rmsf = self.rmsf_analysis_data[selection]['atom_indices']

        # sns.tsplot(time="x", unit="y",  data=conv_data,
        #            size=4,  fit_reg=False,
        #            scatter_kws={"s": 50, "alpha": 1})
        # sns.plt.show()

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm

        leg = plt.legend(loc='best', shadow=True, prop={'size': 16})

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(9.0)

        # plt.title(title)

        # remove part of ticksg

        sns.despine()

        fig.savefig('Multi_Plot_RMSF_confidence_' + '_' + title + '_' + selection + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('RMSF +confidence plot created')

    @hlp.timeit
    def prep_mdtraj_object(self, filename):
        '''
        Prepare receptor mdtraj object

        get mdtraj topology and save as pandas dataframe

        Calculate pdb receptor center of mass


        :return:
        '''
        self.receptor_file = filename
        self.receptor_mdtraj = md.load_pdb(self.receptor_file)

        self.receptor_mdtraj_topology = self.receptor_mdtraj.topology
        self.receptor_mdtraj_topology_dataframe = self.receptor_mdtraj.topology.to_dataframe()


        topology = self.receptor_mdtraj.topology
        atom_indices = topology.select('backbone')


        test = 1

        # self.center_of_mass_receptor = md.compute_center_of_mass(self.receptor_mdtraj)[0]
        #
        # self.x_center = math.ceil(self.center_of_mass_receptor[0] * 10)
        # self.y_center = math.ceil(self.center_of_mass_receptor[1] * 10)
        # self.z_center = math.ceil(self.center_of_mass_receptor[2] * 10)
        #
        # self.receptor_pybel = pybel.readfile("pdb", self.receptor_file).__next__()
        # self.ligand_pybel = pybel.readfile("pdb", self.ligand_file).__next__()

        test = 1




    @hlp.timeit
    def plot_rmsf_multi(self, selection,
                        title='LasR RMSF',
                        xlabel=r"Residue",
                        ylabel=r"RMSF(nm)",
                        custom_dpi=1200):

        '''

        ylabel=r"C$_\alpha$ RMSF from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        :param title:
        :param xlabel:
        :param ylabel:
        :param custom_dpi:
        :return:
        '''
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=0.8)

        # fig = plt.figure(figsize=(14, 7))

        title = 'Cluster Simulation {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

        # fig = plt.figure(figsize=(10, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        # fig.suptitle(title, fontsize=16)
        fig.suptitle(title)


        # self.receptor_mdtraj_topology.atom(3000).residue.resSeq

        for i in self.sim_indexes:
            self.sim_time = self.simulation_data[str(i)]['time']
            curr_label = 'Simulation {0}'.format(i)

            traj_rmsf = self.simulation_data[str(i)]['rmsf'][selection]['rmsf']
            atom_indices_rmsf = self.simulation_data[str(i)]['rmsf'][selection]['ref_atom_indices']
            curr_color = self.colors_[i - 1]

            converted_resseq,converted_index = converters.convert_mdtraj_atom_nums_to_resseq(self.receptor_mdtraj_topology,
                                                                             atom_indices_rmsf)



            conv_data_temp = converters.convert_data_to_pandas(atom_indices_rmsf, traj_rmsf)


            conv_data = conv_data_temp.ix[converted_index]

            conv_data['x'] = converted_resseq
            test = 1

            # plt.plot(self.sim_time, traj_rmsd, color=curr_color,
            #          linewidth=0.6, label=curr_label)

            plt.plot(conv_data['x'], conv_data['y'], color=curr_color,
                     linewidth=0.52, label=curr_label)

            #plt.xlim(min(conv_data['x']) - 100, max(conv_data['x']) + 100)

        # traj_rmsf = self.rmsf_analysis_data[selection]['rmsf']
        # atom_indices_rmsf = self.rmsf_analysis_data[selection]['atom_indices']

        # sns.tsplot(time="x", unit="y",  data=conv_data,
        #            size=4,  fit_reg=False,
        #            scatter_kws={"s": 50, "alpha": 1})
        # sns.plt.show()

        # plt.xlabel(xlabel, fontsize=16)
        # plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  #

        # leg = plt.legend(loc='best', shadow=True, prop={'size': 16})
        leg = plt.legend(loc='best', shadow=True)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(6.0)

        # plt.title(title)

        # remove part of ticksg

        sns.despine()

        fig.savefig('Multi_Plot_RMSF_' + '_' + title + '_' + selection + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('RMSF plot created')


    def count_lig_hbond(self, t, hbonds, ligand):
        label = lambda hbond: '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))

        hbond_atoms = []
        hbond_indexes_sel = []
        hbond_count = 0
        for hbond in hbonds:
            res = label(hbond)
            # print('res ', res)
            if ligand in res:
                # print("res is ", res)
                hbond_atoms.append(res)
                hbond_indexes_sel.append(hbond)
                hbond_count += 1
                test=1
        # print('------------------------------------------------')
        test = 1
        return hbond_atoms, hbond_count, hbond_indexes_sel

    @hlp.timeit
    def hbond_lig_count_analysis(self,
                       ligand_name='HSL',
                       title='Simulation',
                       xlabel=r"Time $t$ (ns)",
                       ylabel=r"Number of Hydrogen Bonds",
                       custom_dpi=600):

        sns.set(style="ticks", context='paper')
        # sns.set(font_scale=2)

        fig = plt.figure(figsize=(14, 7))

        title = 'Simulations of Clusters {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

        # fig = plt.figure(figsize=(10, 7))

        fig.suptitle(title, fontsize=16)

        traj_frame = self.simulation_data[str(self.sim_indexes[0])]['clustersCentroid']

        self.sim_time = self.simulation_data[str(self.sim_indexes[0])]['time']

        t = traj_frame[0]

        for i in self.sim_indexes:
            self.sim_time = self.simulation_data[str(i)]['time']


            hbonds_frames = self.simulation_data[str(i)]['hbondFrames']

            sim_hbond_atoms = []
            sim_hbond_count = []

            for hbonds in hbonds_frames:
                hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(t, hbonds, ligand_name)

                sim_hbond_atoms.append(hbond_atoms)
                sim_hbond_count.append(hbond_count)

            sim_hbound_np = np.array(sim_hbond_count)

            self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
            self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

            curr_color = self.colors_[i - 1]
            # curr_label = 'Simulation {0}'.format(i)
            curr_label = "Simulation of Cluster {0} mean: {1}±{2}".format(i, round(np.mean(sim_hbound_np),3),
                                                                          round(np.std(sim_hbond_count),3))

            # Version 1
            plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
                     linewidth=0.2, label=curr_label)

            # Version 2
            # plt.scatter(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
            #          linewidth=0.3, label=curr_label)


            # data_frame = converters.convert_data_to_pandas(self.sim_time, self.hbond_count)
            #
            # y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()

            # atom_indices_rmsf = self.simulation_data[str(i)]['rmsf'][selection]['ref_atom_indices']
            # curr_color = self.colors_[i - 1]
            #
            # conv_data = converters.convert_data_to_pandas(atom_indices_rmsf, traj_rmsf)
            #
            # # plt.plot(self.sim_time, traj_rmsd, color=curr_color,
            # #          linewidth=0.6, label=curr_label)
            #
            # plt.plot(conv_data['x'], conv_data['y'], color=curr_color,
            #          linewidth=0.6, label=curr_label)

            # plt.xlim(min(conv_data['x']) - 100, max(conv_data['x']) + 100)

        test = 1


        # traj_rmsf = self.rmsf_analysis_data[selection]['rmsf']
        # atom_indices_rmsf = self.rmsf_analysis_data[selection]['atom_indices']

        # sns.tsplot(time="x", unit="y",  data=conv_data,
        #            size=4,  fit_reg=False,
        #            scatter_kws={"s": 50, "alpha": 1})
        # sns.plt.show()

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm

        leg = plt.legend(loc='best', shadow=True, prop={'size': 16})

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(9.0)

        # plt.title(title)

        # remove part of ticksg

        sns.despine()

        fig.savefig('Multi_Plot_HBOND_count_Lig_' + '_' + title + '_' + ligand_name + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('Multi HBond lig count plot created')


    @hlp.timeit
    def hbond_freq_plot_analysis(self,
                       ligand_name='HSL',
                       title='Simulation',
                       xlabel=r"Time $t$ (ns)",
                       ylabel=r"Number of Hydrogen Bonds",
                       custom_dpi=600):

        sns.set(style="ticks", context='paper')
        # sns.set(font_scale=2)



        traj_frame = self.simulation_data[str(self.sim_indexes[0])]['clustersCentroid']

        self.sim_time = self.simulation_data[str(self.sim_indexes[0])]['time']

        t = traj_frame[0]

        for i in self.sim_indexes:

            plt.clf()

            fig = plt.figure(figsize=(14, 7))

            title = 'Simulations of Clusters {0}-{1}'.format(self.sim_indexes[0], self.sim_indexes[-1])

            # fig = plt.figure(figsize=(10, 7))

            fig.suptitle(title, fontsize=16)

            self.sim_time = self.simulation_data[str(i)]['time']


            hbonds_frames = self.simulation_data[str(i)]['hbondFrames']

            sim_hbond_atoms = []
            sim_hbond_count = []

            sim_hbond_sel = []

            for hbonds in hbonds_frames:
                hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(t, hbonds, ligand_name)

                sim_hbond_atoms.append(hbond_atoms)
                sim_hbond_count.append(hbond_count)

                if len( hbond_indexes_sel) > 0:
                    sim_hbond_sel+= hbond_indexes_sel

            sim_hbound_np = np.array(sim_hbond_count)
            sim_hbound_sel_np = np.array(sim_hbond_sel)


            # self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
            # self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

            # curr_color = self.colors_[i - 1]
            # curr_label = 'Simulation {0}'.format(i)
            curr_label = "Simulation of Cluster {0} mean: {1}±{2}".format(i, round(np.mean(sim_hbound_np),3),
                                                                          round(np.std(sim_hbond_count),3))

            # This won't work here
            da_distances = md.compute_distances(t, sim_hbound_sel_np[:, [0, 2]], periodic=False)

            # Version 1
            # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
            #          linewidth=0.2, label=curr_label)
            # color = itertools.cycle(['r', 'b', 'gold'])

            colors = sns.cubehelix_palette(n_colors=len(da_distances), rot=-.4)
            # self.colors_ = colors
            label = lambda hbond: '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))

            color = itertools.cycle(['r', 'b', 'gold'])

            for i in [0]:
                plt.hist(da_distances[:, i], color=colors[i], label=label(sim_hbound_sel_np[i]), alpha=0.5)
            plt.legend()
            plt.ylabel('Freq');
            plt.xlabel('Donor-acceptor distance [nm]')


            # plt.xlabel(xlabel, fontsize=16)
            # plt.ylabel(ylabel, fontsize=16)  # fix Angstrom need to change to nm
            #
            # leg = plt.legend(loc='best', shadow=True, prop={'size': 16})
            #
            # # set the linewidth of each legend object
            # for legobj in leg.legendHandles:
            #     legobj.set_linewidth(9.0)


            sns.despine()

            fig.savefig('Multi_Plot_HBOND_frequency_' + '_' + title + '_' + str(i)+ '_'+ ligand_name + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('Multi HBond frequency lig  plot created')







    @hlp.timeit
    def plot_solvent_area_multi(self, show=False):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.sasa_traj.time, self.total_sasa)
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('Total SASA (nm)^2', size=16)

        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_plot.png', dpi=300, bbox_inches='tight')

    @hlp.timeit
    def plot_solvent_area_frame_multi(self, frame, show=False):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.sasa_traj.time, self.sasa[frame])
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('Total SASA (nm)^2', size=16)
        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_plot_{0}.png'.format(frame), dpi=300, bbox_inches='tight')

    @hlp.timeit
    def plot_solvent_area_autocorr_multi(self, show=False):
        self.sasa_autocorr = protein_analysis.autocorr(self.total_sasa)
        fig = plt.figure(figsize=(10, 10))
        plt.semilogx(self.sasa_traj.time, self.sasa_autocorr)
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('SASA autocorrelation', size=16)

        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_autocorrelation.png', dpi=300, bbox_inches='tight')

    @hlp.timeit
    def plot_rmsd_cluster_color_multi(self, selection,
                                      title='LasR RMSD',
                                      xlabel=r"Time $t$ (ns)",
                                      ylabel=r"RMSD(nm)",
                                      custom_dpi=300,
                                      lang='rus'):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(14, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # plt.plot(self.sim_time, self.sim_rmsd, color=self.cluster_colors,
        #          linewidth=0.6, label='LasR')


        if lang == 'rus':
            title = 'Симуляция'
            xlabel = r"Время $t$ (нс)"
            ylabel = r"RMSD(нм)"
        else:
            title = 'Simulation'
            xlabel = r"Time $t$ (ns)"
            ylabel = r"RMSD(nm)"

        sns.set(font_scale=2)
        plt.plot(self.sim_time, self.sim_rmsd, zorder=1)
        traj_rmsd = self.rmsd_analysis_data[selection]

        plt.scatter(self.sim_time, traj_rmsd, marker='o', s=30, facecolor='0.5', lw=0,
                    c=self.cluster_colors, zorder=2)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.xlim(self.sim_time[0], self.sim_time[-1])

        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        fig.tight_layout()

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(self.simulation_name + '_' + title + '_' + selection + '_cluster_color' + '_' + lang + '.png',
                    dpi=custom_dpi, bbox_inches='tight')

        print('RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def find_best_fit_regressor(self):

        # from sklearn.tree import DecisionTreeRegressor


        self.best = 100
        self.index = 100

        self.best_rg = 100
        self.index_rg = 100

        self.regr_index = []
        self.regr_scores = {}
        self.regr_index_rg = []
        self.regr_scores_rg = {}

        self.reshaped_time = self.sim_time.reshape(-1, 1)
        for i in list(range(1, self.regression_fit_range + 1)):
            self.create_fit(i)

        print('best score is ', self.best)
        print('best index is', self.index)
        print('-=-' * 10)
        print('best score Rg is ', self.best_rg)
        print('best index Rg is', self.index_rg)

    @hlp.timeit
    def create_fit(self, i):
        from sklearn import tree
        from sklearn.model_selection import cross_val_score

        self.reshaped_time = self.sim_time.reshape(-1, 1)
        regressor = tree.DecisionTreeRegressor(max_depth=i)  # interesting absolutely
        fitVal = regressor.fit(self.reshaped_time, self.sim_rmsd)

        print('fitVal ', fitVal)
        rmsd_pred = regressor.predict(self.reshaped_time)
        # cv how is it determined?
        #  A good compromise is ten-fold cross-validation. 10ns
        # Maybe mse better?
        cross_val = cross_val_score(regressor,
                                    self.reshaped_time,
                                    self.sim_rmsd,
                                    scoring="neg_mean_squared_error",
                                    cv=10)

        regressor_rg = tree.DecisionTreeRegressor(max_depth=i)  # interesting absolutely
        fitVal_rg = regressor_rg.fit(self.reshaped_time, self.rg_res)

        fitVal_rg = regressor_rg.fit(self.reshaped_time, self.rg_res)
        print('fitVal ', fitVal)
        rmsd_pred_rg = regressor_rg.predict(self.reshaped_time)
        # cv how is it determined?
        #  A good compromise is ten-fold cross-validation. 10ns
        cross_val_rg = cross_val_score(regressor,
                                       self.reshaped_time,
                                       self.rg_res,
                                       scoring="neg_mean_squared_error",
                                       cv=10)

        self.regr_scores.update({i: cross_val})
        self.regr_index.append(i)

        self.regr_scores_rg.update({i: cross_val_rg})
        self.regr_index_rg.append(i)

        cross_val_score = -cross_val.mean()
        cross_val_std = cross_val.std()

        cross_val_score_rg = -cross_val_rg.mean()
        cross_val_std_rg = cross_val_rg.std()

        print('Cross validation score is ', cross_val)
        print("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(i, -cross_val.mean(), cross_val.std()))
        print('-=-' * 10)
        print('Cross validation Rg score is ', cross_val_rg)
        print("Rg Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(i, -cross_val_rg.mean(), cross_val_rg.std()))
        # r2_score = regressor.score(self.sim_time.reshape(-1, 1), self.sim_rmsd)
        # if r2_score > self.r2_best:
        #     self.r2_best = r2_score
        #     self.r2_index = i

        if cross_val_score < self.best:
            self.best = cross_val_score
            self.index = i

        if cross_val_score_rg < self.best_rg:
            self.best_rg = cross_val_score_rg
            self.index_rg = i

        del regressor
        del fitVal
        del rmsd_pred
        time.sleep(2)
        # print('R2 score is ', r2_score)
        print('---------------------------------------------------------------\n')

    @hlp.timeit
    def error_bar_rmsd_fit(self):
        import matplotlib.pyplot as plt

        x = self.regr_index

        y = []
        yerr_list = []

        for i in self.regr_index:
            # plt.boxplot(self.regr_scores[i])
            cross_val_score = -self.regr_scores[i].mean()
            cross_val_std = self.regr_scores[i].std()

            y.append(cross_val_score)
            yerr_list.append(cross_val_std)

        fig = plt.figure(figsize=(10, 10))

        plt.errorbar(x, y, yerr=yerr_list)
        plt.scatter(x, y, s=160, c='b', marker='h',
                    label="Best score at Max Depth={}\nMSE = {:.2e}(+/- {:.2e})".format(self.index,
                                                                                        -self.regr_scores[
                                                                                            self.index].mean(),
                                                                                        self.regr_scores[
                                                                                            self.index].std()))
        plt.legend(loc="best", prop={'size': 20})
        plt.title("Mean squared error (MSE) averages for RMSD")
        fig.savefig(self.simulation_name + '_errorBar_rmsd.png', dpi=300, bbox_inches='tight')
        # plt.show()
        print('Errorbar created ')
        print('---------------------------------------------------------------\n')

    @hlp.timeit
    def error_bar_Rg_fit(self):
        import matplotlib.pyplot as plt

        x = self.regr_index

        y = []
        yerr_list = []

        for i in self.regr_index:
            # plt.boxplot(self.regr_scores[i])
            cross_val_score = -self.regr_scores_rg[i].mean()
            cross_val_std = self.regr_scores_rg[i].std()

            y.append(cross_val_score)
            yerr_list.append(cross_val_std)

        fig = plt.figure(figsize=(10, 10))

        plt.errorbar(x, y, yerr=yerr_list)
        plt.scatter(x, y, s=160, c='b', marker='h',
                    label="Best score at Max Depth={}\nMSE = {:.2e}(+/- {:.2e})".format(self.index_rg,
                                                                                        -self.regr_scores_rg[
                                                                                            self.index_rg].mean(),
                                                                                        self.regr_scores_rg[
                                                                                            self.index_rg].std()))
        plt.legend(loc="best", prop={'size': 20})
        plt.title("Mean squared error (MSE) averages for Rg")
        fig.savefig(self.simulation_name + '_errorBar_Rg.png', dpi=300, bbox_inches='tight')
        # plt.show()
        print('Errorbar created ')
        print('---------------------------------------------------------------\n')

    @hlp.timeit
    def error_bar_fit_test(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # example data
        x = np.arange(0.1, 4, 0.5)
        y = np.exp(-x)

        # example variable error bar values
        yerr = 0.1 + 0.2 * np.sqrt(x)
        xerr = 0.1 + yerr

        # First illustrate basic pyplot interface, using defaults where possible.
        plt.figure()
        plt.errorbar(x, y, xerr=0.2, yerr=0.4)
        plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

        # Now switch to a more OO interface to exercise more features.
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        ax = axs[0, 0]
        ax.errorbar(x, y, yerr=yerr, fmt='o')
        ax.set_title('Vert. symmetric')

        # With 4 subplots, reduce the number of axis ticks to avoid crowding.
        ax.locator_params(nbins=4)

        ax = axs[0, 1]
        ax.errorbar(x, y, xerr=xerr, fmt='o')
        ax.set_title('Hor. symmetric')

        ax = axs[1, 0]
        ax.errorbar(x, y, yerr=[yerr, 2 * yerr], xerr=[xerr, 2 * xerr], fmt='--o')
        ax.set_title('H, V asymmetric')

        ax = axs[1, 1]
        ax.set_yscale('log')
        # Here we have to be careful to keep all y values positive:
        ylower = np.maximum(1e-2, y - yerr)
        yerr_lower = y - ylower

        ax.errorbar(x, y, yerr=[yerr_lower, 2 * yerr], xerr=xerr,
                    fmt='o', ecolor='g', capthick=2)
        ax.set_title('Mixed sym., log y')

        fig.suptitle('Variable errorbars')

        plt.show()

    @hlp.timeit
    def plot_boxplot_fit_regr(self):
        data_to_plot = []
        for i in self.regr_index:
            # plt.boxplot(self.regr_scores[i])
            data_to_plot.append(self.regr_scores[i])

        # Create a figure instance
        fig = plt.figure(figsize=(10, 10))

        # Create an axes instance
        ax = fig.add_subplot(111)

        # Create the boxplot
        # change outlier to hexagon
        # bp = ax.boxplot(data_to_plot, 0, 'gD')

        # dont show outlier
        bp = ax.boxplot(data_to_plot, 0, '')

        # Save the figure
        fig.savefig(self.simulation_name + '_boxplot.png', dpi=600, bbox_inches='tight')
        # plt.show()
        print('Box plot created ')
        print('---------------------------------------------------------------\n')

    @hlp.timeit
    def example_test(self):
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        degrees = [1, 4, 8, 15, 20]

        # true_fun = lambda X: np.cos(1.5 * np.pi * X)
        X = self.sim_time
        y = self.sim_rmsd

        plt.figure(figsize=(14, 5))
        for i in range(len(degrees)):
            ax = plt.subplot(1, len(degrees), i + 1)
            plt.setp(ax, xticks=(), yticks=())

            polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                     include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                                 ("linear_regression", linear_regression)])
            pipeline.fit(X, y)

            # Evaluate the models using crossvalidation
            scores = cross_val_score(pipeline, X, y,
                                     scoring="neg_mean_squared_error", cv=10)

            X_test = self.sim_time
            plt.plot(X_test, pipeline.predict(X_test), label="Model")
            plt.plot(X_test, self.sim_rmsd, label="True function")
            plt.scatter(X, y, label="Samples")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((0, 1))
            plt.ylim((-2, 2))
            plt.legend(loc="best")
            plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
                degrees[i], -scores.mean(), scores.std()))
        plt.show()

    @hlp.timeit
    def plot_rmsd_with_regressor(self, title='LasR Simulation RMSD',
                                 xlabel=r"time  $t$ (ns)",
                                 ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)"):
        import pylab as plt

        from sklearn import tree

        rfc = tree.DecisionTreeRegressor(max_depth=self.index)  # interesting absolutely
        fitVal = rfc.fit(self.sim_time.reshape(-1, 1), self.sim_rmsd)
        print('fitVal ', fitVal)
        self.rmsd_pred = rfc.predict(self.sim_time.reshape(-1, 1))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        plt.plot(self.sim_time, self.sim_rmsd, color='b',
                 linewidth=0.6, label='Original Data')

        plt.plot(self.sim_time, self.rmsd_pred, color='r',
                 linewidth=4, label='Fitted Data')

        plt.legend(loc="best", prop={'size': 30})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # In[28]:

        fig.savefig(self.simulation_name + '_' + title + '_tree' + '.png', dpi=300, bbox_inches='tight')
        print('RMSD plot created with regressor')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_Rg_with_regressor(self, title='LasR Radius of Gyration',
                               xlabel=r"time  $t$ (ns)",
                               ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)"):
        import pylab as plt

        from sklearn import tree

        rfc = tree.DecisionTreeRegressor(max_depth=self.index_rg)  # interesting absolutely
        fitVal = rfc.fit(self.sim_time.reshape(-1, 1), self.rg_res)
        print('fitVal ', fitVal)
        self.rmsd_pred_rg = rfc.predict(self.sim_time.reshape(-1, 1))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        plt.plot(self.sim_time, self.rg_res, color='b',
                 linewidth=0.6, label='Original Data')

        plt.plot(self.sim_time, self.rmsd_pred_rg, color='r',
                 linewidth=4, label='Fitted Data')

        plt.legend(loc="best", prop={'size': 30})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # In[28]:

        fig.savefig(self.simulation_name + '_' + title + '_tree' + '.png', dpi=300, bbox_inches='tight')
        print('RMSD plot created with regressor')
        print('-----------------------------------\n')

    @hlp.timeit
    def md_full_load(self, custom_stride=10):
        print('MD Load  has been called\n')
        print('-------------------------------\n')

        self.full_traj = md.load(self.md_trajectory_file, top=self.md_topology_file,
                                 stride=custom_stride)

        self.sim_time = self.full_traj.time / 1000
        print("Full trajectory loaded successfully")
        print('-----------------------------------\n')

    @hlp.timeit
    def rg_analysis(self, selection='protein'):

        self.called_rg_analysis = True

        # self.rg_traj = self.full_traj[:]
        #
        # self.topology = self.rmsd_traj.topology
        #
        # self.selection = self.topology.select(selection)
        #
        # # self.selection = self.topology.select(selection)
        # # print('selection is ', self.selection)
        #
        # self.rg_traj.restrict_atoms(self.selection)

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        self.rg_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        self.rg_res = md.compute_rg(self.rg_traj)

        self.rg_analysis_data.update({selection: self.rg_res})
        print("Rg has been calculated")
        print('-----------------------------------\n')


    @hlp.timeit
    def hbond_analysis_count(self, selection='protein',
                             title='LasR H-Bonds',
                             xlabel=r"Time $t$ (ns)",
                             ylabel=r"Number of Hydrogen Bonds",
                             custom_dpi=300):

        sns.set(style="ticks", context='paper')

        self.called_hbond_analysis_count = True
        print('HBonds analysis has been called\n')
        print('-------------------------------\n')

        self.topology = self.full_traj.topology
        self.selection = self.topology.select(selection)
        print('selection is ', self.selection)

        # this is for keeping selection from trajectory
        # self.full_traj.restrict_atoms(self.selection)

        self.hbond_count = []
        self.sim_time = self.full_traj.time / 1000

        # paral = Pool(processes=16)
        # data_count = list(map(self.hbond_frame_calc, self.full_traj))
        #
        # print('data count ',data_count)

        # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
        # print('count of hbonds is ', len(hbonds))

        # self.hbond_count.append(len(hbonds))
        hbonds_frames = md.wernet_nilsson(self.full_traj, exclude_water=True, periodic=False)

        self.hbonds_frames = hbonds_frames

        for hbonds in hbonds_frames:
            self.hbond_count.append(len(hbonds))

        data_frame = converters.convert_data_to_pandas(self.sim_time, self.hbond_count)

        y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()
        fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        plt.plot(data_frame['x'], data_frame['y'], color='b',
                 linewidth=0.6, label='LasR')

        # Dont plot rolling mean
        plt.plot(data_frame['x'], y_average_mean, color='r',
                 linewidth=0.9, label='LasR rolling mean')

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # remove part of ticks
        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('HBond count plot created')
        print('-----------------------------------\n')


        # for hbond in hbonds:
        #     print(hbond)
        #     print(label(hbond))
        # atom1 = self.full_traj.topology.atom(hbond[0])
        # atom2 = self.full_traj.topology.atom(hbond[2])
        # # atom3 = traj_sim1_hbonds.topology.atom(hbond[2])
        # if atom1.residue.resSeq != atom2.residue.resSeq:
        #     if atom1.residue.resSeq + 1 != atom2.residue.resSeq:
        #         #  for domain reside analysis
        #         if atom1.residue.resSeq < 171 and atom2.residue.resSeq > 172:
        #             diff_hbonds.append(hbond)

    @hlp.timeit
    def hbond_analysis(self, selection='protein'):
        self.topology = self.full_traj.topology
        self.selection = self.topology.select(selection)
        print('selection is ', self.selection)

        # this is for keeping selection from trajectory
        self.full_traj.restrict_atoms(self.selection)

        if self.save_pdb_hbond is True:
            traj_sim1_hbonds = md.load_pdb(self.pdb_file_name)

        hbonds = md.baker_hubbard(traj_sim1_hbonds, periodic=False)

        # hbonds = md.wernet_nilsson(traj_sim1_hbonds, periodic=True)[0]

        label = lambda hbond: '%s -- %s' % (traj_sim1_hbonds.topology.atom(hbond[0]),
                                            traj_sim1_hbonds.topology.atom(hbond[2]))

        diff_hbonds = []

        for hbond in hbonds:
            # print(hbond)
            # print(label(hbond))
            atom1 = traj_sim1_hbonds.topology.atom(hbond[0])
            atom2 = traj_sim1_hbonds.topology.atom(hbond[2])
            # atom3 = traj_sim1_hbonds.topology.atom(hbond[2])
            if atom1.residue.resSeq != atom2.residue.resSeq:
                if atom1.residue.resSeq + 1 != atom2.residue.resSeq:
                    #  domain reside analysis
                    if atom1.residue.resSeq < 171 and atom2.residue.resSeq > 172:
                        diff_hbonds.append(hbond)

        for hbond in diff_hbonds:
            print(hbond)
            print(label(hbond))
        print('Diff hbonds printed\n')

        diff_hbonds = np.asarray(diff_hbonds)

        self.da_distances = md.compute_distances(traj_sim1_hbonds, diff_hbonds[:, [0, 2]], periodic=False)

        import itertools

        # color = itertools.cycle(['r', 'b', 'gold'])
        # fig = plt.figure(figsize=(7, 7))
        # color = np.linspace(0, len(diff_hbonds),len(diff_hbonds))
        #
        # # color = itertools.cycle(['r', 'b','g','gold'])
        # for i in list(range(0,len(diff_hbonds))):
        #     plt.hist(self.da_distances[:, i], color=next(color), label=label(diff_hbonds[i]), alpha=0.5)
        # plt.legend()
        # plt.ylabel('Freq');
        # plt.xlabel('Donor-acceptor distance [nm]')
        # plt.show()

        # this works wel, but needs to be modified
        fig = plt.figure(figsize=(7, 7))
        color = np.linspace(0, len(diff_hbonds), len(diff_hbonds))
        color = itertools.cycle(['r', 'b', 'g', 'tan', 'black', 'grey', 'yellow', 'gold'])
        for i in list(range(0, len(diff_hbonds))):
            plt.hist(self.da_distances[:, i], color=next(color), label=label(diff_hbonds[i]), alpha=0.5)
        plt.legend()
        plt.ylabel('Freq');
        plt.xlabel('Donor-acceptor distance [nm]')
        plt.show()

        fig.savefig(self.simulation_name + '_hbonds.png', dpi=600, bbox_inches='tight')
        print("Hbonds have been calculated")
        print('-----------------------------------\n')

    @hlp.timeit
    def rmsd_analysis(self, selection):
        '''

        :param selection: has to be mdtraj compatible
        :return:
        '''

        self.called_rmsd_analysis = True

        # self.rmsd_traj = self.full_traj[:]
        #
        # self.topology = self.rmsd_traj.topology
        #
        # self.selection = self.topology.select(selection)
        #
        # # self.selection = self.topology.select(selection)
        # # print('selection is ', self.selection)
        #
        # self.rmsd_traj.restrict_atoms(self.selection)
        # self.full_traj.save(selection +'.pdb')

        #  this is for keeping selection from trajectory
        # self.rmsd_traj.restrict_atoms(self.selection)


        # self.rmsd_traj = self.full_traj[:]

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        # self.selection = self.topology.select(selection)
        # print('selection is ', self.selection)
        self.rmsd_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        self.sim_rmsd = md.rmsd(self.rmsd_traj, self.rmsd_traj, 0)

        self.sim_time = self.rmsd_traj.time / 1000

        self.rmsd_analysis_data.update({selection: self.sim_rmsd})

        self.regression_fit_range = 10

        print('RMSD analysis has been called on selection {0}\n'.format(selection))
        print('-----------------------------\n')

    @hlp.timeit
    def plot_rmsd_cluster_color(self, selection,
                                title='LasR RMSD',
                                xlabel=r"Time $t$ (ns)",
                                ylabel=r"RMSD(nm)",
                                custom_dpi=300,
                                lang='rus'):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(14, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # plt.plot(self.sim_time, self.sim_rmsd, color=self.cluster_colors,
        #          linewidth=0.6, label='LasR')


        if lang == 'rus':
            title = 'Симуляция'
            xlabel = r"Время $t$ (нс)"
            ylabel = r"RMSD(нм)"
        else:
            title = 'Simulation'
            xlabel = r"Time $t$ (ns)"
            ylabel = r"RMSD(nm)"

        sns.set(font_scale=2)
        plt.plot(self.sim_time, self.sim_rmsd, zorder=1)
        traj_rmsd = self.rmsd_analysis_data[selection]

        plt.scatter(self.sim_time, traj_rmsd, marker='o', s=30, facecolor='0.5', lw=0,
                    c=self.cluster_colors, zorder=2)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.xlim(self.sim_time[0], self.sim_time[-1])

        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        fig.tight_layout()

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(self.simulation_name + '_' + title + '_' + selection + '_cluster_color' + '_' + lang + '.png',
                    dpi=custom_dpi, bbox_inches='tight')

        print('RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsf(self, selection,
                  title='LasR RMSF',
                  xlabel=r"Residue",
                  ylabel=r"RMSF(nm)",
                  custom_dpi=300):

        '''

        ylabel=r"C$_\alpha$ RMSF from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        :param title:
        :param xlabel:
        :param ylabel:
        :param custom_dpi:
        :return:
        '''
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)

        traj_rmsf = self.rmsf_analysis_data[selection]['rmsf']
        atom_indices_rmsf = self.rmsf_analysis_data[selection]['atom_indices']

        conv_data = converters.convert_data_to_pandas(atom_indices_rmsf, traj_rmsf)

        # sns.tsplot(time="x", unit="y",  data=conv_data,
        #            size=4,  fit_reg=False,
        #            scatter_kws={"s": 50, "alpha": 1})
        # sns.plt.show()

        fig = plt.figure(figsize=(14, 7))
        plt.plot(conv_data['x'], conv_data['y'], color='b',
                 linewidth=0.6, label=title)
        plt.xlabel(xlabel)
        plt.xlim(min(conv_data['x']) - 100, max(conv_data['x']) + 100)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # remove part of ticks
        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '_rmsf.png', dpi=custom_dpi, bbox_inches='tight')

        print('RMSF plot created')

    @hlp.timeit
    def plot_rg(self,
                selection,
                title='LasR Rg',
                xlabel=r"time $t$ (ns)",
                ylabel=r"C$_\alpha$ Rg from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
                custom_dpi=600):
        import pylab as plt

        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)

        # In[27]:

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_rg = self.rg_analysis_data[selection]
        plt.plot((self.sim_time), traj_rg, color='b',
                 linewidth=0.6, label='LasR')

        plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # In[28]:

        fig.savefig(self.simulation_name + '_' + title + '_' + selection + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('RMSD plot created')
        print('-----------------------------------\n')

    #  need to select only protein for analysis
    @hlp.timeit
    def find_centroid(self):
        atom_indices = [a.index for a in self.full_traj.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((self.full_traj.n_frames, self.full_traj.n_frames))
        for i in range(self.full_traj.n_frames):
            distances[i] = md.rmsd(self.full_traj, self.full_traj, i, atom_indices=atom_indices)

        beta = 1
        index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        print(index)

        centroid = self.full_traj[index]
        print(centroid)

        centroid.save('centroid.pdb')

    ####################################################################################################################

    # TODO do PCA transformation of MD simulation
    @hlp.timeit
    def md_pca_analysis(self, selection='protein'):

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

    ####################################################################################################################


    @hlp.timeit
    def extract_info_cluster_data(self, cluster_data, key):
        temp_data = []
        for clust_num in self.range_n_clusters:
            temp_data.append(cluster_data[clust_num][key])
        return temp_data

    @hlp.timeit
    def silhouette_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.sil_pca
        criteria_name = 'Mean Silhouette Coefficient for all samples'
        score_text = 'Objects with a high silhouette value are considered well clustered'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    def calinski_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.calinski_pca
        criteria_name = 'Calinski-Harabaz score'
        score_text = 'Objects with a high Calinski-Harabaz score value are considered well clustered'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    @hlp.timeit
    def dunn_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.dunn_pca
        criteria_name = "Dunn's Index"
        score_text = "Maximum value of the index represents the right partitioning given the index"
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    @hlp.timeit
    def dbi_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.dbi_pca
        criteria_name = 'Davis-Bouldain Index'
        score_text = 'The optimal clustering solution has the smallest Davies-Bouldin index value.'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

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
        clust_num = max(cluster_dict.items(), key=operator.itemgetter(1))[0]

        print("number of clusters is ", clust_num)

        return clust_num

    # def write_model_to_file(self, model,  resnum=None, filename_pdb=None):
    #     curr_df = model['molDetail']['dataframe']
    #     pdb_tools.write_lig(curr_df, resnum, filename_pdb)

    # need to select only protein for analysis

    @hlp.timeit
    def find_max_cluster(self):

        length = 0
        clust_temp_data = []
        for k in self.clusterized_data:
            data = self.clusterized_data[k]
            if len(data) > length:
                length = len(data)
                clust_temp_data = data
                self.max_clust_temp_data = clust_temp_data

        return self.max_clust_temp_data

    @hlp.timeit
    def find_clusters_centroid(self):

        print('Find Clusters centroids is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_centroid = True

        self.clusters_centroids = []

        for k in self.clusterized_data:
            print('Finding centroid for cluster {0}'.format(k))
            clust_temp_data = self.clusterized_data[k]

            atom_indices = [a.index for a in clust_temp_data.topology.atoms if a.element.symbol != 'H']
            distances = np.empty((clust_temp_data.n_frames, clust_temp_data.n_frames))
            for i in range(clust_temp_data.n_frames):
                distances[i] = md.rmsd(clust_temp_data, clust_temp_data, i, atom_indices=atom_indices)

            beta = 1
            index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
            print(index)

            centroid = clust_temp_data[index]
            # self.centroid_conf = centroid
            # print(centroid)

            # self.centroid_conf = centroid
            self.clusters_centroids.append(centroid)
            centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(k))

        print('-----------------------------------\n')

    @hlp.timeit
    def find_max_cluster_centroid(self):

        print('Find Max Cluster centroid is called\n')
        print('-----------------------------------\n')

        self.called_find_max_cluster_centroid = True

        clust_temp_data = self.max_clust_temp_data

        atom_indices = [a.index for a in clust_temp_data.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((clust_temp_data.n_frames, clust_temp_data.n_frames))
        for i in range(clust_temp_data.n_frames):
            distances[i] = md.rmsd(clust_temp_data, clust_temp_data, i, atom_indices=atom_indices)

        beta = 1
        index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        print(index)

        centroid = clust_temp_data[index]
        self.centroid_conf = centroid
        print(centroid)

        self.centroid_conf = centroid
        centroid.save(self.simulation_name + '_' + 'max_cluster_centroid.pdb')

        print('-----------------------------------\n')

    # need to find a way to extract models correctrly
    @hlp.timeit
    def export_cluster_models(self,
                              selection_obj='protein',
                              select_lig=None,
                              save_data=False, nth_frame=1):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''
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
                        self.simulation_name + '_' + 'cluster_' + str(clust_num) + '_' + selection_final_name + '.xtc')
                self.clusterized_data.update({k: sel_traj})

        self.save_pdb_hbond = True

    def save_analysed_data(self, filename):
        '''
        :param filename: Saves clustered data to pickle file
        :return:
        '''
        # import json
        # with open(filename, 'w') as outfile:
        #     json.dump(self.cluster_models, outfile)
        import pickle
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        pickle.dump(self, open(filename, "wb"))

    # should I add json saving of information or not?
    def load_analysed_data(self, filename):
        '''

        :param filename: load pickle file
        :return:
        '''
        self.analysed_data = pickle.load(open(filename, "rb"))
        print('test')

    ####################################################################################################################
    # TODO calc ramachandran part

    @hlp.timeit
    def ramachandran_calc(self):
        self.atoms, self.bonds = self.full_traj.topology.to_dataframe()

        self.phi_indices, self.phi_angles = md.compute_phi(self.full_traj, periodic=False)
        self.psi_indices, self.psi_angles = md.compute_psi(self.full_traj, periodic=False)

        self.angles_calc = md.compute_dihedrals(self.full_traj, [self.phi_indices[0], self.psi_indices[0]])

    @hlp.timeit
    def ramachandran_plot(self):
        from math import pi

        fig = plt.figure(figsize=(7, 7))
        plt.title('Dihedral Map:')
        plt.scatter(self.angles_calc[:, 0], self.angles_calc[:, 1], marker='x', c=self.full_traj.time)
        cbar = plt.colorbar()
        cbar.set_label('Time [ps]')
        plt.xlabel(r'$\Phi$ Angle [radians]')
        plt.xlim(-pi, pi)
        plt.ylabel(r'$\Psi$ Angle [radians]')
        plt.ylim(-pi, pi)
        fig.savefig(self.simulation_name + '_' + 'Ramachandran_analysis' + '.png', dpi=600, bbox_inches='tight')
        print("Ramachandran plot created")
        print('-----------------------------------\n')

    @hlp.timeit
    def ramachandran_calc_centroid(self, selection='backbone'):
        print('Ramachandran centroid calc has been called\n')
        print('------------------------------------------\n')

        self.called_ramachandran_centroid_calc = True

        self.centroid_topology = self.centroid_conf.topology

        self.centroid_selection = self.centroid_topology.select(selection)

        self.centroid_new_traj = self.centroid_conf.atom_slice(atom_indices=self.centroid_selection)

        self.atoms_centroid, self.bonds_centroid = self.centroid_new_traj.topology.to_dataframe()

        self.phi_indices_centroid, self.phi_angles_centroid = md.compute_phi(self.centroid_conf, periodic=False)
        self.psi_indices_centroid, self.psi_angles_centroid = md.compute_psi(self.centroid_conf, periodic=False)

        self.angles_calc_centroid_list = []

        for i, y in zip(self.phi_indices_centroid, self.psi_indices_centroid):
            temp = md.compute_dihedrals(self.centroid_conf, [i, y])
            self.angles_calc_centroid_list.append(temp[0])

        self.angles_calc_centroid = np.array(self.angles_calc_centroid_list, dtype=np.float64)

        print('------------------------------------------\n')

    @hlp.timeit
    def ramachandran_plot_centroid(self):
        from math import pi

        fig = plt.figure(figsize=(7, 7))
        plt.title('Dihedral Map:')
        plt.scatter(self.angles_calc_centroid[:, 0], self.angles_calc_centroid[:, 1], marker='x')
        # cbar = plt.colorbar()
        # cbar.set_label('Time [ps]')
        plt.xlabel(r'$\Phi$ Angle [radians]')
        plt.xlim(-pi, pi)
        plt.ylabel(r'$\Psi$ Angle [radians]')
        plt.ylim(-pi, pi)
        fig.savefig(self.simulation_name + '_' + 'Ramachandran_analysis_centroid' + '.png', dpi=600,
                    bbox_inches='tight')
        print("Ramachandran plot created")
        print('-----------------------------------\n')

    ####################################################################################################################


    # gmx trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_noPBC.xtc -pbc mol -ur compact

    # gmx trjconv -s md_0_3.tpr -f md_0_3_noPBC.xtc -o md_0_3_clear.xtc -fit rot+trans


    # def get_gmx_command(self):
    #     sim1_file_tpr = sim1 + '/md_0_3.tpr'
    #
    #     # In[39]:
    #
    #     sim1_out = sim1 + '/md_sim1.pdb'
    #
    #     # In[40]:
    #
    #     index = sim1 + '/index.ndx'
    #
    #     # In[41]:
    #
    #     trj_conv = 'gmx trjconv -f {0} -s {1} -n {2} -o {3} -dt 500'.format(sim1_file_traj, sim1_file_tpr, index,
    #                                                                         sim1_out)
    #


    # # traj_sim1_hbonds = md.load(sim1_out)
    #
    #
    # # In[44]:
    #
    # # traj_sim1_hbonds
    #
    #
    # # In[45]:
    #
    # sim1_clear = sim1 + '/md_sim1_clear.pdb'
    #
    # # In[46]:
    #
    # traj_sim1_hbonds = md.load_pdb(sim1_clear)
    #
    # # In[47]:
    #
    # traj_sim1_hbonds
    #
    # # In[48]:
    #
    # traj_sim1_hbonds[-1].save('QRC_sim0_lastFrame.pdb')
    #
    # # In[49]:
    #
    # traj_sim1_hbonds[0].save('QRC_sim0_firstFrame.pdb')
    #
    # # In[50]:
    #
    # traj_sim1_hbonds[0:-1:30].save('QRC_sim0_shortAnimation.pdb')
    #
    # # In[51]:
    #
    # hbonds = md.baker_hubbard(traj_sim1_hbonds, freq=0.8, periodic=False)
    #
    # # In[52]:
    #
    # hbonds = md.wernet_nilsson(traj_sim1_hbonds[-1], periodic=True)[0]
    #
    # # In[53]:
    #
    # sel
    #
    # # In[54]:
    #
    # # for hbond in hbonds:
    # #     # print(hbond)
    # #     print(label(hbond))
    #
    #
    # # In[55]:
    #
    # da_distances = md.compute_distances(traj_sim1_hbonds, hbonds[:, [0, 2]], periodic=False)
    #
    # # In[56]:
    #
    # import itertools
    #
    # # In[57]:
    #
    # color = itertools.cycle(['r', 'b', 'gold'])
    # for i in [2, 3, 4]:
    #     plt.hist(da_distances[:, i], color=next(color), label=label(hbonds[i]), alpha=0.5)
    # plt.legend()
    # plt.ylabel('Freq');
    # plt.xlabel('Donor-acceptor distance [nm]')
    #
    # # TEST ORIGIANL EXAMPLE
    # #
    #
    # # Check for HSL_LasR_1
    #
    # # In[ ]:





    def get_data_for_analysis(self):
        return self.analysis_structure

    def drawVectors(self, transformed_features, components_, columns, plt, scaled):
        if not scaled:
            return plt.axes()  # No cheating ;-)

        num_columns = len(columns)

        # This funtion will project your *original* feature (columns)
        # onto your principal component feature-space, so that you can
        # visualize how "important" each one was in the
        # multi-dimensional scaling

        # Scale the principal components by the max value in
        # the transformed set belonging to that component
        xvector = components_[0] * max(transformed_features[:, 0])
        yvector = components_[1] * max(transformed_features[:, 1])

        ## visualize projections

        # Sort each column by it's length. These are your *original*
        # columns, not the principal components.
        important_features = {columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2) for i in range(num_columns)}
        important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
        print("Features by importance:\n", important_features)

        ax = plt.axes()

        for i in range(num_columns):
            # Use an arrow to project each original feature as a
            # labeled vector on your principal component axes
            plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
            plt.text(xvector[i] * 1.2, yvector[i] * 1.2, list(columns)[i], color='b', alpha=0.75)

        return ax

    # test code
    @hlp.timeit
    def rmsf_calc(self, target=None, reference=None, frame=0, wrt=False, atom_indices=None, ref_atom_indices=None):
        '''
        use backbone for selection


        Looks like GROMACS uses WRT
        '''

        self.called_rmsf_calc = True
        print('RMSF analysis has been called\n')
        print('-----------------------------\n')

        self.topology = self.full_traj.topology
        atom_indices = self.topology.select(atom_indices)

        ref_atom_indices_name = ref_atom_indices

        ref_atom_indices = self.topology.select(ref_atom_indices)

        self.atom_indices = atom_indices
        self.ref_atom_indices = ref_atom_indices

        #  this is for keeping selection from trajectory
        # self.full_traj.restrict_atoms(self.selection)

        self.sim_time = self.full_traj.time / 1000

        trajectory = self.full_traj
        trajectory.superpose(self.full_traj[frame], atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)

        if wrt is True:
            avg_xyz = np.mean(trajectory.xyz[:, atom_indices, :], axis=0)
            self.avg_xyz = avg_xyz
            self.sim_rmsf = np.sqrt(3 * np.mean((trajectory.xyz[:, atom_indices, :] - avg_xyz) ** 2, axis=(0, 2)))
        else:
            reference = trajectory[frame]
            self.sim_rmsf = np.sqrt(
                3 * np.mean((trajectory.xyz[:, atom_indices, :] - reference.xyz[:, ref_atom_indices, :]) ** 2,
                            axis=(0, 2)))

        self.rmsf_analysis_data.update({ref_atom_indices_name: {'atom_indices': self.atom_indices,
                                                                'ref_atom_indices': self.ref_atom_indices,
                                                                'rmsf': self.sim_rmsf}})

        print('-----------------------------\n')

        return self.sim_rmsf

    @hlp.timeit
    def pca_analysis(self):
        scaleFeatures = False

        df = self.data_for_analysis
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(df)
        T = pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T)

        T.columns = ['component1', 'component2']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T

    @hlp.timeit
    def pca_analysis_reshape(self):
        scaleFeatures = False

        df = self.data_for_analysis
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(df)
        T = pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T)

        T.columns = ['component1', 'component2']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T

    @hlp.timeit
    def iso_analysis(self, n_neighbours=3):

        scaleFeatures = False

        df = self.data_for_analysis

        from sklearn import manifold
        iso = manifold.Isomap(n_neighbours, n_components=2)
        iso.fit(df)
        manifold = iso.transform(df)
        # Plot2D(manifold, 'ISOMAP 0 1', 0, 1, num_to_plot=40)
        # Plot2D(manifold, 'ISOMAP 1 2', 1, 2, num_to_plot=40)

        # ax = self.drawVectors(manifold, iso.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(manifold)
        T.columns = ['component1', 'component2']
        T.plot.scatter(x='component1', y='component2', marker='o', alpha=0.75)  # , ax=ax)
        plt.show()

    @hlp.timeit
    def hdbscan_pca(self):
        # fignum = 2
        # fig = plt.figure(fignum)
        # plt.clf()

        # plt.subplot(321)

        X = self.pca_data

        db = hdbscan.HDBSCAN(min_cluster_size=200)
        labels = db.fit_predict(X)
        print('labels ', labels)
        #
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_

        # print('labels is ',labels)
        print('labels shape is ', labels.shape[0])
        # print('db  are  ',db.components_)
        labelsShape = labels.shape[0]

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # plot_frequency(labels)

        print('Estimated number of clusters: %d' % n_clusters_)
        unique_labels = list(set(labels))
        print('Unique labels ', unique_labels)

        worthy_data = labels[labels != -1]
        notWorthy_data = labels[labels == -1]
        real_labels = set(worthy_data)
        # print("Worthy Data ",worthy_data)
        print("Real Labels man ", real_labels)
        shape_worthy = worthy_data.shape[0]
        print("All Worthy data points ", int(shape_worthy))
        print("Not Worthy data points ", int(notWorthy_data.shape[0]))

        # plt.cla()

        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        # print("Colors is ",colors)

        # Here could be the solution
        dtype = [('label', np.int8), ('CLx', np.float64), ('CLy', np.float64), ('CLz', np.float64),
                 ('bindMean', np.float64),
                 ('bindStd', np.float64), ('quantity', int), ('percentage', np.float64), ('rmsd', np.float64), ]
        cluster_Center_Data = np.empty((0,), dtype=dtype)  # This is for clusters
        # print("cluster_Center_Data ",clean_Data, clean_Data.shape)
        # print("clean Data dtype ", clean_Data.dtype)
        # print("clean Data [0] dtype" ,dtype[0])

        label_percent = {}
        # Need to return X, clean_data, and another dict for best position


        molOrder = {}
        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            xyz = X[labels == k]
            if k == -1:
                color = 'b'
                # print('what the hell ', xyz[:, 4])
                plt.scatter(xyz['component1'], xyz['component2'], facecolor=(0, 0, 0, 0), marker='^', s=80, c=color,
                            label='Outlier size={0}'.format(xyz.shape))
                # xyz.plot.scatter(x='component1', y='component2', marker='^',s=100, alpha=0.75)
            else:
                # Need to make this function a lot better
                print('xyz is ', xyz)

                plt.scatter(xyz['component1'], xyz['component2'], marker='o', s=120, c=colors[k], edgecolor='g',
                            label="size={0}".format(xyz.shape))
                # label="deltaG = %s±%s (%s%%) label=%s   rmsd = %s A" % (
                #     round(bind_mean, 2), round(bind_std, 2), percentage, k, curr_rmsd))
                # xyz.plot.scatter(x='component1', y='component2', marker='o', s=100, c=alpha=0.75)

        # plt.set_xlabel('X')
        # plt.set_ylabel('Y')
        # plt.set_zlabel('Z')

        plt.legend(loc='lower left', ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        plt.title('Estimated number of clusters: %d (%d/%d)' % (n_clusters_, shape_worthy, X.shape[0]))
        plt.show()  # not now

    @hlp.timeit
    def silhouette_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.sil_pca
        criteria_name = 'Mean Silhouette Coefficient for all samples'
        score_text = 'Objects with a high silhouette value are considered well clustered'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    @hlp.timeit
    def calinski_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.calinski_pca
        criteria_name = 'Calinski-Harabaz score'
        score_text = 'Objects with a high Calinski-Harabaz score value are considered well clustered'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    @hlp.timeit
    def dunn_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.dunn_pca
        criteria_name = "Dunn's Index"
        score_text = "Maximum value of the index represents the right partitioning given the index"
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

    @hlp.timeit
    def dbi_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.dbi_pca
        criteria_name = 'Davis-Bouldain Index'
        score_text = 'The optimal clustering solution has the smallest Davies-Bouldin index value.'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)

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
        clust_num = max(cluster_dict.items(), key=operator.itemgetter(1))[0]

        print("number of clusters is ", clust_num)

        return clust_num

    @hlp.timeit
    def collect_cluster_info(self):
        data = self.clusters_info[self.clust_num]
        print(data)

        labels = data['labels']
        # Make more flexible whether pca_data or not
        pca_data = self.full_traj
        original_data = self.analysis_structure  # self.pca_data

        cluster_list = {}
        unique_labels = list(set(labels))
        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                cluster_data = []
                xyz = original_data[labels == k]
                model_num = xyz['ModelNum']
                for i in model_num:
                    # print(i)
                    temp_data = self.equiv_models[i]
                    cluster_data.append(temp_data)
                    # print(xyz.describe())
                cluster_list.update({k: cluster_data})
        # print(cluster_list)
        return cluster_list

    # def write_model_to_file(self, model,  resnum=None, filename_pdb=None):
    #     curr_df = model['molDetail']['dataframe']
    #     pdb_tools.write_lig(curr_df, resnum, filename_pdb)


    def save_analysed_data(self, filename):
        '''
        :param filename: Saves clustered data to pickle file
        :return:
        '''
        # import json
        # with open(filename, 'w') as outfile:
        #     json.dump(self.cluster_models, outfile)
        import pickle
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        pickle.dump(self, open(filename, "wb"))

    #  should I add json saving of information or not?
    def load_analysed_data(self, filename):
        '''

        :param filename: load pickle file
        :return:
        '''
        self.analysed_data = pickle.load(open(filename, "rb"))
        print('test')

    #  create another function that shows only the best plot for kmeans
    @hlp.timeit
    def show_silhouette_analysis_pca_best(self, show_plot=False, custom_dpi=300):

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

        fig.savefig(self.simulation_name + '_' + 'Best_cluster_analysis_simple_md_' + '.png', dpi=custom_dpi,
                    bbox_inches='tight')
        if show_plot is True:
            plt.show()

    @hlp.timeit
    def silhouette_analysis_pca(self, show_plots=False):

        self.sil_pca = []
        self.calinski_pca = []
        self.dunn_pca = []
        self.dbi_pca = []
        X = self.pca_data

        for n_clusters in self.range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            centers = clusterer.cluster_centers_

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)

            calinski_avg = calinski_harabaz_score(X, cluster_labels)

            # looks like this is ok
            dunn_avg = dunn_fast(X, cluster_labels)

            converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)
            david_bouldain = davisbouldin(converted_values, centers)

            # pseudo_f = pseudoF_permanova(X, cluster_labels)
            # print("For n_clusters =", n_clusters,
            #       "The pseudo_f is :", pseudo_f)


            print("For n_clusters =", n_clusters,
                  "The average dunn is :", dunn_avg)

            print("For n_clusters =", n_clusters,
                  "The average dbd is :", david_bouldain)

            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            print("For n_clusters =", n_clusters,
                  "The average calinski_harabaz_score is :", calinski_avg)

            # Store info for each n_clusters
            # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
            #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
            #                                         'labels': cluster_labels, 'centers': centers}})

            # Make decision based on average and then round value that would be your cluster quanity

            print('------------------------------------------------------------')

            self.sil_pca.append(silhouette_avg)
            self.calinski_pca.append(calinski_avg)
            self.dunn_pca.append(dunn_avg)
            self.dbi_pca.append(david_bouldain)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                                    'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                                    'labels': cluster_labels, 'centers': centers,
                                                    'silhouette_values': sample_silhouette_values}})

            if show_plots is True:
                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)

                fig.set_size_inches(18, 7)

                # The 1st subplot is the silhouette plot
                # The silhouette coefficient can range from -1, 1 but in this example all
                # lie within [-0.1, 1]
                ax1.set_xlim([-1, 1])
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                y_lower = 10
                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
                ax2.scatter(X['component1'], X['component2'], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors)

                # Labeling the clusters
                centers = clusterer.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1],
                            marker='o', c="white", alpha=1, s=100)

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=100)

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')

                plt.show()

    @hlp.timeit
    def silhouette_analysis(self):
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        X = self.pca_data

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns



            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X['X'], X['Y'], X['Z'], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors)

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],
                        marker='o', c="white", alpha=1, s=200)

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.show()

    @hlp.timeit
    def plotHist(self):
        self.analysis_structure['BindingEnergy'].plot.hist()
        plt.show()

    @hlp.timeit
    def MeanShift(self):

        # print(X.describe)
        bandwidth = estimate_bandwidth(X)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)

        import matplotlib.pyplot as plt
        from itertools import cycle

        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    @hlp.timeit
    def plot_results(self, X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        plt.xlim(-9., 5.)
        plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    @hlp.timeit
    def VBGMM(self):
        X = self.pca_data
        gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
        self.plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                          'Gaussian Mixture')

        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                                covariance_type='full').fit(X)
        self.plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                          'Bayesian Gaussian Mixture with a Dirichlet process prior')

        plt.show()

    @hlp.timeit
    def transform_for_analysis(self):
        model = 1

        columns_dock_center = ['ModelNum', 'X', 'Y', 'Z', 'BindingEnergy']

        dock_df = pd.DataFrame(columns=columns_dock_center)

        for i in sorted(self.samples_data.keys()):
            models = self.samples_data[i]
            # print(model)
            for y in models.mol_data__:
                # This should be the structure for equivalency of models
                # print(model, i, y)
                self.equivalent_models.update({model: {'file': i, 'modelNum': y,
                                                       'molDetail': models.mol_data__[y]}})

                curr_model = models.mol_data__[y]
                curr_frame = curr_model['dataframe']
                curr_x = curr_frame['X'].mean()
                curr_y = curr_frame['Y'].mean()
                curr_z = curr_frame['Z'].mean()
                curr_bind = curr_model['vina_info'][0]

                dock_df.loc[model] = [int(model), curr_x, curr_y, curr_z, curr_bind]
                # print(y, models.mol_data__[y]['dataframe'])
                model += 1
        # print(self.equivalent_models)

        dock_df['ModelNum'] = dock_df['ModelNum'].astype(int)
        return dock_df

    def get_mol_data(self):
        return self.mol_data__

    @hlp.timeit
    def transform_data(self):
        mol_data = {}
        for model, model_info in zip(self.object, self.info):
            # print(model_info)
            pandas_model = self.pandas_transformation(model)
            mol_data.update({model_info[0]: {'dataframe': pandas_model, 'vina_info': model_info[1:]}})

        return mol_data

    @hlp.timeit
    def pandas_transformation(self, list_object_mol):

        columns_pdbqt = ['ATOM', 'SerialNum', 'AtomName', 'ResidueName', 'ChainId',
                         'ChainNum', 'X', 'Y', 'Z', 'Occupancy', 'TempFactor', 'Charge', 'ElemSymbol']

        self.df = pd.DataFrame(list_object_mol, columns=columns_pdbqt)

        self.df['X'] = pd.to_numeric(self.df['X'])
        self.df['Y'] = pd.to_numeric(self.df['Y'])
        self.df['Z'] = pd.to_numeric(self.df['Z'])

        self.df['Charge'] = pd.to_numeric(self.df['Charge'])

        return self.df

    def save_pretty_info(self):
        pass

    def save_json_info(self):
        pass

    def load_molecule(self, load_way='molmolpy'):
        """
        Load molecule whether using molmolpy or pybel
        Parameters
        ----------
        load_way : str, optional
            use molmolpy or pybel version
        """
        pass

    def write_molecule(self, write_way='molmolpy'):
        """
        Write molecule whether using molmolpy or pybel to file
        Parameters
        ----------
        write_way : str, optional
            use molmolpy or pybel version
        """
        pass

    ####################################################################################################################

    def get_trajectory(self):
        return self.full_traj

    # TODO Computing native contacts with MDTraj

    def compute_best_hummer_q(self, selection, frame_native):

        self.called_compute_best_hummer_q = True
        print('Native Contacts has been called\n')
        print('-------------------------------\n')

        traj = self.full_traj[:]
        topology = traj.topology

        selection = topology.select(selection)

        traj.restrict_atoms(selection)

        self.q = protein_analysis.best_hummer_q(traj, traj[frame_native])

        print('-------------------------------\n')

    def plot_native_contacts(self):
        import matplotlib.pyplot as plt
        plt.plot(self.q)
        plt.xlabel('Frame', fontsize=14)
        plt.ylabel('Q(X)', fontsize=14)
        plt.show()

    ####################################################################################################################

    # TODO Parse real experimental NMR data
    def parse_experimental_nmr(self, filename):
        self.exper_nmr = pd.read_csv(filename)


        # test = 1
        #
        # d = pd.read_csv(filename,  index_col=False, header=None).drop([3], axis=1)
        # #
        # d = d.rename(columns={0: "resSeq", 1: "resName", 2: "name"})
        # d = d.drop("resName", axis=1)
        # d = d.set_index(["resSeq", "name"])

        # TODO How to select all N self.exper_nmr[self.exper_nmr['Atom_ID'] == 'N']


        #

    def prepare_nmr_data(self, res_start, res_end, atom_type, errorbar=True,
                         custom_dpi=600):

        type_experimental_data = self.exper_nmr[self.exper_nmr['Atom_ID'] == atom_type]

        # list(self.nmr_shifts_sparta.index)

        index_list = list(self.nmr_shifts_sparta.index)
        self.index_list_sparta = index_list

        get_type_rows_name = []
        self.residues_sparta = []

        for i in range(len(index_list)):
            if index_list[i][-1] == atom_type:
                get_type_rows_name.append(i)
                self.residues_sparta.append(index_list[i][0])

        self.get_type_rows_name_sparta = get_type_rows_name

        sparta_type = self.nmr_shifts_sparta.iloc[get_type_rows_name, :]  # select list index
        self.sparta_type = sparta_type

        index_list = list(self.nmr_shifts_shift2x.index)
        self.index_list_shiftx2 = index_list

        get_type_rows_name = []
        self.residues_shiftx2 = []

        for i in range(len(index_list)):
            if index_list[i][-1] == atom_type:
                get_type_rows_name.append(i)
                self.residues_shiftx2.append(index_list[i][0])

        self.get_type_rows_name_shiftx2 = get_type_rows_name

        shiftx2_type = self.nmr_shifts_shift2x.iloc[get_type_rows_name, :]  # select list index
        self.shiftx2_type = shiftx2_type

        self.sparta_mean = sparta_type.mean(axis=1).values.tolist()
        self.shiftx2_mean = shiftx2_type.mean(axis=1).values.tolist()

        self.sparta_std = sparta_type.std(axis=1).values.tolist()
        self.shiftx2_std = shiftx2_type.std(axis=1).values.tolist()

        self.residues = type_experimental_data['Comp_index_ID']

        self.experimental_val = type_experimental_data['Val']
        self.experimental_val = self.experimental_val.convert_objects(convert_numeric=True)

        self.experimental_error = type_experimental_data['Val_err']
        self.experimental_error = self.experimental_error.convert_objects(convert_numeric=True)

        test = 1

    def plot_whole_nmr(self, atom_type, errorbar=False, custom_dpi=600):

        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(14, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # residues = list(range(res_start, res_end+1))



        colors = sns.cubehelix_palette(n_colors=3, start=2.8, rot=.1)
        sns.set(font_scale=2)

        ax = plt.subplot(111)

        ax.plot(self.residues, self.experimental_val, marker='s', markersize=8, color=colors[0],
                label='Experimental')

        ax.plot(self.residues_sparta, self.sparta_mean, marker='^', linestyle='--', markersize=8, color=colors[1],
                label='Sparta+')

        ax.plot(self.residues_shiftx2, self.shiftx2_mean, marker='v', linestyle='-.', markersize=8, color=colors[2],
                label='ShiftX2')

        if errorbar is True:
            ax.errorbar(self.residues, self.experimental_val, yerr=self.experimental_error, color=colors[0],
                        linewidth=0.6, label='Experimental', capsize=4, elinewidth=2)

            ax.errorbar(self.residues_sparta, self.sparta_mean[res_start - 1: res_end],
                        yerr=self.sparta_std[res_start - 1: res_end], color=colors[1],
                        linewidth=0.8, label='Sparta+', capsize=4, elinewidth=2)

            ax.errorbar(self.residues_shiftx2, self.shiftx2_mean[res_start - 1: res_end],
                        yerr=self.shiftx2_std[res_start - 1: res_end], color=colors[2],
                        linewidth=1.0, label='ShiftX2', capsize=4, elinewidth=2)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel('Residue')
        plt.ylabel('Chemical shift value(ppm)')  # fix Angstrom need to change to nm

        if atom_type == 'CA':
            atom_title = r'C$\alpha$'
        else:
            atom_title = atom_type

        title = 'Chemical shift values for {0}-atom vs. residue number'.format(atom_title)
        plt.title(title)
        # plt.legend(loc='lower center')
        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(flip(handles, 3), flip(labels, 3), loc=4, ncol=3)

        min_y = min(self.experimental_val)

        ax.set_ylim(min_y - 4)

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(self.simulation_name + '_' + title + '.png', dpi=custom_dpi, bbox_inches='tight')

        print('NMR comparison plot created')
        print('-----------------------------------\n')

    def plot_errorbar_nmr(self, atom_type, min_x=1, max_x=171, errorbar=False, custom_dpi=600):

        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(14, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # residues = list(range(res_start, res_end+1))



        colors = sns.cubehelix_palette(n_colors=3, start=2.8, rot=.1)
        sns.set(font_scale=2)

        ax = plt.subplot(111)

        # ax.plot(self.residues, self.experimental_val, marker='s',markersize=8,color=colors[0],
        #        label='Experimental')
        #
        #
        #
        #
        # ax.plot(self.residues_sparta, self.sparta_mean, marker='^',linestyle='--',markersize=8,  color=colors[1],
        #        label='Sparta+')
        #
        # ax.plot(self.residues_shiftx2, self.shiftx2_mean, marker='v', linestyle='-.', markersize=8, color=colors[2],
        #           label='ShiftX2')

        ax.errorbar(self.residues, self.experimental_val, yerr=self.experimental_error, color=colors[0],
                    linewidth=0.6, label='Experimental', elinewidth=2, fmt='o', zorder=10, capthick=1, capsize=2)

        ax.errorbar(self.residues_sparta, self.sparta_mean, yerr=self.sparta_std, color=colors[1],
                    linewidth=0.8, label='Sparta+', elinewidth=0.5, fmt='^', zorder=1, capthick=1, capsize=2)

        ax.errorbar(self.residues_shiftx2, self.shiftx2_mean, yerr=self.shiftx2_std, color=colors[2],
                    linewidth=1.0, label='ShiftX2', elinewidth=0.7, fmt='v', zorder=2, capthick=1, capsize=2)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel('Residue')
        plt.ylabel('Chemical shift value(ppm)')  # fix Angstrom need to change to nm

        if atom_type == 'CA':
            atom_title = r'C$\alpha$'
        else:
            atom_title = atom_type

        title = 'Chemical shift values for {0}-atom vs. residue number'.format(atom_title)
        plt.title(title)
        # plt.legend(loc='lower center')
        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(flip(handles, 3), flip(labels, 3), loc=4, ncol=3)

        min_y = min(self.experimental_val)

        # TODO need to think about this for errorbar
        ax.set_ylim(min_y - 4)

        ax.set_xlim(min_x - 5, max_x)

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(self.simulation_name + '_' + title + '_errorBar.png', dpi=custom_dpi, bbox_inches='tight')

        # TODO similarity test for NMR data
        # TODO need to fix shape  ValueError: Incompatible dimension for X and Y matrices: X.shape[1] == 159 while Y.shape[1] == 229
        # from sklearn.metrics.pairwise import paired_euclidean_distances
        #
        # sim1 = paired_euclidean_distances(self.experimental_val, self.sparta_mean)
        # sim2 = paired_euclidean_distances(self.experimental_val, self.shiftx2_mean)
        #
        # print('Similarity between experiment and sparta+ ', sim1)
        # print('Similarity between experiment and shiftx2 ', sim2)
        #
        #
        #
        print('NMR comparison plot created')
        print('-----------------------------------\n')

    def plot_nmr_jointplot(self, atom_type='CA', res_start=1, res_end=171):
        sns.set(font_scale=2)

        pd_data1 = converters.convert_data_to_pandas(self.experimental_val, self.sparta_mean[res_start:res_end],
                                                     x_axis_name='Experimental',
                                                     y_axis_name='Sparta+')

        pd_data2 = converters.convert_data_to_pandas(self.experimental_val, self.shiftx2_mean[res_start:res_end],
                                                     x_axis_name='Experimental',
                                                     y_axis_name='ShiftX2')

        g = sns.jointplot(x="Experimental", y="Sparta+", data=pd_data1, kind="reg")

        if atom_type == 'CA':
            atom_title = r'C$\alpha$'
        else:
            atom_title = atom_type

        title = 'Chemical shift values for {0}-atom. Experimental vs Sparta+'.format(atom_title)
        g.fig.suptitle(title)

        f = sns.jointplot(x="Experimental", y="ShiftX2", data=pd_data2, kind="reg")
        title = 'Chemical shift values for {0}-atom. Experimental vs ShiftX2'.format(atom_title)
        f.fig.suptitle(title)
        sns.plt.show()

    # TODO Computing NMR SHIFTS with Sparta+ with MDTraj

    def calc_nmr_shifts(self, selection, from_frame=0, to_frame=-1, use_cluster_data=True,
                        save_data=None, load_data=None):
        print('Calculating nmr shifts ------>>>>>>>>>')
        if use_cluster_data is False:
            traj = self.full_traj[:]
            topology = traj.topology

            selection = topology.select(selection)

            traj.restrict_atoms(selection)
        else:
            traj = self.max_clust_temp_data

        curr_traj = traj[from_frame:to_frame]
        print("Trajectory length to analyze is ", len(curr_traj))
        print('Time to do it !!!!!!!!!!!!!!!!!!!!!!\n')
        self.nmr_shifts_sparta = protein_analysis.calc_shifts_nmr(curr_traj)
        self.nmr_shifts_shift2x = protein_analysis.calc_shifts_nmr(curr_traj, tool_to_use='shift2x')
        # self.nmr_shifts_ppm = protein_analysis.calc_shifts_nmr(traj[from_frame:to_frame], tool_to_use='ppm')

        print(self.nmr_shifts_sparta)
        print('---------------------')
        print(self.nmr_shifts_shift2x)
        print('---------------------')

        self.nmr_shift_data = {'sparta': self.nmr_shifts_sparta,
                               'shift2x': self.nmr_shifts_shift2x}

        if save_data is not None:
            filehandler = open("{0}".format(save_data), "wb")
            pickle.dump(self.nmr_shift_data, filehandler)
            filehandler.close()

            # TODO save to HDF format

            hdf = HDFStore("{0}_hdf.h5".format(save_data))

            hdf.put('sparta', self.nmr_shifts_sparta)
            hdf.put('shiftx2', self.nmr_shifts_shift2x)

            print('Sparta')
            print(hdf['sparta'].shape)
            print('Shiftx2')
            print(hdf['shiftx2'].shape)
            hdf.close()  # closes the file

    def load_pre_calc_nmr_shifts(self, load_data=None):
        if load_data is not None:

            if 'pickle' in load_data:
                file = open(load_data, 'rb')
                self.nmr_shift_data = pickle.load(file)
                file.close()

                # import hickle as hkl
                #
                # hkl.dump(self.nmr_shift_data, 'data_hdf.hkl', mode='w')
                # hkl.dump(self.nmr_shift_data, 'data_hdf_gzip.hkl', mode='w', compression='gzip')


                self.nmr_shifts_sparta = self.nmr_shift_data['sparta']
                self.nmr_shifts_shift2x = self.nmr_shift_data['shift2x']
            elif 'h5' in load_data:

                from pandas import HDFStore, DataFrame
                hdf = HDFStore(load_data)

                self.nmr_shifts_sparta = hdf['sparta']
                self.nmr_shifts_shift2x = hdf['shiftx2']
                hdf.close()

            print('Saved NMR file loaded')
            print('---------------------------------------------------------------------------------------------------')
            # self.nmr_shifts_sparta.to_csv('sparta.csv')
            # # self.nmr_shifts_sparta.to_hdf('sparta.h5', 'table', append=True)
            # self.nmr_shifts_shift2x.to_csv('shiftx2.csv')
            # self.nmr_shifts_shift2x.to_hdf('shiftx2.h5', 'table', append=True)


            # # TODO this is buggy
            # from pandas import HDFStore, DataFrame
            # hdf = HDFStore('LasR_full.h5',  complib='bzip2')
            #
            # hdf.put('sparta', self.nmr_shifts_sparta)
            # hdf.put('shiftx2', self.nmr_shifts_shift2x)
            #
            # print('Sparta')
            # print(hdf['sparta'].shape)
            # print('Shiftx2')
            # print(hdf['shiftx2'].shape)
            # hdf.close()  # closes the file



            # print(self.nmr_shifts_ppm)
            # print('---------------------')

    ####################################################################################################################

    # TODO calculate solvent area
    @hlp.timeit
    def calc_solvent_area(self, selection, from_frame=0, to_frame=-1, stride=20, parallel=True, n_sphere_points=960):

        self.called_calc_solvent_area = True
        print('Sasa calculation has been called\n')
        print('-----------------------------\n')

        traj = self.full_traj[:]
        topology = traj.topology

        selection = topology.select(selection)

        traj.restrict_atoms(selection)

        self.sasa_traj = traj[from_frame:to_frame:stride]
        print(self.sasa_traj)

        # TODO start by for single machine ipcluster start -n 4
        self.sasa, self.total_sasa = protein_analysis.calc_sasa(self.sasa_traj, parallel,
                                                                n_sphere_points=n_sphere_points)

        print('-----------------------------\n')

    @hlp.timeit
    def plot_solvent_area(self, show=False):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.sasa_traj.time, self.total_sasa)
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('Total SASA (nm)^2', size=16)

        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_plot.png', dpi=300, bbox_inches='tight')

    @hlp.timeit
    def plot_solvent_area_frame(self, frame, show=False):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.sasa_traj.time, self.sasa[frame])
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('Total SASA (nm)^2', size=16)
        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_plot_{0}.png'.format(frame), dpi=300, bbox_inches='tight')

    @hlp.timeit
    def plot_solvent_area_autocorr(self, show=False):
        self.sasa_autocorr = protein_analysis.autocorr(self.total_sasa)
        fig = plt.figure(figsize=(10, 10))
        plt.semilogx(self.sasa_traj.time, self.sasa_autocorr)
        plt.xlabel('Time [ps]', size=16)
        plt.ylabel('SASA autocorrelation', size=16)

        if show is True:
            plt.show()
        fig.savefig(self.simulation_name + '_' + 'SASA_autocorrelation.png', dpi=300, bbox_inches='tight')

    # TODO  show PCA transformation

    @hlp.timeit
    def plot_simple_md_pca_analysis(self, custom_dpi=600, show=False):
        sns.set(style="ticks", context='paper')

        # cmap = sns.cubehelix_palette(n_colors=len(self.pca_traj.time), as_cmap=True, reverse=True)
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=self.pca_traj.time)
        # plt.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=cmap)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Conformation PCA Analysis')
        cbar = plt.colorbar()
        cbar.set_label('Time [ps]')
        fig.savefig(self.simulation_name + '_' + 'simple_PCA_analysis' + '.png', dpi=custom_dpi, bbox_inches='tight')
        if show is True:
            plt.show()
        print("simple PCA plot created")
        print('-----------------------------------\n')

    ####################################################################################################################

    # TODO kmeans cluster analysis on PCA data
    @hlp.timeit
    def md_pca_full_analysis(self, show=False, algorithm='kmeans'):
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

    ####################################################################################################################
    # TODO show cluster scoring
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

    ####################################################################################################################

    # TODO  compute dssp and plot

    #  only need to select protein for dssp analysis
    @hlp.timeit
    def compute_dssp(self, selection, simplified_state=False):

        self.called_compute_dssp = True
        print("DSSP has been called\n")
        print('-----------------------------------\n')

        self.dssp_traj = self.full_traj[:]
        self.dssp_topology = self.dssp_traj.topology

        self.dssp_selection = self.dssp_topology.select(selection)

        self.dssp_traj.restrict_atoms(self.dssp_selection)
        self.dssp_data = md.compute_dssp(self.dssp_traj, simplified=simplified_state)

        # indexes =  self.dssp_traj.time/1000
        self.dssp_df = pd.DataFrame(self.dssp_data)

        print('-----------------------------------\n')

    @hlp.timeit
    def plot_dssp(self, title='LasR DSSP',
                  xlabel=r"Time $t$ (ns)",
                  ylabel=r"Residue",
                  x_stepsize=200,
                  custom_dpi=600):
        '''

        ylabel=r"C$_\alpha$ RMSF from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        :param title:
        :param xlabel:
        :param ylabel:
        :param custom_dpi:
        :return:
        '''
        sns.set(style="ticks", context='paper')

        # create dictionary with value to integer mappings
        value_to_int = {value: i for i, value in enumerate(sorted(pd.unique(self.dssp_df.values.ravel())))}

        f, ax = plt.subplots()

        self.dssp_plot_data = self.dssp_df.replace(value_to_int).T
        self.dssp_plot_data = self.dssp_plot_data.iloc[::-1]

        cmap = sns.cubehelix_palette(n_colors=len(value_to_int), as_cmap=True, reverse=True)

        hm = sns.heatmap(self.dssp_plot_data, cmap=cmap, ax=ax, cbar=False)
        # add legend

        x_label_key = []

        sim_time = self.dssp_traj.time / 1000
        start = sim_time.min
        end = sim_time.max

        last_frame = len(sim_time) - 1

        # this is buggy

        for ind, label in enumerate(ax.get_xticklabels()):
            if ind == last_frame:
                label.set_visible(True)
            elif ind % 1000 == 0:  # every 100th label is kept
                label.set_visible(True)
                # label =  round(sim_time[ind])
                # x_label_key.append(ind)
            else:
                label.set_visible(False)
            x_label_key.append(ind)

        for ind, tick in enumerate(ax.get_xticklines()):
            # tick part doesn't work
            if ind == last_frame - 1:
                tick.set_visible(True)
            elif ind % 1000 == 0:  # every 100th label is kept
                tick.set_visible(True)
            else:
                tick.set_visible(False)

        for ind, label in enumerate(ax.get_yticklabels()):
            if ind % 50 == 0:  # every 100th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

        for ind, tick in enumerate(ax.get_yticklines()):
            if ind % 50 == 0:  # every 100th label is kept
                tick.set_visible(True)
            else:
                tick.set_visible(False)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        legend_ax = f.add_axes([.7, .5, 1, .1])
        legend_ax.axis('off')

        # major ticks every 20, minor ticks every 5


        #

        #
        labels = [item.get_text() for item in ax.get_xticklabels()]
        print('labels ', labels)
        labels_new = [round(sim_time[i]) for i in x_label_key]
        print('labels new ', labels_new)
        ax.set_xticklabels(labels_new)

        # reconstruct color map
        # colors = plt.cm.Pastel2(np.linspace(0, 1, len(value_to_int)))
        #
        # colors = sns.color_palette("cubehelix", len(value_to_int) )


        # add color map to legend
        colors = sns.cubehelix_palette(n_colors=len(value_to_int), reverse=True)
        patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in colors]
        legend = legend_ax.legend(patches,
                                  sorted(value_to_int.keys()),
                                  handlelength=0.8, loc='lower left')
        for t in legend.get_texts():
            t.set_ha("left")

        # sns.plt.show()
        f.savefig('DSSP_plot.png', dpi=custom_dpi)


        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # plt.plot(self.atom_indices, self.sim_rmsf, color='b',
        #          linewidth=0.6, label='LasR')
        #
        # plt.legend(loc="best", prop={'size': 8})
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # plt.title(title)
        #
        # # In[28]:
        #
        # fig.savefig(title + '.png', dpi=custom_dpi, bbox_inches='tight')
        #
        # print('RMSD plot created')
        # print('-----------------------------------\n')

    ####################################################################################################################
    # TODO save processed data to pickle for multiple analysis

    def save_processed_data(self):
        self.save_processed_data_info = {}

        if self.called_md_pca_analysis is True:
            self.save_processed_data_info.update({'PCA': self.pca_traj})

        if self.called_rmsd_analysis is True:
            self.save_processed_data_info.update({'rmsd': self.rmsd_analysis_data})
            self.save_processed_data_info.update({'time': self.sim_time})

        if self.called_rg_analysis is True:
            self.save_processed_data_info.update({'Rg': self.rg_analysis_data})

        if self.called_calc_solvent_area is True:
            self.save_processed_data_info.update({'sasa': self.sasa})
            self.save_processed_data_info.update({'totalSasa': self.total_sasa})

        if self.called_compute_dssp is True:
            self.save_processed_data_info.update({'dssp': self.dssp_df})

        if self.called_compute_best_hummer_q is True:
            self.save_processed_data_info.update({'nativeContacts': self.q})

        if self.called_hbond_analysis_count is True:
            self.save_processed_data_info.update({'hbondCount': self.hbond_count})
            self.save_processed_data_info.update({'hbondFrames': self.hbonds_frames})

        if self.called_find_max_cluster_centroid is True:
            self.save_processed_data_info.update({'centroidConf': self.centroid_conf})

        if self.called_ramachandran_centroid_calc is True:
            self.save_processed_data_info.update({'ramachandranCentroid': self.angles_calc_centroid})

        self.save_processed_data_info.update({'simulationName': self.simulation_name})
        import pickle
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        filename = self.simulation_name + '_processed_data.pickle'
        pickle.dump(self.save_processed_data_info, open(filename, "wb"))
