# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    md_analysis.py
# @brief   md_analysis object
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
#from bokeh.core.compat.mplexporter.utils import get_line_style
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
import os
import sys
import pickle
import time
import math

import pylab as plt

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

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
import multiprocessing

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import protein_analysis
from molmolpy.utils import nucleic_analysis
from molmolpy.utils import filter_items
from molmolpy.utils import calculate_rmsd
from molmolpy.utils import filter_items
from molmolpy.utils import pymol_tools

from molmolpy.tools import featurizers

from molmolpy.utils import helper as hlp

from itertools import combinations

import seaborn as sns

import numba
from numba import jit

matplotlib.rcParams.update({'font.size': 12})
# matplotlib.style.use('ggplot')
sns.set(style="white", context='paper')


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)


class MDAnalysisObject(object):
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

    # @profile
    def __init__(self, md_trajectory_file,
                 md_topology_file,
                 load_way='molmolpy',
                 molname='Unknown',
                 receptor_name='Unknown',
                 sim_num=1,
                 k_clust=10):

        self.receptor_name = receptor_name
        self.molecule_name = molname

        if receptor_name != 'Unknown':
            self.simulation_name = 'simulation_' + receptor_name

        if molname != 'Unknown':
            self.simulation_name += '_' + molname

        if receptor_name == 'Unknown' and molname == 'Unknown':
            self.simulation_name = 'simulation_' + 'Unknown'

        self.sim_num = sim_num

        self.simulation_name += '_' + str(sim_num)

        self.initial_name = self.simulation_name

        self.md_trajectory_file = md_trajectory_file

        if 'gro' in md_topology_file or 'pdb' in md_topology_file:
            self.md_topology_file = md_topology_file
        else:
            print('Topology has to be gro or pdb')
            sys.exit(1)

        self.range_n_clusters = list(range(2, k_clust + 1))

        # This part is for checking which methods were called
        self.called_md_pca_analysis = False

        self.called_rmsd_analysis = False

        self.called_rg_analysis = False

        self.called_rmsf_calc = False

        self.called_hbond_analysis_count = False

        self.called_calc_solvent_area = False

        self.called_compute_dssp = False

        self.called_compute_best_hummer_q = False

        self.called_find_max_cluster_centroid = False

        self.called_ramachandran_centroid_calc = False

        self.called_rmsf_analysis = False

        self.called_find_clusters_centroid = False

        self.called_find_clusters_hbond = False

        # DATA ANALYSIS OBJECTS

        self.rmsd_analysis_data = {}
        self.rg_analysis_data = {}
        self.rmsf_analysis_data = {}

        self.sasa_analysis_data = {}

        self.hbond_cluster_analysis_data = {}

        self.center_of_mass_analysis_data = {}

        self.md_pca_analysis_data = {}

        self.md_dssp_analysis_data = {}

        self.md_pre_pca_analysis_data = {}

        self.md_pairwise_rmsd_analysis_data = {}

        self.cluster_selection_analysis_data = {}

        self.cluster_selection_color_data = {}

        self.k_clust = k_clust

        # rpy2 objects part
        # MAYBE CREATE ANOTHER OBJECT
        self.bio3d_traj_filenames = {}

        # TODO need to know which aminoacid residues interact
        self.analyzed_mmpbsa_data = {}

    @hlp.timeit
    def rmsd_analysis_iterative(self, selection):
        '''

        :param selection: has to be mdtraj compatible
        :return:
        '''
        first_frame = md.load_frame(self.md_trajectory_file, 0, top=self.md_topology_file)

        topology = first_frame.topology
        selection = topology.select(selection)
        print('selection is ', selection)

        time_sim1 = []
        rmsds_sim1 = []

        for chunk in md.iterload(self.md_trajectory_file, top=self.md_topology_file, chunk=100):
            rmsds_sim1.append(md.rmsd(chunk, first_frame, atom_indices=selection))
            # print(chunk, '\n', chunk.time)
            time_sim1.append(chunk.time)

        time_sim1_np = np.concatenate(time_sim1) / 1000

        rmsd_sim1_np = np.concatenate(rmsds_sim1)

        self.sim_time = time_sim1_np
        self.sim_rmsd = rmsd_sim1_np

        self.regression_fit_range = 10

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
        # TODO How to prepare for mdtraj
        # mdconvert -t md_0_1.pdb -s 4 -o md_traj.xtc md_0_3_clear.xtc

        print('MD Load  has been called\n')
        print('-------------------------------\n')

        self.full_traj = md.load(self.md_trajectory_file, top=self.md_topology_file,
                                 stride=custom_stride)

        self.sim_time = self.full_traj.time / 1000

        self.mdtraj_loaded = True
        self.mdtraj_deleted = False
        print("Full trajectory loaded successfully")
        print('-----------------------------------\n')

    ####################################################################################################################
    # TODO Bio3D test

    # USE PDB for bio3d, dcd was problematic
    def prep_bio3d(self, selection, n_steps=10):
        selection_text = selection
        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        self.bio3d_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        filename = 'bio3d_{0}.pdb'.format(selection_text)
        print('bio3d filename is ', filename)

        self.bio3d_traj_filenames.update({selection: filename})

        self.bio3d_traj[::n_steps].save(filename)

    def prepare_bio3d_env(self):
        import rpy2.robjects.packages as rpackages

        self.r_utils = rpackages.importr('utils')
        self.r_bio3d = rpackages.importr('bio3d')

        test = 1

        test = self.r_bio3d.read_pdb('bio3d_protein.pdb')

        modes = self.r_bio3d.nma(test)

        print(modes)

    def read_bio3d_traj(self, selection, filename=None):
        test = self.r_bio3d.read_dcd('bio3d_protein.dcd')

        modes = self.r_bio3d.nma(test)

        from rpy2 import robjects
        from rpy2.robjects import Formula
        from rpy2.robjects.vectors import IntVector, FloatVector
        from rpy2.robjects.lib import grid
        from rpy2.robjects.packages import importr

        # The R 'print' function
        rprint = robjects.globalenv.get("print")
        stats = importr('stats')
        grdevices = importr('grDevices')
        base = importr('base')

        self.r_bio3d.mktrj(modes, mode=7)
        pass

    def delete_mdtraj(self):
        del self.full_traj

        import gc
        gc.collect()

        self.mdtraj_loaded = False
        self.mdtraj_deleted = True

    ####################################################################################################################

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


        import gc
        gc.collect()
        len(gc.get_objects())

        print("Rg has been calculated")
        print('-----------------------------------\n')

    @hlp.timeit
    def hbond_frame_calc(self, frame):
        hbonds = md.baker_hubbard(frame, exclude_water=True, periodic=False)
        # print('yay {0}'.format(frame.time, len(hbonds)))
        return len(hbonds)

    # @hlp.timeit
    def count_lig_hbond(self, t, hbonds, ligand):
        label = lambda hbond: '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))

        hbond_atoms = []
        hbond_indexes_sel = []
        hbond_count = 0
        for hbond in hbonds:

            test = 1
            res = label(hbond)
            test = 1
            # print('res ', res)
            if ligand in res:
                # print("verify yolo res is ", res)
                hbond_atoms.append(res)
                hbond_indexes_sel.append(hbond)
                hbond_count += 1
                test = 1
        # print('------------------------------------------------')
        test = 1
        return hbond_atoms, hbond_count, hbond_indexes_sel

    # @hlp.timeit
    def count_lig_hbond_baker(self, t, hbonds, ligand):
        label = lambda hbond: '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))

        hbond_atoms = []
        hbond_indexes_sel = []
        hbond_count = 0

        test = 1
        res = label(hbonds)
        test = 1
        print('res ', res)
        if ligand in res:
            # print("verify yolo res is ", res)
            hbond_atoms.append(res)
            hbond_indexes_sel.append(hbonds)
            hbond_count += 1
            test = 1
            # print('------------------------------------------------')
            test = 1
            return [res, 1, hbonds]
        else:
            return None

    @hlp.timeit
    def hbond_freq_plot_analysis(self,
                                 receptor_name='LasR',
                                 ligand_name='HSL',
                                 title='Simulation',
                                 xlabel=r"Time $t$ (ns)",
                                 ylabel=r"Number of Hydrogen Bonds",
                                 lang='en',
                                 hbond_length=0.4,
                                 custom_dpi=600):
        print('-------------------------------------\n')

        sns.set(style="ticks", context='paper')
        # sns.set(font_scale=2)

        plt.clf()

        fig = plt.figure(figsize=(14, 7))

        title = 'Frequency of H-Bonds between {0}-{1}'.format(receptor_name, ligand_name)

        # fig = plt.figure(figsize=(10, 7))

        fig.suptitle(title, fontsize=16)

        hbonds_frames = self.hbonds_frames

        sim_hbond_atoms = []
        sim_hbond_count = []

        sim_hbond_sel = []

        for hbonds in hbonds_frames:
            hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(self.full_traj, hbonds, ligand_name)

            sim_hbond_atoms.append(hbond_atoms)
            sim_hbond_count.append(hbond_count)

            if len(hbond_indexes_sel) > 0:
                sim_hbond_sel += hbond_indexes_sel

        sim_hbound_np = np.array(sim_hbond_count)

        # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

        sim_hbound_sel_np = np.array(sim_hbond_sel)
        # sim_hbound_sel_np = np.array(updated_indexes)

        unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)

        # self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
        # self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

        # curr_color = self.colors_[i - 1]
        # curr_label = 'Simulation {0}'.format(i)
        curr_label = "Simulation of Cluster  mean: {0}±{1}".format(round(np.mean(sim_hbound_np), 3),
                                                                   round(np.std(sim_hbond_count), 3))
        print('Curr label ', curr_label)
        # This won't work here
        da_distances = md.compute_distances(self.full_traj, unique_bonds[:, [0, 2]], periodic=False)

        # Version 1
        # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
        #          linewidth=0.2, label=curr_label)
        # color = itertools.cycle(['r', 'b', 'gold'])

        colors = sns.cubehelix_palette(n_colors=len(unique_bonds), rot=-.4)
        # self.colors_ = colors
        label = lambda hbond: '%s -- %s' % (
            self.full_traj.topology.atom(hbond[0]), self.full_traj.topology.atom(hbond[2]))

        color = itertools.cycle(['r', 'b', 'gold'])

        for i in range(len(unique_bonds)):
            data = da_distances[:, i]
            data_hbonds = data[data <= hbond_length]

            plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds[i]), alpha=0.5)
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

        fig.savefig(self.simulation_name + '_' + title + '_' + '_' + ligand_name + '_' + lang + '.png',
                    dpi=custom_dpi, bbox_inches='tight')

        # fig.savefig('Multi_Plot_HBOND_frequency_' + '_' + title + '_' + str(i) + '_' + ligand_name + '.png',
        #             dpi=custom_dpi, bbox_inches='tight')

        print('Multi HBond frequency lig  plot created')

    @hlp.timeit
    def find_contacts_freq(self,
                           receptor_name='LasR',
                           ligand_name='HSL',
                           title='Contacts_FREQ',
                           xlabel=r"Time $t$ (ns)",
                           ylabel=r"Number of Hydrogen Bonds",
                           lang='en',
                           hbond_length=0.4,
                           custom_dpi=600):

        print('Find Contacts FREQ is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # sns.set(font_scale=2)
        sns.set(style="ticks", context='paper')

        sns.set(font_scale=1)

        plt.clf()

        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        title = 'contact frequency between {0}-{1} '.format(receptor_name, ligand_name)

        hbonds_frames = md.wernet_nilsson(self.full_traj, exclude_water=True, periodic=False)

        sim_hbond_atoms = []
        sim_hbond_count = []

        sim_hbond_sel = []

        for hbonds in hbonds_frames:
            hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(self.full_traj, hbonds, ligand_name)

            sim_hbond_atoms.append(hbond_atoms)
            sim_hbond_count.append(hbond_count)

            if len(hbond_indexes_sel) > 0:
                sim_hbond_sel += hbond_indexes_sel

        sim_hbound_np = np.array(sim_hbond_count)

        # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

        sim_hbound_sel_np = np.array(sim_hbond_sel)
        # sim_hbound_sel_np = np.array(updated_indexes)

        if len(sim_hbound_np) == 0:
            print('No Hbonds found ')

        if len(sim_hbound_sel_np) == 0:
            print('No Hbonds found ')

        # TODO this part is very important, for removing duplicates
        unique_bonds_list = sim_hbound_sel_np[:, [0, 2]]
        unique_bonds_analysis = filter_items.filter_similar_lists(unique_bonds_list)

        unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)

        # da_distances = md.compute_distances(clust_temp_data, unique_bonds[:, [0, 2]], periodic=False)
        da_distances = md.compute_distances(self.full_traj, unique_bonds_analysis[:, [0, 1]], periodic=False)

        # Version 1
        # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
        #          linewidth=0.2, label=curr_label)
        # color = itertools.cycle(['r', 'b', 'gold'])

        # colors = sns.cubehelix_palette(n_colors=len(unique_bonds), rot=-.4)
        colors = sns.cubehelix_palette(n_colors=len(unique_bonds_analysis), rot=-.4)
        # self.colors_ = colors
        # label = lambda hbond: '%s -- %s' % (
        #     clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[2]))

        label = lambda hbond: '%s -- %s' % (
            self.full_traj.topology.atom(hbond[0]), self.full_traj.topology.atom(hbond[1]))

        # color = itertools.cycle(['r', 'b', 'gold'])

        freq_contact_analysis_data = {'hbonds': hbonds_frames,
                                      'distances': da_distances,
                                      'unique': unique_bonds,
                                      'colors': colors}

        for i in range(len(unique_bonds_analysis)):
            data = da_distances[:, i]
            data_hbonds = data[data <= hbond_length]

            # plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds[i]), alpha=0.5)
            plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds_analysis[i]), alpha=0.5)

        # TODO move legend outside
        # plt.legend(loc='best')
        art = []
        # 0.5, -0.1
        # -1.2, 0.0 Nope
        # 2.0, 0.0 a lot better

        # 1.6, 1.0 close to best
        # 1.5, 1.07 very close
        lgd = plt.legend(loc=9, bbox_to_anchor=(1.5, 1.050), ncol=1)
        art.append(lgd)

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

        print("Len of full traj")
        print(len(self.full_traj))

        print('-------------------')

        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '_' + ligand_name + '_' + lang + '.png',
                    dpi=custom_dpi, bbox_inches='tight', additional_artists=art)

        print('-----------------------------------\n')

    @hlp.timeit
    def find_ligand_neighbours_freq(self,
                                    receptor_name='LasR',
                                    ligand_name='HSL',
                                    title='Contacts_FREQ',
                                    xlabel=r"Time $t$ (ns)",
                                    ylabel=r"Number of Hydrogen Bonds",
                                    lang='en',
                                    num_of_threads=7,
                                    parallel=True,
                                    cutoff_len=0.4,
                                    custom_dpi=600):

        print('Find Neighbour Contacts FREQ is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # sns.set(font_scale=2)
        sns.set(style="ticks", context='paper')

        sns.set(font_scale=1)

        # plt.clf()
        #
        # fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        #
        # title = 'Neighbours contact frequency between {0}-{1} '.format(receptor_name, ligand_name)

        # prepare dataframe topology for parallelization

        traj_topology = self.full_traj.topology
        # function_arguments_to_call = [[neighbour_frame,
        #                                neighbours_data[neighbour_frame]] for
        #                               neighbour_frame in neighbour_frames]
        #
        test = 1
        self.dataframe_topology = traj_topology.to_dataframe()[0]

        self.dataframe_topology_smaller = self.dataframe_topology.drop(['name', 'element', 'chainID', 'segmentID'],
                                                                       axis=1)

        test = 1
        # Calculate Neighbours

        selection_text = ligand_name

        self.topology = self.full_traj.topology

        ligand_atoms = self.topology.select('resname ' + ligand_name)

        test = 1

        self.neighbours_data = md.compute_neighbors(self.full_traj, cutoff_len, ligand_atoms, haystack_indices=None,
                                                    periodic=False)
        print('-----------------------------------\n')

    def filter_ligand_neighbours_freq(self,
                                      receptor_name='LasR',
                                      ligand_name='HSL',
                                      title='Contacts_FREQ',
                                      xlabel=r"Time $t$ (ns)",
                                      ylabel=r"Number of Hydrogen Bonds",
                                      lang='en',
                                      num_of_threads=7,
                                      parallel=True,
                                      save_data=True,
                                      cutoff_len=0.4,
                                      custom_dpi=600):
        # PARALLElIZE TEST
        print('Filter neighbour list\n ------>>>>>>>>>>>')
        self.filter_ligand_name = ligand_name
        self.filter_receptor_name = receptor_name

        if parallel is False:
            results_dict = []
            for frame in range(len(self.full_traj)):
                temp_data = filter_items.run_neighbour_analysis_parallel(frame, self.dataframe_topology_smaller,
                                                                         self.neighbours_data[frame])
                results_dict.append(temp_data)
        else:
            prot = pickle.HIGHEST_PROTOCOL
            print('Highest protocol ', prot)
            neighbour_frames = list(range(len(self.full_traj)))

            # Find what is provoking problem
            function_arguments_to_call = ([frame, self.dataframe_topology_smaller, self.neighbours_data[frame]] for
                                          frame in
                                          neighbour_frames)

            test = 1
            pool = multiprocessing.Pool(num_of_threads)
            # ALWAYS MAKE PARALLEL CALCULATION METHOD SEPARATE FROM CLASS
            results = pool.starmap(filter_items.run_neighbour_analysis_parallel, function_arguments_to_call)

            pool.close()
            pool.join()

            results_dict = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            print('-----------------------------------')
            print(results)
            print('----------=====================================-------------------------')

        total_length = len(self.full_traj)

        # percentage = (top*100)/length

        self.res_seq_frequency_dict, self.frequency_dict, self.freq_stuff = filter_items.filter_neighbour_frequency(
            results_dict,
            total_length)

        data_to_save = {'resSeqFreqDict': self.res_seq_frequency_dict,
                        'freqDict': self.frequency_dict,
                        'freqStuff': self.freq_stuff,
                        'neighboursData': self.neighbours_data}

        self.filtered_neighbours = data_to_save

        if save_data is True:
            save_name = 'filter_ligand_neighbours_freq_{0}_{1}.pickle'.format(self.filter_receptor_name,
                                                                              self.filter_ligand_name)
            filehandler = open("{0}".format(save_name), "wb")

            pickle.dump(data_to_save, filehandler)
            filehandler.close()
            print('Filter data saved----->>>>>\n')

        test = 1
        print('-----------------------------------\n')

    def load_filtered_pickle(self, filename):
        self.load_filter_pickle_data = pickle.load(open(filename, "rb"))
        self.filtered_neighbours = self.load_filter_pickle_data
        self.res_seq_frequency_dict = self.load_filter_pickle_data['resSeqFreqDict']
        self.frequency_dict = self.load_filter_pickle_data['freqDict']
        self.freq_stuff = self.load_filter_pickle_data['freqStuff']
        self.neighbours_data = self.load_filter_pickle_data['neighboursData']
        print('Load filtered data complete --- >>>> \n')

    def print_top_ligand_neighbours_freq(self,
                                         top_select=40,
                                         receptor_name='LasR',
                                         ligand_name='HSL',
                                         title='Contacts_FREQ',
                                         lang='en',
                                         num_of_threads=7,
                                         parallel=True,
                                         cutoff_len=0.4,
                                         custom_dpi=600):

        top_data = self.freq_stuff[-top_select:-1]

        for top in top_data:
            data = self.frequency_dict[top]
            name_vip = data['nameVIP']

            start_frame = self.full_traj[min(data['frames'])]
            end_frame = self.full_traj[max(data['frames'])]

            start_frame_time = start_frame.time[0]
            end_frame_time = end_frame.time[0]

            length = len(self.full_traj)

            percentage = (top * 100) / length

            print(
                'Residue {0} interacts with Ligand for {1:.2f}% from {2} to {3} ps (Number of frames: {4}/{5})'.format(
                    name_vip, percentage,
                    start_frame_time,
                    end_frame_time,
                    top, length))
            print('---------------------------------------------------')

        print('--->>>>>>>----------------\n')
        print('--->>>>>>>----------------\n')

    @hlp.timeit
    def find_clusters_hbonds(self,
                             receptor_name='LasR',
                             ligand_name='HSL',
                             title='Simulation',
                             xlabel=r"Time $t$ (ns)",
                             ylabel=r"Number of Hydrogen Bonds",
                             lang='en',
                             hbond_length=0.4,
                             custom_dpi=600):

        print('Find Clusters centroids is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # sns.set(font_scale=2)

        for k in self.clusterized_data:
            print('Finding H-Bonds for cluster {0}'.format(k))
            clust_temp_data = self.clusterized_data[k]

            # self.sim_time = self.full_traj.time / 1000

            # paral = Pool(processes=16)
            # data_count = list(map(self.hbond_frame_calc, self.full_traj))
            #
            # print('data count ',data_count)

            # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
            # print('count of hbonds is ', len(hbonds))

            sns.set(style="ticks", context='paper')

            sns.set(font_scale=1)

            plt.clf()

            fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

            title = 'Frequency of H-Bonds between {0}-{1} in Cluster {2}'.format(receptor_name, ligand_name, k)

            # self.hbond_count.append(len(hbonds))
            hbonds_frames = md.wernet_nilsson(clust_temp_data, exclude_water=True, periodic=False)

            # hbonds_frames_baker = md.baker_hubbard(clust_temp_data, exclude_water=True, periodic=False)
            # label = lambda hbond : '%s -- %s' % (clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[2]))
            # for hbond in hbonds_frames_baker:
            #     print(label(hbond))




            sim_hbond_atoms = []
            sim_hbond_count = []

            sim_hbond_sel = []

            for hbonds in hbonds_frames:
                hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(clust_temp_data, hbonds, ligand_name)

                sim_hbond_atoms.append(hbond_atoms)
                sim_hbond_count.append(hbond_count)

                if len(hbond_indexes_sel) > 0:
                    sim_hbond_sel += hbond_indexes_sel

            sim_hbound_np = np.array(sim_hbond_count)

            # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

            sim_hbound_sel_np = np.array(sim_hbond_sel)
            # sim_hbound_sel_np = np.array(updated_indexes)

            if len(sim_hbound_np) == 0:
                print('No Hbonds found ')
                continue

            if len(sim_hbound_sel_np) == 0:
                print('No Hbonds found ')
                continue

            # TODO this part is very important, for removing duplicates
            unique_bonds_list = sim_hbound_sel_np[:, [0, 2]]
            unique_bonds_analysis = filter_items.filter_similar_lists(unique_bonds_list)

            unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)

            # self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
            # self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

            # curr_color = self.colors_[i - 1]
            # curr_label = 'Simulation {0}'.format(i)
            print('Curr Cluster Trajectory ', clust_temp_data)
            curr_label = "Simulation of Cluster {0}  mean: {1}±{2}".format(k, round(np.mean(sim_hbound_np), 3),
                                                                           round(np.std(sim_hbond_count), 3))
            print('Curr label ', curr_label)
            print('-=-=-' * 5)
            # This won't work here

            # da_distances = md.compute_distances(clust_temp_data, unique_bonds[:, [0, 2]], periodic=False)
            da_distances = md.compute_distances(clust_temp_data, unique_bonds_analysis[:, [0, 1]], periodic=False)

            # Version 1
            # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
            #          linewidth=0.2, label=curr_label)
            # color = itertools.cycle(['r', 'b', 'gold'])

            # colors = sns.cubehelix_palette(n_colors=len(unique_bonds), rot=-.4)
            colors = sns.cubehelix_palette(n_colors=len(unique_bonds_analysis))
            # colors = sns.cubehelix_palette(n_colors=len(unique_bonds_analysis), rot=-.4)
            # self.colors_ = colors
            # label = lambda hbond: '%s -- %s' % (
            #     clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[2]))

            label = lambda hbond: '%s -- %s' % (
                clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[1]))

            # color = itertools.cycle(['r', 'b', 'gold'])

            self.hbond_cluster_analysis_data.update({k: {'hbonds': hbonds_frames,
                                                         'distances': da_distances,
                                                         'unique': unique_bonds,
                                                         'colors': colors}})

            columns = ['Receptor-Atom', 'Ligand-Atom', 'NumHBonds', 'TotalFrames', 'Percent', 'Label']

            df = pd.DataFrame(columns=columns)

            for i in range(len(unique_bonds_analysis)):
                data = da_distances[:, i]
                data_hbonds = data[data <= hbond_length]

                curr_label_hbond = label(unique_bonds_analysis[i])

                curr_label_split = curr_label_hbond.split('--')

                percent = (len(data_hbonds) / len(clust_temp_data)) * 100

                temp_data = [curr_label_split[0], curr_label_split[1], len(data_hbonds), len(clust_temp_data),
                             round(percent, 3), curr_label_hbond]

                df.loc[i] = temp_data

                # plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds[i]), alpha=0.5)
                plt.hist(data_hbonds, color=colors[i], label=curr_label_hbond, alpha=0.5)


            csv_filename = self.simulation_name + '_' + title + '_' + 'cluster:{0}'.format(
                k) + '_' + ligand_name + '_hbonds_freq' + lang + '.csv'

            final_hbonds_dataframe = df.sort_values(['Percent'], ascending=False)
            final_hbonds_dataframe.to_csv(csv_filename)


            # TODO move legend outside
            # plt.legend(loc='best')
            art = []
            # 0.5, -0.1
            # -1.2, 0.0 Nope
            # 2.0, 0.0 a lot better

            # 1.6, 1.0 close to best
            # 1.5, 1.07 very close
            lgd = plt.legend(loc=9, bbox_to_anchor=(1.5, 1.050), ncol=1)
            art.append(lgd)

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

            fig.savefig(self.simulation_name + '_' + title + '_' + 'cluster:{0}'.format(
                k) + '_' + ligand_name + '_' + lang + '.png',
                        dpi=custom_dpi, bbox_inches='tight', additional_artists=art)
            # self.clusters_centroids.append(centroid)
            # centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(k))

        print('-----------------------------------\n')

    @hlp.timeit
    def find_clusters_hbonds_mmpbsa_baker(self,
                                          receptor_name='LasR',
                                          ligand_name='HSL',
                                          title='Simulation',
                                          xlabel=r"Time $t$ (ns)",
                                          ylabel=r"Number of Hydrogen Bonds",
                                          lang='en',
                                          hbond_length=0.4,
                                          custom_dpi=600):

        print('Find Clusters centroids is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # sns.set(font_scale=2)

        for k in self.clusterized_data:
            print('Finding H-Bonds from mmpbsa stable 10ns trajectory for cluster {0}'.format(k))

            # clust_temp_data = self.clusterized_data[k]
            clust_temp_data = self.clusterized_data_mmpbsa[k]['stableLong']

            # self.sim_time = self.full_traj.time / 1000

            # paral = Pool(processes=16)
            # data_count = list(map(self.hbond_frame_calc, self.full_traj))
            #
            # print('data count ',data_count)

            # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
            # print('count of hbonds is ', len(hbonds))

            sns.set(style="ticks", context='paper')

            sns.set(font_scale=1.0)

            plt.clf()

            fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

            title = 'Frequency of H-Bonds between {0}-{1} in Cluster {2}'.format(receptor_name, ligand_name, k)

            # self.hbond_count.append(len(hbonds))

            # Baker hubbard more classical
            hbonds_frames = md.baker_hubbard(clust_temp_data, exclude_water=True, periodic=False)

            # Wernet Nilson quite strange
            # hbonds_frames = md.wernet_nilsson(clust_temp_data, exclude_water=True, periodic=False)

            sim_hbond_atoms = []
            sim_hbond_count = []

            sim_hbond_sel = []

            hbond_atoms = []
            hbond_count = 0
            hbond_indexes_sel = []

            for hbond in hbonds_frames:
                data = self.count_lig_hbond_baker(clust_temp_data, hbond, ligand_name)

                if data != None:

                    hbond_atoms_curr = data[0]
                    hbond_count_curr = data[1]
                    hbond_indexes_sel_cur = data[2]

                    hbond_atoms.append(hbond_atoms_curr)
                    hbond_count += 1
                    hbond_indexes_sel.append(hbond_indexes_sel_cur)

                    sim_hbond_atoms.append(hbond_atoms)
                    sim_hbond_count.append(hbond_count)

                    if len(hbond_indexes_sel) > 0:
                        sim_hbond_sel += hbond_indexes_sel

            sim_hbound_np = np.array(sim_hbond_count)

            # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

            sim_hbound_sel_np = np.array(sim_hbond_sel)
            # sim_hbound_sel_np = np.array(updated_indexes)

            hbond_indexes_sel_np = np.array(hbond_indexes_sel)

            if len(sim_hbound_np) == 0:
                print('No Hbonds found ')
                continue

            if len(sim_hbound_sel_np) == 0:
                print('No Hbonds found ')
                continue

            # TODO this part is very important, for removing duplicates
            unique_bonds_list = sim_hbound_sel_np[:, [0, 2]]
            unique_bonds_analysis = filter_items.filter_similar_lists(unique_bonds_list)

            unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)

            # self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
            # self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

            # curr_color = self.colors_[i - 1]
            # curr_label = 'Simulation {0}'.format(i)
            print('Curr Cluster Trajectory ', clust_temp_data)
            curr_label = "Simulation of Cluster {0}  mean: {1}±{2}".format(k, round(np.mean(sim_hbound_np), 3),
                                                                           round(np.std(sim_hbond_count), 3))
            print('Curr label ', curr_label)
            print('-=-=-' * 5)
            # This won't work here

            # hbond_indexes_sel[0][0]
            # da_distances = md.compute_distances(clust_temp_data, unique_bonds[:, [0, 2]], periodic=False)
            da_distances = da_distances = md.compute_distances(clust_temp_data, hbond_indexes_sel_np[:, [0, 1]],
                                                               periodic=False)

            # Version 1
            # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
            #          linewidth=0.2, label=curr_label)
            # color = itertools.cycle(['r', 'b', 'gold'])

            # colors = sns.cubehelix_palette(n_colors=len(unique_bonds), rot=-.4)
            colors = sns.cubehelix_palette(n_colors=len(unique_bonds_analysis), rot=-.4)
            # self.colors_ = colors
            # label = lambda hbond: '%s -- %s' % (
            #     clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[2]))

            label = lambda hbond: '%s -- %s' % (
                clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[1]))

            # color = itertools.cycle(['r', 'b', 'gold'])

            self.hbond_cluster_analysis_data.update({k: {'hbonds': hbonds_frames,
                                                         'distances': da_distances,
                                                         'unique': unique_bonds,
                                                         'colors': colors}})

            for i in range(len(unique_bonds_analysis)):
                data = da_distances[:, i]
                data_hbonds = data[data <= hbond_length]

                # plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds[i]), alpha=0.5)
                plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds_analysis[i]), alpha=0.5)
            art = []
            # 0.5, -0.1
            # -1.2, 0.0 Nope
            # 2.0, 0.0 a lot better

            # 1.6, 1.0 close to best
            # 1.5, 1.07 very close
            lgd = plt.legend(loc=9, bbox_to_anchor=(1.5, 1.050), ncol=1)
            art.append(lgd)

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

            fig.tight_layout()
            fig.savefig(self.simulation_name + '_mmpbsa_' + title + '_' + 'cluster:{0}'.format(
                k) + '_' + ligand_name + '_' + lang + '.png',
                        dpi=custom_dpi, bbox_inches='tight', additional_artists=art)
            # self.clusters_centroids.append(centroid)
            # centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(k))

        print('-----------------------------------\n')

    @hlp.timeit
    def find_clusters_hbonds_mmpbsa(self,
                                    receptor_name='LasR',
                                    ligand_name='HSL',
                                    title='Simulation',
                                    xlabel=r"Time $t$ (ns)",
                                    ylabel=r"Number of Hydrogen Bonds",
                                    lang='en',
                                    hbond_length=0.4,
                                    custom_dpi=600):

        print('Find Clusters centroids is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_hbond = True

        self.clusters_centroids = []

        # sns.set(font_scale=2)

        for k in self.clusterized_data:
            print('Finding H-Bonds from mmpbsa stable 10ns trajectory for cluster {0}'.format(k))

            # clust_temp_data = self.clusterized_data[k]
            clust_temp_data = self.clusterized_data_mmpbsa[k]['stableLong']

            # self.sim_time = self.full_traj.time / 1000

            # paral = Pool(processes=16)
            # data_count = list(map(self.hbond_frame_calc, self.full_traj))
            #
            # print('data count ',data_count)

            # hbonds = md.baker_hubbard(self.full_traj, exclude_water=True, periodic=False)
            # print('count of hbonds is ', len(hbonds))

            sns.set(style="ticks", context='paper')

            sns.set(font_scale=1.0)

            plt.clf()

            fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

            title = 'Frequency of H-Bonds between {0}-{1} in Cluster {2}'.format(receptor_name, ligand_name, k)

            # self.hbond_count.append(len(hbonds))

            # Baker hubbard more classical
            # hbonds_frames = md.baker_hubbard(clust_temp_data, exclude_water=True, periodic=False)

            # Wernet Nilson quite strange
            hbonds_frames = md.wernet_nilsson(clust_temp_data, exclude_water=True, periodic=False)

            sim_hbond_atoms = []
            sim_hbond_count = []

            sim_hbond_sel = []

            for hbonds in hbonds_frames:
                hbond_atoms, hbond_count, hbond_indexes_sel = self.count_lig_hbond(clust_temp_data, hbonds, ligand_name)

                sim_hbond_atoms.append(hbond_atoms)
                sim_hbond_count.append(hbond_count)

                if len(hbond_indexes_sel) > 0:
                    sim_hbond_sel += hbond_indexes_sel

            sim_hbound_np = np.array(sim_hbond_count)

            # updated_indexes = filter_items.filter_similar_lists(sim_hbond_sel)

            sim_hbound_sel_np = np.array(sim_hbond_sel)
            # sim_hbound_sel_np = np.array(updated_indexes)

            if len(sim_hbound_np) == 0:
                print('No Hbonds found ')
                continue

            if len(sim_hbound_sel_np) == 0:
                print('No Hbonds found ')
                continue

            # TODO this part is very important, for removing duplicates
            unique_bonds_list = sim_hbound_sel_np[:, [0, 2]]
            unique_bonds_analysis = filter_items.filter_similar_lists(unique_bonds_list)

            unique_bonds = filter_items.filter_similar_lists(sim_hbound_sel_np)

            # self.simulation_data[str(i)].update({'hbond_atoms':sim_hbond_atoms})
            # self.simulation_data[str(i)].update({'hbond_count':sim_hbond_count})

            # curr_color = self.colors_[i - 1]
            # curr_label = 'Simulation {0}'.format(i)
            print('Curr Cluster Trajectory ', clust_temp_data)
            curr_label = "Simulation of Cluster {0}  mean: {1}±{2}".format(k, round(np.mean(sim_hbound_np), 3),
                                                                           round(np.std(sim_hbond_count), 3))
            print('Curr label ', curr_label)
            print('-=-=-' * 5)
            # This won't work here

            # da_distances = md.compute_distances(clust_temp_data, unique_bonds[:, [0, 2]], periodic=False)
            da_distances = md.compute_distances(clust_temp_data, unique_bonds_analysis[:, [0, 1]], periodic=False)

            # Version 1
            # plt.plot(self.sim_time, sim_hbond_count, color=curr_color, marker = 'x',
            #          linewidth=0.2, label=curr_label)
            # color = itertools.cycle(['r', 'b', 'gold'])

            # colors = sns.cubehelix_palette(n_colors=len(unique_bonds), rot=-.4)
            colors = sns.cubehelix_palette(n_colors=len(unique_bonds_analysis), rot=-.4)
            # self.colors_ = colors
            # label = lambda hbond: '%s -- %s' % (
            #     clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[2]))

            label = lambda hbond: '%s -- %s' % (
                clust_temp_data.topology.atom(hbond[0]), clust_temp_data.topology.atom(hbond[1]))

            # color = itertools.cycle(['r', 'b', 'gold'])

            self.hbond_cluster_analysis_data.update({k: {'hbonds': hbonds_frames,
                                                         'distances': da_distances,
                                                         'unique': unique_bonds,
                                                         'colors': colors}})

            for i in range(len(unique_bonds_analysis)):
                data = da_distances[:, i]
                data_hbonds = data[data <= hbond_length]

                # plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds[i]), alpha=0.5)
                plt.hist(data_hbonds, color=colors[i], label=label(unique_bonds_analysis[i]), alpha=0.5)
            art = []
            # 0.5, -0.1
            # -1.2, 0.0 Nope
            # 2.0, 0.0 a lot better

            # 1.6, 1.0 close to best
            # 1.5, 1.07 very close
            lgd = plt.legend(loc=9, bbox_to_anchor=(1.5, 1.050), ncol=1)
            art.append(lgd)

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

            fig.tight_layout()
            fig.savefig(self.simulation_name + '_mmpbsa_' + title + '_' + 'cluster:{0}'.format(
                k) + '_' + ligand_name + '_' + lang + '.png',
                        dpi=custom_dpi, bbox_inches='tight', additional_artists=art)
            # self.clusters_centroids.append(centroid)
            # centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(k))

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
        plt.ylabel('Freq')
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

        import gc
        gc.collect()
        len(gc.get_objects())

        print('RMSD analysis has been called on selection {0}\n'.format(selection))
        print('-----------------------------\n')

    @hlp.timeit
    def compute_com_analysis(self, selection):
        '''

        to filter H atoms example "resname DQC and (not type H)"

        :param selection: has to be mdtraj compatible
        :return:
        '''

        self.called_com_analysis = True

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
        self.com_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        self.sim_com = md.compute_center_of_mass(self.com_traj)

        self.sim_com_rmsd = calculate_rmsd.rmsd_custom(self.sim_com, frame=0)

        self.sim_com_distance = calculate_rmsd.vector_distance(self.sim_com, frame=0)

        test = 1

        self.sim_time = self.com_traj.time / 1000

        self.center_of_mass_analysis_data.update({selection: {'com': self.sim_com,
                                                              'rmsd': self.sim_com_rmsd,
                                                              'time': self.sim_time,
                                                              'distance': self.sim_com_distance}})

        self.regression_fit_range = 10

        print('Center of Mass analysis has been called on selection {0}\n'.format(selection))
        print('-----------------------------\n')

    @hlp.timeit
    def plot_com_rmsd(self, selection,
                      title='LasR Monomer Simulation',
                      xlabel=r"Time $t$ (ns)",
                      ylabel=r"RMSD(A)",
                      custom_dpi=300):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_rmsd = self.center_of_mass_analysis_data[selection]['rmsd']

        sim_time = self.center_of_mass_analysis_data[selection]['time']

        print('RMSD mean is ', np.mean(traj_rmsd))

        plt.plot(sim_time, traj_rmsd, color='b',
                 linewidth=0.6, label='LasR')

        data_frame = converters.convert_data_to_pandas(sim_time, traj_rmsd)

        y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()

        plt.plot(data_frame['x'], y_average_mean, color='r',
                 linewidth=0.9, label='LasR rolling mean')

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # remove part of ticks
        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '_com_RMSD_' + selection + '.png', dpi=custom_dpi)

        plt.cla()
        plt.close(fig)

        print('RMSD COM plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_com_distance(self, selection,
                          title='COM distance',
                          xlabel=r"Time $t$ (ns)",
                          ylabel=r"Distance(nm)",
                          custom_dpi=300,
                          rolling_window=150):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_distance = self.center_of_mass_analysis_data[selection]['distance']

        sim_time = self.center_of_mass_analysis_data[selection]['time']

        print('Distance  mean is ', np.mean(traj_distance))

        plt.plot(sim_time, traj_distance, color='b',
                 linewidth=0.6, label='LasR')

        data_frame = converters.convert_data_to_pandas(sim_time, traj_distance)

        y_average_mean = data_frame['y'].rolling(center=False, window=rolling_window).mean()

        plt.plot(data_frame['x'], y_average_mean, color='r',
                 linewidth=0.9, label='rolling mean')

        y_average_median = data_frame['y'].rolling(center=False, window=rolling_window).median()

        plt.plot(data_frame['x'], y_average_median, color='g',
                 linewidth=0.9, label='rolling mean')

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # plt.title(title)

        # remove part of ticks
        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '_com_distance_' + selection + '.png', dpi=custom_dpi)

        plt.cla()
        plt.close(fig)

        print('RMSD COM plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_3D_com_pos(self, selection,
                        title='LasR Monomer Simulation',
                        xlabel=r"Time $t$ (ns)",
                        ylabel=r"RMSD(A)",
                        custom_dpi=300):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np

        plt.clf()
        fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        com_pos = self.center_of_mass_analysis_data[selection]['com']

        traj_rmsd = self.center_of_mass_analysis_data[selection]['rmsd']

        sim_time = self.center_of_mass_analysis_data[selection]['time']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = com_pos[:, 0]
        ys = com_pos[:, 1]
        zs = com_pos[:, 2]
        ax.scatter(xs, ys, zs, marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

        print('3D  COM plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsd(self, selection,
                  title='LasR Monomer Simulation',
                  xlabel=r"Time $t$ (ns)",
                  ylabel=r"RMSD(nm)",
                  custom_dpi=600):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(7, 7))
        fig = plt.figure(figsize=(plot_tools.cm2inch(8.4, 8.4)))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_rmsd = self.rmsd_analysis_data[selection]

        print('RMSD mean is ', np.mean(traj_rmsd))

        plt.plot(self.sim_time, traj_rmsd, color='b',
                 linewidth=0.6, label='LasR')

        data_frame = converters.convert_data_to_pandas(self.sim_time, traj_rmsd)

        y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()

        plt.plot(data_frame['x'], y_average_mean, color='r',
                 linewidth=0.9, label='LasR rolling mean')

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # No title
        # plt.title(title)

        # remove part of ticks
        sns.despine()
        fig.tight_layout()
        fig.savefig(self.simulation_name + '_' + title + '_RMSD_' + selection + '.png', dpi=custom_dpi)

        # cla beforce closing or not 2018 aug 25
        plt.cla()
        plt.close(fig)

        print('RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsd_histogram(self, selection,
                            bins=40,
                            kde=True,
                            rug=True,
                            title='Simulation',
                            xlabel=r"Time $t$ (ns)",
                            ylabel=r"RMSD(nm)",
                            custom_dpi=600):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        fig, ax = plt.subplots()
        # the size of A4 paper
        # fig.set_size_inches(7, 7)
        fig.set_size_inches(plot_tools.cm2inch(8.4, 8.4))

        traj_rmsd = self.rmsd_analysis_data[selection]

        print('RMSD mean is ', np.mean(traj_rmsd))

        traj_rmsd_pandas = pd.Series(traj_rmsd, name="RMSD(nm)")

        ax = sns.distplot(traj_rmsd_pandas, bins=bins, kde=kde, rug=rug, ax=ax)

        ax.set(xlabel='RMSD(nm)', ylabel='Density')

        # # plt.plot(self.sim_time, traj_rmsd, color='b',
        # #          linewidth=0.6, label='LasR')
        #
        # data_frame = converters.convert_data_to_pandas(self.sim_time, traj_rmsd)
        #
        # y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()
        #
        # plt.plot(data_frame['x'], y_average_mean, color='r',
        #          linewidth=0.9, label='LasR rolling mean')
        #
        # # plt.legend(loc="best", prop={'size': 8})
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # # No title
        # # plt.title(title)

        # remove part of ticks
        sns.despine()
        fig.tight_layout()
        fig.savefig(self.simulation_name + '_' + title + '_RMSD_histogram_' + selection + '.png', dpi=custom_dpi,
                    bbox_inches='tight')

        plt.cla()
        plt.close(fig)

        print('RMSD Histogram plot created')
        print('-----------------------------------\n')

    # TODO need to think about this
    @hlp.timeit
    def plot_rmsd_histogram_with_peaks(self, selection,
                                       bins=40,
                                       kde=True,
                                       rug=True,
                                       title='Simulation',
                                       xlabel=r"Time $t$ (ns)",
                                       ylabel=r"RMSD(nm)",
                                       custom_dpi=600):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=2)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        fig, ax = plt.subplots()
        # the size of A4 paper
        # fig.set_size_inches(7, 7)
        fig.set_size_inches(plot_tools.cm2inch(8.4, 8.4))

        traj_rmsd = self.rmsd_analysis_data[selection]

        print('RMSD mean is ', np.mean(traj_rmsd))

        traj_rmsd_pandas = pd.Series(traj_rmsd, name="RMSD(nm)")

        ax = sns.distplot(traj_rmsd_pandas, bins=bins, kde=kde, rug=rug, ax=ax)

        ax.set(xlabel='RMSD(nm)', ylabel='Density')

        # # plt.plot(self.sim_time, traj_rmsd, color='b',
        # #          linewidth=0.6, label='LasR')
        #
        # data_frame = converters.convert_data_to_pandas(self.sim_time, traj_rmsd)
        #
        # y_average_mean = data_frame['y'].rolling(center=False, window=20).mean()
        #
        # plt.plot(data_frame['x'], y_average_mean, color='r',
        #          linewidth=0.9, label='LasR rolling mean')
        #
        # # plt.legend(loc="best", prop={'size': 8})
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # # No title
        # # plt.title(title)

        # remove part of ticks
        sns.despine()

        fig.savefig(self.simulation_name + '_' + title + '_RMSD_histogram_' + selection + '.png', dpi=custom_dpi)

        print('RMSD Histogram plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def extract_normal_modes(self,
                             mode=0,
                             rmsd=0.06,
                             n_steps=600,
                             selection='backbone'):
        print('Extraction Normal Mode')

        '''
        Buggy as hell
        
        '''
        from nma import ANMA

        anma = ANMA(mode=mode, rmsd=rmsd, n_steps=n_steps, selection=selection)

        # Transform the PDB into a short trajectory of a given mode
        anma_traj = anma.fit_transform(self.full_traj)

        anma_traj.save('Test.pdb')

        print('Normal Mode has been extracted')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_pca_color(self, selection,
                       title='LasR RMSD',
                       xlabel=r"Time $t$ (ns)",
                       ylabel=r"RMSD(nm)",
                       custom_dpi=600,
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

        # TODO PCA cluster part

        sns.set(font_scale=2)

        fig, ax1 = plt.subplots(1, 1)  #

        n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        fig.set_size_inches(20, 12)

        cluster_labels = self.clusters_info[n_clusters]['labels']
        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        X = self.reduced_cartesian

        # TODO a new try
        colors = self.colors_

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        ax1.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
                    c=self.pca_traj.time)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # # Draw white circles at cluster centers
        # ax1.scatter(centers[:, 0], centers[:, 1],
        #             marker='o', c="white", alpha=1, s=800)
        #
        # for i, c in enumerate(centers):
        #     clust_num = i + 1
        #     ax1.scatter(c[0], c[1], marker='$%d$' % clust_num, alpha=1, s=800)

        cbar = plt.colorbar()
        cbar.set_label('Time [ps]')

        ax1.set_title("The visualization of the clustered data")
        # ax1.set_xlabel("Feature space for the 1st feature")

        # ax1.set_ylabel("Feature space for the 2nd feature")

        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")

        # # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        # #               "with n_clusters = %d" % n_clusters),
        # #              fontsize=14, fontweight='bold')
        #
        #
        #
        #
        #
        # # TODO RMSD part
        #
        # if lang == 'rus':
        #     title = 'Симуляция'
        #     xlabel = r"Время $t$ (нс)"
        #     ylabel = r"RMSD(нм)"
        # else:
        #     title = 'Simulation'
        #     xlabel = r"Time $t$ (ns)"
        #     ylabel = r"RMSD(nm)"
        #
        # ax2.plot(self.sim_time, self.sim_rmsd, zorder=1)
        # traj_rmsd = self.rmsd_analysis_data[selection]
        #
        # ax2.scatter(self.sim_time, traj_rmsd, marker='o', s=30, facecolor='0.5', lw=0,
        #             c=self.cluster_colors, zorder=2)
        #
        # # plt.legend(loc="best", prop={'size': 8})
        # ax2.set_xlabel(xlabel)
        # ax2.set_xlim(self.sim_time[0], self.sim_time[-1])
        #
        # ax2.set_ylabel(ylabel)  # fix Angstrom need to change to nm
        # ax2.set_title(title)

        fig.tight_layout()

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(
            self.simulation_name + '_' + title + '_' + selection + '_pca_color_new' + '_' + lang + '.png',
            dpi=custom_dpi, bbox_inches='tight')

        print('PCA color new plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_md_pca_cluster_analysis(self, selection='protein',
                                     color_selection=None,
                                     title=None, custom_dpi=600, show=False,
                                     transparent_alpha=False,
                                     lang='eng'):
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)

        # cmap = sns.cubehelix_palette(n_colors=len(self.pca_traj.time), as_cmap=True, reverse=True)
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ((ax1, ax2)) = plt.subplots(1, 2)  #

        # n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        # fig.set_size_inches(20, 12)
        fig.set_size_inches(plot_tools.cm2inch(17.7, 10))

        time_ns = self.pca_traj.time / 1000

        data = self.md_pca_analysis_data[selection]
        X = self.md_pca_analysis_data[selection]

        scatter = ax1.scatter(data[:, 0], data[:, 1], marker='o', s=60, c=time_ns)
        # ax1.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=cmap)
        # ax1.set_xlabel('Feature space for the 1st feature')
        # ax1.set_ylabel('Feature space for the 2nd feature')

        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        if title is None:
            title_to_use = 'Conformation PCA Analysis: {0}'.format(selection)
        else:
            title_to_use = title

        ax1.set_title(title_to_use)

        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

        # im = ax1.imshow(time_ns, cmap=time_ns)
        # fig.colorbar(im, cax=cax, orientation='vertical')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.2)

        fig.colorbar(scatter, cax=cax, orientation='vertical')

        # cbar = plt.colorbar()
        # cbar.set_label('Time $t$ (ns)')

        if color_selection is not None:
            cluster_colors = self.cluster_selection_color_data[color_selection]['colorData']
        else:
            cluster_colors = self.cluster_selection_color_data[selection]['colorData']

        if self.cluster_selection_analysis_data[selection]['overrideClustNum'] is None:
            n_clusters = self.cluster_selection_analysis_data[selection]['clustNum']
        else:
            n_clusters = self.cluster_selection_analysis_data[selection]['overrideClustNum']

        clustus_info = self.cluster_selection_analysis_data[selection]['clusterInfo']
        cluster_analysis_info = self.cluster_selection_analysis_data[selection]['clusterAnalysisInfo']

        cluster_labels = clustus_info[n_clusters]['labels']
        sample_silhouette_values = clustus_info[n_clusters]['silhouette_values']
        silhouette_avg = clustus_info[n_clusters]['silhouette']

        centers = clustus_info[n_clusters]['centers']

        # TODO a new try
        # colors = self.colors_

        # 2nd Plot showing the actual clusters formed
        # colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        # self.cluster_colors = colors

        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                    c=cluster_colors)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=130)

        for i, c in enumerate(centers):
            clust_num = i + 1
            ax2.scatter(c[0], c[1], marker='$%d$' % clust_num, alpha=1, s=125)

        ax2.set_title("The visualization of the clustered data")
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")

        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # fig.tight_layout()
        #
        # # remove part of ticks
        # sns.despine()
        # # plt.show()
        #
        # fig.savefig(
        #     self.simulation_name + '_' + title + '_' + selection + '_pca+rmsd_cluster_color' + '_' + lang + '.png',
        #     dpi=custom_dpi, bbox_inches='tight')

        print('PCA+RMSD plot created')
        print('-----------------------------------\n')

        fig.savefig(self.simulation_name + '_' + selection + '_PCA&cluster_analysis' + '.png', dpi=custom_dpi,
                    bbox_inches='tight',
                    transparent=transparent_alpha)

        if show is True:
            plt.show()

        plt.cla()
        plt.close(fig)

        import gc
        gc.collect()
        len(gc.get_objects())

        print("PCA + cluster plot: {0}  -> created".format(selection))
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_pca_rmsd_cluster_color(self, selection,
                                    color_selection=None,
                                    title='LasR RMSD',
                                    xlabel=r"Time $t$ (ns)",
                                    ylabel=r"RMSD(nm)",
                                    custom_dpi=600,
                                    lang='rus'):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(14, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(17.7, 8.4))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # plt.plot(self.sim_time, self.sim_rmsd, color=self.cluster_colors,
        #          linewidth=0.6, label='LasR')

        # TODO PCA cluster part

        sns.set(font_scale=1)

        fig, ((ax1, ax2)) = plt.subplots(1, 2)  #

        # n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        # fig.set_size_inches(20, 12)
        fig.set_size_inches(plot_tools.cm2inch(17.7, 11))

        if color_selection is not None:
            cluster_colors = self.cluster_selection_color_data[color_selection]['colorData']
        else:
            cluster_colors = self.cluster_selection_color_data[selection]['colorData']

        traj_rmsd = self.rmsd_analysis_data[selection]

        if self.cluster_selection_analysis_data[selection]['overrideClustNum'] is None:
            n_clusters = self.cluster_selection_analysis_data[selection]['clustNum']
        else:
            n_clusters = self.cluster_selection_analysis_data[selection]['overrideClustNum']

        clustus_info = self.cluster_selection_analysis_data[selection]['clusterInfo']
        cluster_analysis_info = self.cluster_selection_analysis_data[selection]['clusterAnalysisInfo']

        cluster_labels = clustus_info[n_clusters]['labels']
        sample_silhouette_values = clustus_info[n_clusters]['silhouette_values']
        silhouette_avg = clustus_info[n_clusters]['silhouette']

        centers = clustus_info[n_clusters]['centers']

        X = self.md_pca_analysis_data[selection]

        # TODO a new try
        # colors = self.colors_

        # 2nd Plot showing the actual clusters formed
        # colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        # self.cluster_colors = colors

        ax1.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                    c=cluster_colors)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # Draw white circles at cluster centers
        ax1.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=130)

        for i, c in enumerate(centers):
            clust_num = i + 1
            ax1.scatter(c[0], c[1], marker='$%d$' % clust_num, alpha=1, s=125)

        ax1.set_title("The visualization of the clustered data")
        # ax1.set_xlabel("Feature space for the 1st feature")
        # ax1.set_ylabel("Feature space for the 2nd feature")

        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")

        # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

        # TODO RMSD part

        if lang == 'rus':
            title = 'Симуляция'
            xlabel = r"Время $t$ (нс)"
            ylabel = r"RMSD(нм)"
        else:
            title = 'Simulation'
            xlabel = r"Time $t$ (ns)"
            ylabel = r"RMSD(nm)"

        ax2.plot(self.sim_time, traj_rmsd, zorder=1)
        # traj_rmsd = self.rmsd_analysis_data[selection]

        ax2.scatter(self.sim_time, traj_rmsd, marker='o', s=15, facecolor='0.5', lw=0,
                    c=cluster_colors, zorder=2)

        # plt.legend(loc="best", prop={'size': 8})
        ax2.set_xlabel(xlabel)
        ax2.set_xlim(self.sim_time[0], self.sim_time[-1])

        ax2.set_ylabel(ylabel)  # fix Angstrom need to change to nm
        ax2.set_title(title)

        fig.tight_layout()

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(
            self.simulation_name + '_' + title + '_' + selection + '_pca+rmsd_cluster_color' + '_' + lang + '.png',
            dpi=custom_dpi, bbox_inches='tight')

        plt.cla()
        plt.close(fig)

        print('PCA+RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsd_cluster_color(self, selection,
                                color_selection=None,
                                title='LasR Monomer Simulation',
                                xlabel=r"Time $t$ (ns)",
                                ylabel=r"RMSD(nm)",
                                custom_dpi=600,
                                lang='rus'):
        import pylab as plt
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(14, 7))

        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

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

        if color_selection is not None:
            cluster_colors = self.cluster_selection_color_data[color_selection]['colorData']
        else:
            cluster_colors = self.cluster_selection_color_data[selection]['colorData']

        traj_rmsd = self.rmsd_analysis_data[selection]
        plt.plot(self.sim_time, traj_rmsd, zorder=1)
        plt.scatter(self.sim_time, traj_rmsd, marker='o', s=7, facecolor='0.5', lw=0,
                    c=cluster_colors, zorder=2)

        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.xlim(self.sim_time[0], self.sim_time[-1])

        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # plt.title(title)

        fig.tight_layout()

        # remove part of ticks
        sns.despine()
        # plt.show()

        fig.savefig(self.simulation_name + '_' + title + '_' + selection + '_cluster_color' + '_' + lang + '.png',
                    dpi=custom_dpi, bbox_inches='tight')

        plt.cla()
        plt.close(fig)

        print('RMSD plot created')
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsf(self, selection,
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
        sns.set(font_scale=1)

        traj_rmsf = self.rmsf_analysis_data[selection]['rmsf']
        atom_indices_rmsf = self.rmsf_analysis_data[selection]['atom_indices']

        conv_data = converters.convert_data_to_pandas(atom_indices_rmsf, traj_rmsf)

        # sns.tsplot(time="x", unit="y",  data=conv_data,
        #            size=4,  fit_reg=False,
        #            scatter_kws={"s": 50, "alpha": 1})
        # sns.plt.show()

        # fig = plt.figure(figsize=(14, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        plt.plot(conv_data['x'], conv_data['y'], color='b',
                 linewidth=0.6, label=title)
        plt.xlabel(xlabel)
        plt.xlim(min(conv_data['x']) - 100, max(conv_data['x']) + 100)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        plt.title(title)

        # remove part of ticks
        sns.despine()
        fig.tight_layout()

        fig.savefig(self.simulation_name + '_' + title + '_rmsf.png', dpi=custom_dpi, bbox_inches='tight')

        plt.cla()
        plt.close(fig)


        import gc
        gc.collect()
        len(gc.get_objects())
        
        print('RMSF plot created')

    @hlp.timeit
    def plot_rg(self,
                selection,
                title='LasR Monomer Simulation',
                xlabel=r"time $t$ (ns)",
                ylabel=r"C$_\alpha$ Rg from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
                custom_dpi=600):
        import pylab as plt

        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)

        # In[27]:

        # fig = plt.figure(figsize=(7, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_rg = self.rg_analysis_data[selection]
        plt.plot((self.sim_time), traj_rg, color='b',
                 linewidth=0.6, label='LasR')

        ylabel = r"Rg from $t=0$ (nm)"
        # plt.legend(loc="best", prop={'size': 8})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # plt.title(title)

        # In[28]:
        fig.tight_layout()
        fig.savefig(self.simulation_name + '_' + title + '_Rg_' + selection + '.png', dpi=custom_dpi,
                    bbox_inches='tight')

        plt.cla()
        plt.close(fig)

        print('Rg plot created for selection: {0}'.format(selection))
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_rmsd_versus_rg(self,
                            selection,
                            title='LasR Monomer Simulation',
                            xlabel=r"time $t$ (ns)",
                            ylabel=r"C$_\alpha$ Rg from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
                            custom_dpi=600):
        import pylab as plt

        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)

        # In[27]:

        # fig = plt.figure(figsize=(7, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')

        traj_rmsd = self.rmsd_analysis_data[selection]

        print('RMSD mean is ', np.mean(traj_rmsd))

        traj_rmsd_pandas = pd.Series(traj_rmsd, name="RMSD(nm)")

        traj_rg = self.rg_analysis_data[selection]

        # time_ns = self.pca_traj.time / 1000
        time_ns = self.sim_time

        # data = self.md_pca_analysis_data[selection]

        plt.scatter(traj_rmsd, traj_rg, marker='o', s=60, c=time_ns)
        # plt.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=cmap)
        # plt.xlabel('Feature space for the 1st feature')
        # plt.ylabel('Feature space for the 2nd feature')

        plt.xlabel("RMSD(nm)")
        plt.ylabel("Rg(nm)")

        cbar = plt.colorbar()
        cbar.set_label('Time $t$ (ns)')

        # plt.plot((self.sim_time), traj_rg, color='b',
        #          linewidth=0.6, label='LasR')
        #
        # ylabel = r"Rg from $t=0$ (nm)"
        # # plt.legend(loc="best", prop={'size': 8})
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)  # fix Angstrom need to change to nm
        # # plt.title(title)

        # In[28]:
        fig.tight_layout()
        fig.savefig(self.simulation_name + '_' + title + '_RMSD_versus_Rg_' + selection + '.png', dpi=custom_dpi,
                    bbox_inches='tight')

        print('RMSD versus Rg plot created for selection: {0}'.format(selection))
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

    # Featurize Data

    #@hlp.timeit
    #@numba.jit
    def sep_feat(self):
        from msmbuilder.featurizer import RawPositionsFeaturizer

        featurizer_data = RawPositionsFeaturizer()
        # print('Featurizer created:\n')

        featurized_data = featurizer_data.fit_transform(self.traj_to_featurize)

        return featurized_data


    def featurize_traj(self, selection='protein', featurizer_type=None,
                            client_address=None, custom_ending=''):
        '''

        :param selection: for which atom selection
        :param featurizer_type: dihedral or other
        :param client_address: Dask client address
        :return:
        '''

        print('Featurize md trajectory for selection:{0}\n'.format(selection))
        print('-------------------------------\n')

        selection_text = selection

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        self.traj_to_featurize = self.full_traj.atom_slice(atom_indices=self.selection)

        # This really works strange

        self.featurized_data = self.sep_feat()

        self.feat_data = pd.DataFrame(self.featurized_data)

        self.feat_data.insert(0, 'frame', self.frames)
        self.feat_data.insert(1, 'frameTime', self.frame_time)
        self.feat_data.insert(0, 'simName', self.simulation_name)
        self.feat_data.insert(0, 'MDfile',  self.md_trajectory_file)
        self.feat_data.insert(0, 'MDtopol', self.md_topology_file)
        self.feat_data.insert(0, 'SimNum', self.sim_num)




        test = 1
        # filename = 'feautirezed_data_{0}_sel_{1}_end_{2}.h5'.format(featurizer_type, selection, custom_ending)
        # self.feat_data.to_hdf(filename, key='df')
        # return self.number_pca
        print("Featurize trajectory finished successfully")
        print('-----------------------------------\n')

    @hlp.timeit
    def featurize_traj_dask(self, selection='protein', featurizer_type=None,
                            client_address=None, custom_ending=''):
        '''

        :param selection: for which atom selection
        :param featurizer_type: dihedral or other
        :param client_address: Dask client address
        :return:
        '''


        print('Featurize md trajectory for selection:{0}\n'.format(selection))
        print('-------------------------------\n')

        selection_text = selection

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)


        self.traj_to_featurize = self.full_traj.atom_slice(atom_indices=self.selection)



        test = 1

        filename_ =self.md_trajectory_file

        final_filename = filename_.split(os.sep)[-1]
        final_directory = filename_.split(final_filename)[0]

        self.frames = pd.Series(list(range(len(self.traj_to_featurize))))
        self.frame_time = pd.Series(self.traj_to_featurize.time)


        # this is for prototype purposes
        # TODO yo man
        # file_path = '/media/Work/MEGA/Programming/LasR_DQC/feautirezed_data_Dihedral_sel_protein_end_DBD_DQC.h5'
        # prototype = pd.read_hdf(file_path)

        # need to add column
        # prototype['frames'] = self.frames
        # prototype['frameTime'] = self.frame_time

        # ADD necessary data
        # prototype.insert(0, 'frame', self.frames)
        # prototype.insert(1, 'frameTime', self.frame_time)
        # prototype.insert(0, 'simName', self.simulation_name)
        # prototype.insert(0, 'MDfile',  self.md_trajectory_file)
        # prototype.insert(0, 'MDtopol', self.md_topology_file)
        # prototype.insert(0, 'SimNum', self.sim_num)

        from dask.distributed import Client, as_completed
        from molmolpy.tools import run_dask_tools




        curr_client = Client(client_address)


        # worker_status = run_dask_tools.get_dask_worker_status(curr_client)

        # get_worker_free = run_dask_tools.check_free_resources(worker_status)

        info = curr_client.scheduler_info()
        # print(info)

        ncores_step1 = curr_client.ncores

        ncore_step2 = str(ncores_step1).split('cores=')
        ncore_step3 = ncore_step2[-1].split('>>')

        ncores_final = int(ncore_step3[0])

        test = 1

        self.feat_data = featurizers.featurize_data(self.traj_to_featurize , featurizer_type,
                                                    client=curr_client,
                                                    step=1,
                                                    times_divide=4,
                                                    num_of_threads=ncores_final)


        self.feat_data.insert(0, 'frame', self.frames)
        self.feat_data.insert(1, 'frameTime', self.frame_time)
        self.feat_data.insert(0, 'simName', self.simulation_name)
        self.feat_data.insert(0, 'MDfile',  self.md_trajectory_file)
        self.feat_data.insert(0, 'MDtopol', self.md_topology_file)
        self.feat_data.insert(0, 'SimNum', self.sim_num)




        test = 1
        # filename = 'feautirezed_data_{0}_sel_{1}_end_{2}.h5'.format(featurizer_type, selection, custom_ending)
        # self.feat_data.to_hdf(filename, key='df')
        #return self.number_pca
        print("Featurize trajectory finished successfully")
        print('-----------------------------------\n')




    def save_featurized_data(self, custom_name):
        test = 1
        if custom_name is not None:
            filename = '{0}{1}.h5'.format(custom_name, self.sim_num)
        else:
            filename = 'feautirezed_data_{0}_sel_{1}_end_{2}.h5'.format(featurizer_type, selection, custom_ending)
        self.feat_data.to_hdf(filename, key='df')
        #return self.number_pca
        print("Featurize trajectory finished successfully")
        print('-----------------------------------\n')





    # TODO do PCA transformation of MD simulation

    @hlp.timeit
    def md_pca_cum_variance_analysis(self, selection='protein', show_plot=False, custom_dpi=600,
                                     percentage=87, number_of_components=20, featurizer_type = None ):

        self.called_md_pca_analysis = True
        print('PCA Cumulative Variance analysis has been called for selection:{0}\n'.format(selection))
        print('-------------------------------\n')

        selection_text = selection

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
        sns.set(style="ticks", context='paper')
        # fig = plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        sns.set(font_scale=1)

        self.topology = self.full_traj.topology

        self.selection = self.topology.select(selection)

        if number_of_components is not None:
            pca1 = PCA(n_components=number_of_components)
        else:
            pca1 = PCA(n_components=len(self.selection))

        self.pca_traj = self.full_traj.atom_slice(atom_indices=self.selection)



        if featurizer_type == None:

            self.pca_traj.superpose(self.pca_traj, 0)

            self.reduced_cartesian = pca1.fit_transform(
                self.pca_traj.xyz.reshape(self.pca_traj.n_frames, self.pca_traj.n_atoms * 3))
        else:
            self.feat_data = featurizers.featurize_data(self.pca_traj, featurizer_type)
            test = 1


        # The amount of variance that each PC explains
        var = pca1.explained_variance_ratio_
        print('Explained variance ratio: ', var)

        self.md_pre_pca_analysis_data.update({selection_text: {'varExplainedRatio': pca1.explained_variance_ratio_,
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

        fig.savefig(self.simulation_name + '_{0}_'.format(selection_text) + 'PCA_cumsum_analysis_' + '.png',
                    dpi=custom_dpi,
                    bbox_inches='tight')

        if show_plot is True:
            plt.show()

        import heapq

        max_num_list = 3
        var_array = np.array(var1)

        best_score = 0
        best_index = 0
        for i in range(len(var_array)):
            if var_array[i] >= percentage:
                best_score = var_array[i]
                best_index = i
                break

        bottom_var = heapq.nsmallest(max_num_list, range(len(var_array)), var_array.take)
        print('Bottom Var', bottom_var)

        # self.md_pca_analysis_data.update({selection_text: self.reduced_cartesian})
        # self.number_pca = bottom_var[-1] + 1
        self.number_pca = best_index + 1
        print('Percentage of PCA : ', best_score)

        if best_score == 0:
            self.number_pca += 1

        print('Number of PCA : ', self.number_pca)
        print(self.reduced_cartesian.shape)
        return self.number_pca
        print("PCA transformation finished successfully")
        print('-----------------------------------\n')

    @hlp.timeit
    def plot_pca_proportion_of_variance(self, selection, custom_dpi=600):
        # np.round(pca1.explained_variance_ratio_, decimals=4) * 100

        data = self.md_pre_pca_analysis_data[selection]

        explained_variance_ratio_ = data['varExplainedRatio']

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
        fig.savefig(self.simulation_name + '_ProportionOfVariance_' + selection + '.png', dpi=custom_dpi,
                    bbox_inches='tight')

        plt.cla()
        plt.close(fig)
        print('RMSD plot created')
        print('-----------------------------------\n')

    # @profile
    @hlp.timeit
    def md_pca_analysis(self, selection='protein', PC_number=2):

        self.called_md_pca_analysis = True
        print('PCA analysis has been called for selection:{0}\n'.format(selection))
        print('-------------------------------\n')

        pca1 = PCA(n_components=PC_number)

        selection_text = selection

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

        # TODO change here see if this affects superimpose
        self.pca_traj = self.full_traj.atom_slice(atom_indices=self.selection, inplace=False)

        self.pca_traj.superpose(self.pca_traj, 0)

        self.reduced_cartesian = pca1.fit_transform(
            self.pca_traj.xyz.reshape(self.pca_traj.n_frames, self.pca_traj.n_atoms * 3))

        # self.md_pca_analysis_data.update({selection_text:{'data': self.reduced_cartesian,
        #                                                   'n_dim':PC_number})

        self.md_pca_analysis_data.update({selection_text: self.reduced_cartesian})

        print(self.reduced_cartesian.shape)
        print("PCA transformation finished successfully")
        print('-----------------------------------\n')

    ####################################################################################################################
    # TODO Pairwise RMSD calc

    @hlp.timeit
    def md_pairwise_rmsd_analysis(self, selection='protein'):

        self.called_md_pairwise_rmsd_analysis = True
        print('Pairwise RMSD analysis has been called for selection:{0}\n'.format(selection))
        print('-------------------------------\n')

        selection_text = selection

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

        self.pairwise_rmsd_traj = self.full_traj.atom_slice(atom_indices=self.selection)

        import scipy.cluster.hierarchy
        from scipy.spatial.distance import squareform

        distances = np.empty((self.pairwise_rmsd_traj.n_frames, self.pairwise_rmsd_traj.n_frames))
        for i in range(self.pairwise_rmsd_traj.n_frames):
            distances[i] = md.rmsd(self.pairwise_rmsd_traj, self.pairwise_rmsd_traj, i)
        print('Max pairwise rmsd: %f nm' % np.max(distances))

        # assert np.all(distances - distances.T < 1e-6)
        reduced_distances = squareform(distances, checks=False)

        # pca1 = PCA(n_components=2)
        #
        #
        # reduced_distances = pca1.fit_transform(distances)
        #
        # fig = plt.figure(figsize=(10, 10))
        #
        #
        # data = reduced_distances
        #
        # plt.scatter(data[:, 0], data[:, 1], marker='o', s=60)
        # # plt.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=cmap)
        # plt.xlabel('Feature space for the 1st feature')
        # plt.ylabel('Feature space for the 2nd feature')
        # plt.show()
        #
        #
        #
        # # The amount of variance that each PC explains
        # var = pca1.explained_variance_ratio_
        # print('Explained variance ratio: ', var)
        #
        # # Cumulative Variance explains
        # var1 = np.cumsum(np.round(pca1.explained_variance_ratio_, decimals=4) * 100)
        #
        # print("Cumulative Variance explains ", var1)
        #
        # # # plt.plot(var)
        # # plt.plot(var1)
        # # plt.xlabel("Principal Component")
        # # plt.ylabel("Cumulative Proportion of Variance Explained")
        # # plt.show()
        #
        # import heapq
        #
        # max_num_list = 3
        # var_array = np.array(var1)
        # bottom_var = heapq.nsmallest(max_num_list, range(len(var_array)), var_array.take)
        # print('Bottom Var', bottom_var)
        #
        # # self.md_pca_analysis_data.update({selection_text: self.reduced_cartesian})
        # self.number_pca = bottom_var[-1] + 1
        # print('Number of PCA : ', self.number_pca)
        # print(reduced_distances.shape)

        self.md_pairwise_rmsd_analysis_data.update({selection_text: distances})

        test = 1

        #  linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
        # cutree = cluster.hierarchy.cut_tree(linkage, n_clusters=[5, 10])

        # self.pca_traj.superpose(self.pca_traj, 0)
        #
        # self.reduced_cartesian = pca1.fit_transform(
        #     self.pca_traj.xyz.reshape(self.pca_traj.n_frames, self.pca_traj.n_atoms * 3))
        #
        # self.md_pca_analysis_data.update({selection_text: self.reduced_cartesian})
        #
        # print(self.reduced_cartesian.shape)
        print("Pairwise RMSD calculation finished successfully")
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
        # clust_num = max(cluster_dict.items(), key=operator.itemgetter(1))[0]

        # TODO how to select middle yay
        whole_stuff = max(cluster_dict.items(), key=operator.itemgetter(1))
        # clust_num = max(cluster_dict.iterkeys(), key=lambda k: cluster_dict[k])
        clust_num_pre = [key for key, val in cluster_dict.items() if val == max(cluster_dict.values())]

        import numpy

        def median(lst):
            return numpy.median(numpy.array(lst))

        clust_num = sorted(clust_num_pre)[len(clust_num_pre) // 2]

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
        # THIS IS VERY SLOW
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
            centroid.save(self.simulation_name + '_' + '{0}_cluster_centroid.pdb'.format(str(k + 1)))

        print('-----------------------------------\n')

    @hlp.timeit
    def find_clusters_average_structures(self, last_frames=500):

        print('Find Clusters average structure is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_average = True

        self.clusters_average = []

        for k in self.clusterized_data:
            print('Finding centroid for cluster {0}'.format(k))
            clust_temp_data = self.clusterized_data[k]

            temp_data = clust_temp_data[-last_frames:-1]

            from mdtraj.geometry.alignment import rmsd_qcp, compute_translation_and_rotation, compute_average_structure

            average = compute_average_structure(temp_data.xyz)
            # self.centroid_conf = centroid
            # print(centroid)
            test = 1

            import copy

            average_struct = copy.deepcopy(temp_data[-1])

            orig_shape = average_struct.xyz.shape

            new_average = np.reshape(average, orig_shape)
            # average_struct.xyz = average

            average_struct._xyz = new_average

            # self.centroid_conf = centroid
            self.clusters_average.append(average_struct)
            average_struct.save(self.simulation_name + '_' + '{0}_cluster_average.pdb'.format(str(k + 1)))

        print('-----------------------------------\n')

    @hlp.timeit
    def find_clusters_centroid_mmpbsa(self, state='stableLong'):

        print('Find Clusters centroids MMPBSA is called\n')
        print('-----------------------------------\n')

        self.called_find_clusters_centroid_mmpbsa = True

        self.clusters_centroids_mmpbsa_dict = {}

        self.clusters_centroids_mmpbsa = []

        for k in self.clusterized_data_mmpbsa:
            print('Finding centroid for cluster {0}'.format(k))
            clust_temp_data = self.clusterized_data_mmpbsa[k][state]

            atom_indices = [a.index for a in clust_temp_data.topology.atoms if a.element.symbol != 'H']
            distances = np.empty((clust_temp_data.n_frames, clust_temp_data.n_frames))
            for i in range(clust_temp_data.n_frames):
                distances[i] = md.rmsd(clust_temp_data, clust_temp_data, i, atom_indices=atom_indices)

            beta = 1
            index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
            print(index)

            centroid = clust_temp_data[index]

            centroid_all_atoms = self.clusterized_data_mmpbsa[k]['allAtoms'][index]
            # self.centroid_conf = centroid
            # print(centroid)

            # self.centroid_conf = centroid

            self.clusters_centroids_mmpbsa.append(centroid)

            curr_mmpbsa_centroid = self.simulation_name + '_' + '{0}_cluster_centroid_mmpbsa_{1}.pdb'.format(str(k + 1),
                                                                                                             state)
            curr_mmpbsa_centroid_all_atoms = self.simulation_name + '_' + '{0}_cluster_centroid_mmpbsa_{1}_all_atoms.pdb'.format(
                str(k + 1), state)
            centroid.save(curr_mmpbsa_centroid)
            centroid_all_atoms.save(curr_mmpbsa_centroid_all_atoms)

            self.clusters_centroids_mmpbsa_dict.update({k: {'filename': curr_mmpbsa_centroid,
                                                            'centroid': centroid,
                                                            'index': index,
                                                            'state': state}})

        print('-----------------------------------\n')

    @hlp.timeit
    def find_atoms_interaction_from_gmmpbsa(self, state='stableLong',
                                            cutoff_len=0.26,
                                            data_key=0,
                                            cluster_key=0,
                                            parallel=False,
                                            ligand='QRC',
                                            num_of_threads=7):
        ''''
        Find which atoms interact with aminoacids based on energy contribution from g_mmpbsa
        '''

        print('find_atoms_interaction_from_gmmpbsa is called\n')
        print('-----------------------------------\n')

        # self.called_find_clusters_centroid_mmpbsa = True
        #
        # self.clusters_centroids_mmpbsa_dict = {}
        #
        # self.clusters_centroids_mmpbsa = []

        # for k in self.clusterized_data_mmpbsa:
        print('Finding atoms interactions from mmpbsa for cluster {0}'.format(cluster_key))
        clust_temp_data = self.clusterized_data_mmpbsa[cluster_key][state]

        topology = clust_temp_data.topology

        key = 0
        temp_data = self.analyzed_mmpbsa_data[data_key]

        all_contribs = temp_data['mostAllContrib']

        residue_num = all_contribs['ResidueNum']

        dataframe_topology = topology.to_dataframe()[0]

        # dataframe_topology_smaller = dataframe_topology.drop(
        #     ['name', 'element', 'chainID', 'segmentID'], axis=1)

        dataframe_topology_smaller = dataframe_topology.drop(
            ['segmentID'], axis=1)

        cutoff_len = 0.4

        residue_data_filtered = {}
        residue_neighbour_data = {}

        import pickle
        prot = pickle.HIGHEST_PROTOCOL

        for curr_residue in residue_num:
            # print(curr_residue)

            atoms_closest = topology.select('residue ' + str(curr_residue))

            test = 1

            neighbours_data = md.compute_neighbors(clust_temp_data, cutoff_len, atoms_closest, haystack_indices=None,
                                                   periodic=False)
            residue_neighbour_data.update({curr_residue: neighbours_data})

            # TODO no ligand what the hell very buggy
            test = 1

            if parallel is False:
                results_dict = {}
                for frame in range(len(clust_temp_data)):
                    temp_data = filter_items.run_neighbour_ligand_analysis_parallel(frame, dataframe_topology_smaller,
                                                                                    neighbours_data[frame],
                                                                                    ligand=ligand)
                    results_dict.update(temp_data)
                residue_data_filtered.update({curr_residue: results_dict})
            else:

                # print('Highest protocol ', prot)
                neighbour_frames = list(range(len(clust_temp_data)))

                # Find what is provoking problem
                function_arguments_to_call = ([frame, dataframe_topology_smaller, neighbours_data[frame], ligand] for
                                              frame in
                                              neighbour_frames)

                test = 1
                pool = multiprocessing.Pool(num_of_threads)
                # ALWAYS MAKE PARALLEL CALCULATION METHOD SEPARATE FROM CLASS
                results = pool.starmap(filter_items.run_neighbour_ligand_analysis_parallel, function_arguments_to_call)

                pool.close()
                pool.join()

                results_dict = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}

                residue_data_filtered.update({curr_residue: results_dict})

                test = 1
                time.sleep(2)

            test = 1

        # pickle.dump(self.cluster_models, open(filename, "wb"))
        filename_test = self.simulation_name + '_' + "datkey_" + str(data_key) + '_clust_' + str(cluster_key) + '.pkl'

        final_data = {'topology': dataframe_topology,
                      'topolSmall': dataframe_topology_smaller,
                      'residueData': residue_data_filtered,
                      'neighbour_data': residue_neighbour_data}

        pickle.dump(final_data, open(filename_test, "wb"))

        print('-----------------------------------\n')

        test = 1

        print('-----------------------------------\n')

    @hlp.timeit
    def visualize_interactions_pymol(self):
        # self.clusters_centroids_mmpbsa_dict
        # self.filtered_neighbours
        test = 1

        print('Start of Pymol MD analyis show smethod --->  ')

        for centroid_num in self.clusters_centroids_mmpbsa_dict:
            print('Visualising MMPBSA centroid {0}'.format(centroid_num))
            curr_data = self.clusters_centroids_mmpbsa_dict[centroid_num]
            test = 1

            centroid = curr_data['centroid']
            curr_dssp = self.compute_dssp(custom=centroid, simplified_state=True)

            save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + \
                              'centroid:{0}_mdAnalysis_pymolViz.pse'.format(centroid_num)
            pymol_tools.generate_pymol_interaction_viz(curr_data, curr_dssp, self.filtered_neighbours, save_state_name)

            time.sleep(5)

        # t = threading.Thread(target=self.generate_pymol_viz_for_thread,
        #                      args=(key, self.receptor_file_viz, self.full_data_mdtraj_analysis, save_state_name))
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

    ####################################################################################################################

    @hlp.timeit
    def analyze_pct_change_cluster(self,
                                   selection_obj='protein',
                                   cluster_selection=None,
                                   select_lig=None,
                                   save_data=False, nth_frame=1,
                                   mmpbsa_time=15000):
        pass

    # need to find a way to extract models correctrly
    @hlp.timeit
    def export_cluster_models(self,
                              selection_obj='protein',
                              cluster_selection=None,
                              rmsd_selection=None,
                              select_lig=None,
                              save_data=False,
                              full_save_data=False,
                              nth_frame=1,
                              mmpbsa_time=10000,
                              interval_frame=100,
                              final_timestep=1,
                              relative_stability=0.5):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''
        # n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        # self.cluster_selection_analysis_data.update({selection: {'clusterInfo': self.clusters_info,
        #                                                          'clustNum': self.clust_num,
        #                                                          'clusterAnalysisInfo': self.

        if cluster_selection is None:
            if self.cluster_selection_analysis_data[selection_obj]['overrideClustNum'] is None:
                n_clusters = self.cluster_selection_analysis_data[selection_obj]['clustNum']
            else:
                n_clusters = self.cluster_selection_analysis_data[selection_obj]['overrideClustNum']

            clusters_info = self.cluster_selection_analysis_data[selection_obj]['clusterInfo']
            cluster_labels = clusters_info[n_clusters]['labels']
            labels = cluster_labels
            unique_labels = list(set(cluster_labels))
            print('Unique labels ', unique_labels)

            if rmsd_selection is None:
                rmsd_to_analyze = self.rmsd_analysis_data[selection_obj]
            else:
                rmsd_to_analyze = self.rmsd_analysis_data[rmsd_selection]
            cluster_selection = selection_obj



        else:
            if self.cluster_selection_analysis_data[cluster_selection]['overrideClustNum'] is None:
                n_clusters = self.cluster_selection_analysis_data[cluster_selection]['clustNum']
            else:
                n_clusters = self.cluster_selection_analysis_data[cluster_selection]['overrideClustNum']

            clusters_info = self.cluster_selection_analysis_data[cluster_selection]['clusterInfo']
            cluster_labels = clusters_info[n_clusters]['labels']
            labels = cluster_labels
            unique_labels = list(set(cluster_labels))

            if rmsd_selection == None:
                rmsd_to_analyze = self.rmsd_analysis_data[cluster_selection]
            else:
                test = 1
                rmsd_to_analyze = self.rmsd_analysis_data[rmsd_selection]

        # sample_silhouette_values =clusters_info[n_clusters]['silhouette_values']
        # silhouette_avg = sclusters_info[n_clusters]['silhouette']

        # centers = clusters_info[n_clusters]['centers']

        original_data = self.full_traj

        self.clusterized_data = {}

        self.clusterized_data_mmpbsa = {}

        timestep = original_data.timestep
        print('timestep is ', timestep)

        frame_step = math.ceil(interval_frame / timestep)

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                print('Cluster Label --> {0}'.format(k))
                cluster_data = []
                xyz = original_data[labels == k]
                xyz_no_modif = xyz[:]

                rmsd_cluster = rmsd_to_analyze[labels == k]
                sim_cluster = self.sim_time[labels == k] * 1000

                rmsd_cluster_dataframe = pd.DataFrame(rmsd_cluster)

                pct_change_data = converters.convert_data_to_pandas(sim_cluster,
                                                                    rmsd_cluster,
                                                                    x_axis_name='time',
                                                                    y_axis_name='rmsd')
                # pct_change = rmsd_cluster_dataframe.pct_change()
                pct_change = pct_change_data.pct_change()

                # pct_change[abs(pct_change) < 0.2 ]

                # TODO very important how much percentage change 0.2 or 0.3
                data = pct_change_data[np.abs(pct_change['rmsd']) <= relative_stability]

                # data.iloc[2]-data.iloc[1]

                # data.diff() yes

                data_diff = data.diff()

                data_filter = data[data_diff == timestep]

                # TODO here's very important bug
                non_nan_blocks = filter_items.filter_non_nan_blocks(data_filter, axis_name='time')

                # g_mmpbsa_traj = temp_data[temp_data.time.max() - temp_data.time <= mmpbsa_time]

                non_nan_blocks_keys = non_nan_blocks.keys()

                keys_int = [int(i) for i in non_nan_blocks_keys]

                max_key = max(keys_int)

                data_to_work = np.array(non_nan_blocks[str(max_key)])

                test = 1

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

                temp_data = sel_traj[::nth_frame]

                temp_data_all_atoms = xyz_no_modif[::nth_frame]

                print('------------------------------------------\n')
                print('Cluster Label --> {0}'.format(k))
                print('Cluster Traj ', temp_data)
                print('Cluster End time is ', temp_data.time.max())
                print('Cluster Start time is ', temp_data.time.min())
                print('------------------------------------------\n')

                # SELECT TIME INTERVALS FROM STABLE RMSD s
                min_index = list(temp_data.time).index(data_to_work.min())
                max_index = list(temp_data.time).index(data_to_work.max())

                # temp_data.time.max()
                # temp_data.time.max()-temp_data.time < 15
                g_mmpbsa_traj = temp_data[min_index:max_index + 1]
                g_mmpbsa_traj_all_atoms = temp_data_all_atoms[min_index:max_index + 1]

                time_g_mmpbsa_traj = g_mmpbsa_traj.time.max() - g_mmpbsa_traj.time.min()
                print('------------------------------------------\n')
                print('G_mmpbsa traj ', g_mmpbsa_traj)
                print('G_mmpbsa End time is ', g_mmpbsa_traj.time.max())
                print('G_mmpbsa Start time is ', g_mmpbsa_traj.time.min())
                print('------------------------------------------\n')

                print('================Cluster {0}==========================\n'.format(k))

                g_mmpbsa_traj_final = g_mmpbsa_traj[g_mmpbsa_traj.time.max() - g_mmpbsa_traj.time <= mmpbsa_time]

                g_mmpbsa_traj_final_all_atoms = g_mmpbsa_traj_all_atoms[
                    g_mmpbsa_traj_all_atoms.time.max() - g_mmpbsa_traj_all_atoms.time <= mmpbsa_time]



                g_mmpbsa_traj_final_last = temp_data[temp_data.time.max() - temp_data.time <= mmpbsa_time]
                print('G_mmpbsa traj last  ', g_mmpbsa_traj_final_last)
                print('G_mmpbsa End time is ', temp_data.time.max())
                #print('G_mmpbsa Start time is ', g_mmpbsa_traj.time.min())

                if save_data is True:
                    test = 1
                    # g_mmpbsa_traj = temp_data[-400:-1]

                    # temp_data.time[temp_data.time == data_to_work.min()]
                    # temp_data[temp_data.time == data_to_work]

                    save_name = 'SimClustNum{0}_clustSel:{1}_auto_'.format(k, cluster_selection) + 'cluster_' + str(
                        clust_num) + '_' + selection_final_name + '_time:{0}->{1}_{2}_g_mmpbsa_short'.format(
                        g_mmpbsa_traj_final.time.min(),
                        g_mmpbsa_traj_final.time.max(),
                        mmpbsa_time)
                    print('Save-> ', save_name)
                    print('---------------------------------------------------------------------------')

                    g_mmpbsa_traj_final[::final_timestep].save(save_name + '.pdb')
                    g_mmpbsa_traj_final[::final_timestep].save(save_name + '.xtc')

                    g_mmpbsa_traj_final_all_atoms[::3].save(save_name + 'all_atoms.xtc')

                    temp_data[0].save(self.simulation_name + '_' + 'cluster_' + str(
                        clust_num) + '_' + selection_final_name + '_frame_0.pdb')

                    full_save = self.simulation_name + '_' + 'cluster_' + str(clust_num) + '_' + selection_final_name + \
                                '_time:{0}->{1}.xtc'.format(temp_data.time.min(),
                                                            temp_data.time.max())

                    if full_save_data is True:
                        print('Full save Name -> ', full_save)
                        temp_data.save(full_save)
                    print('------------------------>>>-------------------------->>>>>>>>------------------------------')

                    g_mmpbsa_traj.save(
                        self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_frame_step:{0}_stable_{1}_g_mmpbsa.xtc'.format(
                            frame_step, time_g_mmpbsa_traj))

                    g_mmpbsa_traj_final.save(
                        self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_last_from_stable_{0}_g_mmpbsa.xtc'.format(
                            mmpbsa_time))

                    # save in pdb format for easier visualization
                    g_mmpbsa_traj_final.save(
                        self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_last_from_stable_{0}_g_mmpbsa.pdb'.format(
                            mmpbsa_time))


                    g_mmpbsa_traj_final_last.save(
                        self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_last_stable_frames_{0}_g_mmpbsa.xtc'.format(
                            mmpbsa_time))

                    # save in pdb format for easier visualization
                    g_mmpbsa_traj_final_last.save(
                        self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_last_stable_frames_{0}_g_mmpbsa.pdb'.format(
                            mmpbsa_time))


                self.clusterized_data.update({k: sel_traj})

                self.clusterized_data_mmpbsa.update({k: {'stableLong': g_mmpbsa_traj_final,
                                                         'stable5ns': g_mmpbsa_traj,
                                                         'allAtoms': g_mmpbsa_traj_final_all_atoms,
                                                         'lastStable':g_mmpbsa_traj_final_last}})

        self.save_pdb_hbond = True

    # TODO This loads pickle data of aa energy interaction
    @hlp.timeit
    def pickle_load_mmpbsa_analysis(self, filename):
        analysed_data = pickle.load(open(filename, "rb"))

        key = list(analysed_data.keys())[0]

        if key not in self.analyzed_mmpbsa_data:
            test = 1
            self.analyzed_mmpbsa_data.update(analysed_data)
        print('test')
        test = 1

    # need to find a way to extract models correctrly
    @hlp.timeit
    def export_cluster_models_manual(self,
                                     manual_extraction,
                                     selection_obj='protein',
                                     cluster_selection=None,
                                     rmsd_selection=None,
                                     select_lig=None,
                                     save_data=False, nth_frame=1,
                                     mmpbsa_time=15000,
                                     interval_frame=500,
                                     relative_stability=0.5):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''
        # n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        # self.cluster_selection_analysis_data.update({selection: {'clusterInfo': self.clusters_info,
        #                                                          'clustNum': self.clust_num,
        #                                                          'clusterAnalysisInfo': self.

        if cluster_selection is None:
            if self.cluster_selection_analysis_data[selection_obj]['overrideClustNum'] is None:
                n_clusters = self.cluster_selection_analysis_data[selection_obj]['clustNum']
            else:
                n_clusters = self.cluster_selection_analysis_data[selection_obj]['overrideClustNum']

            clusters_info = self.cluster_selection_analysis_data[selection_obj]['clusterInfo']
            cluster_labels = clusters_info[n_clusters]['labels']
            labels = cluster_labels
            unique_labels = list(set(cluster_labels))
            print('Unique labels ', unique_labels)

            if rmsd_selection is None:
                rmsd_to_analyze = self.rmsd_analysis_data[selection_obj]
            else:
                rmsd_to_analyze = self.rmsd_analysis_data[rmsd_selection]
            cluster_selection = selection_obj



        else:
            if self.cluster_selection_analysis_data[cluster_selection]['overrideClustNum'] is None:
                n_clusters = self.cluster_selection_analysis_data[cluster_selection]['clustNum']
            else:
                n_clusters = self.cluster_selection_analysis_data[cluster_selection]['overrideClustNum']

            clusters_info = self.cluster_selection_analysis_data[cluster_selection]['clusterInfo']
            cluster_labels = clusters_info[n_clusters]['labels']
            labels = cluster_labels
            unique_labels = list(set(cluster_labels))

            if rmsd_selection is None:
                rmsd_to_analyze = self.rmsd_analysis_data[cluster_selection]
            else:
                rmsd_to_analyze = self.rmsd_analysis_data[rmsd_selection]

        # sample_silhouette_values =clusters_info[n_clusters]['silhouette_values']
        # silhouette_avg = sclusters_info[n_clusters]['silhouette']

        # centers = clusters_info[n_clusters]['centers']

        original_data = self.full_traj

        self.clusterized_data = {}

        self.clusterized_data_mmpbsa = {}

        timestep = original_data.timestep

        frame_step = math.ceil(interval_frame / timestep)

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                print('Cluster Label --> {0}'.format(k))
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

                temp_data = sel_traj[::nth_frame]

                for i in manual_extraction:
                    key = list(i.keys())[0]

                    if int(key) == k:

                        start_key = i[key]['start']
                        end_key = i[key]['end']

                        if start_key is None:
                            start = 0
                        else:
                            start = start_key

                        if end_key is None:
                            end = -1
                        else:
                            end = end_key + 1

                        g_mmpbsa_traj_manual = temp_data[start:end]

                        print('Manual End time is ', g_mmpbsa_traj_manual.time.max())
                        print('Manal Start time is ', g_mmpbsa_traj_manual.time.min())

                        g_mmpbsa_traj_final = g_mmpbsa_traj_manual[
                            g_mmpbsa_traj_manual.time.max() - g_mmpbsa_traj_manual.time <= mmpbsa_time]

                        save_name = 'Sim_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                            clust_num) + '_' + selection_final_name + '_time:{0}->{1}_{2}_g_mmpbsa'.format(
                            g_mmpbsa_traj_manual.time.min(),
                            g_mmpbsa_traj_manual.time.max(),
                            mmpbsa_time)

                        g_mmpbsa_traj_final[::3].save(save_name + '.pdb')
                        g_mmpbsa_traj_final[::3].save(save_name + '.xtc')
                        test = 1

                        #         min_index = list(temp_data.time).index(data_to_work.min())
                        #         max_index = list(temp_data.time).index(data_to_work.max())
                        #
                        #         # temp_data.time.max()
                        #         # temp_data.time.max()-temp_data.time < 15
                        #         g_mmpbsa_traj = temp_data[min_index:max_index + 1]
                        #
                        #         time_g_mmpbsa_traj = g_mmpbsa_traj.time.max() - g_mmpbsa_traj.time.min()
                        #         print('G_mmpbsa traj ', g_mmpbsa_traj)
                        #         print('End time is ', g_mmpbsa_traj.time.max())
                        #         print('Start time is ', g_mmpbsa_traj.time.min())
                        #         print('------------------------------------------\n')
                        #
                        #         g_mmpbsa_traj_final = g_mmpbsa_traj[g_mmpbsa_traj.time.max() - g_mmpbsa_traj.time <= mmpbsa_time]
                        #
                        #         if save_data is True:
                        #             test = 1
                        #             # g_mmpbsa_traj = temp_data[-400:-1]
                        #
                        #             # temp_data.time[temp_data.time == data_to_work.min()]
                        #             # temp_data[temp_data.time == data_to_work]
                        #
                        #
                        #
                        #             #
                        #             # temp_data[0].save(self.simulation_name + '_' + 'cluster_' + str(
                        #             #     clust_num) + '_' + selection_final_name + '_frame_0.pdb')
                        #             #
                        #             #
                        #             # temp_data.save(
                        #             #     self.simulation_name + '_' + 'cluster_' + str(
                        #             #         clust_num) + '_' + selection_final_name + '.xtc')
                        #
                        #             g_mmpbsa_traj.save(
                        #                 self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                        #                     clust_num) + '_' + selection_final_name + '_frame_step:{0}_stable_{1}_g_mmpbsa.xtc'.format(
                        #                     frame_step, time_g_mmpbsa_traj))
                        #
                        #             g_mmpbsa_traj_final.save(
                        #                 self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                        #                     clust_num) + '_' + selection_final_name + '_last_from_stable_{0}_g_mmpbsa.xtc'.format(
                        #                     mmpbsa_time))
                        #
                        #             # save in pdb format for easier visualization
                        #             g_mmpbsa_traj_final.save(
                        #                 self.simulation_name + '_clustSel:{0}_'.format(cluster_selection) + 'cluster_' + str(
                        #                     clust_num) + '_' + selection_final_name + '_last_from_stable_{0}_g_mmpbsa.pdb'.format(
                        #                     mmpbsa_time))
                        #
                        #         self.clusterized_data.update({k: sel_traj})
                        #
                        #         self.clusterized_data_mmpbsa.update({k: {'stableLong': g_mmpbsa_traj_final,
                        #                                                  'stable5ns': g_mmpbsa_traj}})
                        #
                        # self.save_pdb_hbond = True

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

        self.called_rmsf_analysis = True

        self.called_rmsf_calc = True
        print('RMSF analysis has been called for selection: {0} \n'.format(atom_indices))
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

        # TODO here is the problem with super imposition
        trajectory = self.full_traj[:]
        # trajectory = self.full_traj.atom_slice(atom_indices=atom_indices, inplace=False)

        trajectory.superpose(trajectory[frame], atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)

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
        # Free space
        del trajectory

        import gc
        gc.collect()
        len(gc.get_objects())

        print('RMSF for selection has been finished-----------------------------\n')

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
    def show_silhouette_analysis_pca_best(self, selection, show_plot=False, custom_dpi=300):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        # n_clusters, clustus_info = select_number_of_clusters_v2(self.clusters_info, self.range_n_clusters)

        if self.cluster_selection_analysis_data[selection]['overrideClustNum'] is None:
            n_clusters = self.cluster_selection_analysis_data[selection]['clustNum']
        else:
            n_clusters = self.cluster_selection_analysis_data[selection]['overrideClustNum']

        clustus_info = self.cluster_selection_analysis_data[selection]['clusterInfo']
        cluster_analysis_info = self.cluster_selection_analysis_data[selection]['clusterAnalysisInfo']

        cluster_labels = clustus_info[n_clusters]['labels']
        sample_silhouette_values = clustus_info[n_clusters]['silhouette_values']
        silhouette_avg = clustus_info[n_clusters]['silhouette']

        centers = clustus_info[n_clusters]['centers']

        X = self.md_pca_analysis_data[selection]  # self.reduced_cartesian

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(18, 7)
        fig.set_size_inches(plot_tools.cm2inch(17.7, 8.4))

        sns.set(font_scale=1)

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

        self.cluster_selection_color_data.update({selection: {'cubehelix': self.colors_}})

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

        self.cluster_selection_color_data.update({selection: {'colorData': self.cluster_colors}})

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
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")

        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")

        # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

        plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
                      "with n_clusters = %d" % n_clusters),
                     fontweight='bold')

        fig.tight_layout()

        fig.savefig(self.simulation_name + '_{0}_'.format(selection) + 'Best_cluster_analysis_md_' + '.png',
                    dpi=custom_dpi,
                    bbox_inches='tight')
        if show_plot is True:
            plt.show()

        import gc
        gc.collect()
        len(gc.get_objects())

        plt.cla()
        plt.close(fig)

    @hlp.timeit
    def show_cluster_analysis_pca_best(self, selection, show_plot=False, custom_dpi=600):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        if self.cluster_selection_analysis_data[selection]['overrideClustNum'] is None:
            n_clusters = self.cluster_selection_analysis_data[selection]['clustNum']
        else:
            n_clusters = self.cluster_selection_analysis_data[selection]['overrideClustNum']

        clustus_info = self.cluster_selection_analysis_data[selection]['clusterInfo']
        cluster_analysis_info = self.cluster_selection_analysis_data[selection]['clusterAnalysisInfo']

        cluster_labels = clustus_info[n_clusters]['labels']
        sample_silhouette_values = clustus_info[n_clusters]['silhouette_values']
        silhouette_avg = clustus_info[n_clusters]['silhouette']

        centers = clustus_info[n_clusters]['centers']

        X = self.md_pca_analysis_data[selection]  # self.reduced_cartesian

        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        # fig.set_size_inches(18, 7)

        sns.set(font_scale=1)

        # TODO a new try
        colors = self.colors_

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        plt.scatter(X[:, 0], X[:, 1], marker='.', s=80, lw=0, alpha=0.7,
                    c=colors)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=250, lw=0, alpha=0.7,
        #             c=self.full_traj.time)

        # Labeling the clusters

        # Draw white circles at cluster centers
        plt.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=140)

        for i, c in enumerate(centers):
            clust_num = i + 1
            plt.scatter(c[0], c[1], marker='$%d$' % clust_num, alpha=1, s=130)

        plt.title("The visualization of the clustered data")
        # plt.xlabel("Feature space for the 1st feature")
        # plt.ylabel("Feature space for the 2nd feature")

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        # plt.suptitle(("Silhouette analysis for KMeans clustering on conformation data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')

        fig.savefig(self.simulation_name + '_{0}_'.format(selection) + 'Best_cluster_analysis_simple_md_' + '.png',
                    dpi=custom_dpi,
                    bbox_inches='tight')
        if show_plot is True:
            plt.show()

        plt.cla()
        plt.close(fig)

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
                # ax2.set_xlabel("Feature space for the 1st feature")
                # ax2.set_ylabel("Feature space for the 2nd feature")

                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')

                plt.show()

            plt.close(fig)

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
            # ax2.set_xlabel("Feature space for the 1st feature")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")

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

        # traj = self.full_traj[:]
        # topology = traj.topology
        #
        # selection = topology.select(selection)
        #
        # traj.restrict_atoms(selection)
        topology = self.full_traj.topology
        selection = topology.select(selection)
        traj = self.full_traj.atom_slice(atom_indices=selection)

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
        # self.experimental_val = self.experimental_val.convert_objects(convert_numeric=True)
        self.experimental_val = pd.to_numeric(self.experimental_val)

        self.experimental_error = type_experimental_data['Val_err']
        # self.experimental_error = self.experimental_error.convert_objects(convert_numeric=True)
        self.experimental_error = pd.to_numeric(self.experimental_error)

        test = 1

    def plot_whole_nmr(self, atom_type, errorbar=False, custom_dpi=600):

        import pylab as plt
        sns.set(style="ticks", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        # fig = plt.figure(figsize=(14, 7))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # residues = list(range(res_start, res_end+1))

        colors = sns.cubehelix_palette(n_colors=3, start=2.8, rot=.1)
        sns.set(font_scale=1)

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
        # sns.set(style="ticks", context='paper')
        sns.set(style="whitegrid", context='paper')
        '''
        ylabel=r"C$_\alpha$ RMSD from $t=0$, $\rho_{\mathrm{C}_\alpha}$ (nm)",
        '''
        plt.clf()
        fig = plt.figure(figsize=(14, 7))
        # ax = fig.add_axes([2, 2, 2, 2])
        # plt.plot(time_sim1_np, rmsd_sim1_np, 'b')
        # residues = list(range(res_start, res_end+1))

        # colors = sns.cubehelix_palette(n_colors=3, start=2.8, rot=.1)
        colors = sns.color_palette(n_colors=3)
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

        test = 1

        print('------------------------------------------------------------------------------\n')
        max_ppm_diff = 12
        for i in range(max_x - 2):
            exp_val = list(self.experimental_val)[i]
            sparta_val = self.sparta_mean[i]
            shift_val = self.shiftx2_mean[i]
            # or abs(shift_val-exp_val) < 10
            if abs(sparta_val - exp_val) > max_ppm_diff or abs(shift_val - exp_val) > max_ppm_diff:
                print('abs(sparta_val-exp_val) ', abs(sparta_val - exp_val))
                print('sparta val ', sparta_val)
                print('shift val ', shift_val)
                print('exper val', exp_val)
                print('Major residue difference ', i)

        ax.errorbar(list(self.residues)[:max_x], self.experimental_val, yerr=list(self.experimental_error), color='red',
                    linewidth=0.6, label='Experimental', elinewidth=2, fmt='o', zorder=10, capthick=1, capsize=2,
                    alpha=0.65, ms=8.2)

        ax.errorbar(self.residues_sparta[:max_x], self.sparta_mean[:max_x], yerr=self.sparta_std[:max_x], color='blue',
                    linewidth=0.8, label='Sparta+', elinewidth=0.5, fmt='^', zorder=1, capthick=1, capsize=2)

        ax.errorbar(self.residues_shiftx2[:max_x], self.shiftx2_mean[:max_x], yerr=self.shiftx2_std[:max_x],
                    color='green',
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

        legend = plt.legend(flip(handles, 3), flip(labels, 3), loc=4, ncol=3, frameon=True)
        frame = legend.get_frame()

        frame.set_linewidth(2)
        frame.set_edgecolor("black")

        min_y = min(self.experimental_val)

        # TODO need to think about this for errorbar
        ax.set_ylim(bottom=min_y - 4)

        ax.set_xlim(min_x - 5, max_x + 5)

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

    def plot_nmr_jointplot(self, atom_type='CA', res_start=1, res_end=171, custom_dpi=600):

        pd_data1 = converters.convert_data_to_pandas(self.experimental_val, self.sparta_mean[res_start:res_end],
                                                     x_axis_name='Experimental',
                                                     y_axis_name='Sparta+')

        pd_data2 = converters.convert_data_to_pandas(self.experimental_val, self.shiftx2_mean[res_start:res_end],
                                                     x_axis_name='Experimental',
                                                     y_axis_name='ShiftX2')

        if atom_type == 'CA':
            atom_title = r'C$\alpha$'
        else:
            atom_title = atom_type

        # the size of A4 paper

        # fig, ax = plt.subplots(figsize=plot_tools.cm2inch(17.7, 14))
        sns.set(font_scale=1.0)

        # size_x = 17.7
        # size_y = 12

        size_x = 8.4
        size_y = 8.4

        top_adjust = 0.88

        title = 'Chemical shift values for {0}-atom.\n Experimental vs Sparta+'.format(atom_title)
        g = sns.jointplot(x="Experimental", y="Sparta+", data=pd_data1, kind="reg")
        g.fig.set_size_inches(plot_tools.cm2inch(size_x, size_y))

        g.fig.suptitle(title)

        sns.set(font_scale=0.9)
        g.fig.tight_layout()

        g.fig.subplots_adjust(top=top_adjust)
        g.fig.savefig(title + '.png', dpi=custom_dpi)

        # fig, ax = plt.subplots(figsize=plot_tools.cm2inch(17.7, 14))

        f = sns.jointplot(x="Experimental", y="ShiftX2", data=pd_data2, kind="reg")
        title = 'Chemical shift values for {0}-atom.\n Experimental vs ShiftX2'.format(atom_title)
        f.fig.set_size_inches(plot_tools.cm2inch(size_x, size_y))

        f.fig.suptitle(title)

        sns.set(font_scale=1.0)
        f.fig.tight_layout()

        # subplots adjust after tight layout
        f.fig.subplots_adjust(top=top_adjust)
        f.fig.savefig(title + '.png', dpi=custom_dpi)
        # sns.plt.show()

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

        topology = self.full_traj.topology

        selection_indices = self.topology.select(selection)

        traj = self.full_traj.atom_slice(atom_indices=selection_indices)

        # traj = self.full_traj[:]
        # topology = traj.topology
        #
        # selection = topology.select(selection)
        #
        # traj.restrict_atoms(selection)

        self.sasa_traj = traj[from_frame:to_frame:stride]
        print(self.sasa_traj)

        # TODO start by for single machine ipcluster start -n 4
        self.sasa, self.total_sasa = protein_analysis.calc_sasa(self.sasa_traj, parallel,
                                                                n_sphere_points=n_sphere_points)

        self.sasa_analysis_data.update({selection: {'sasa': self.sasa,
                                                    'totalSasa': self.total_sasa,
                                                    'sasaTraj': self.sasa_traj}})

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
    def plot_simple_md_pca_analysis(self, selection='protein', title=None, custom_dpi=600, show=False,
                                    transparent_alpha=False):
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)

        # cmap = sns.cubehelix_palette(n_colors=len(self.pca_traj.time), as_cmap=True, reverse=True)
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        time_ns = self.pca_traj.time / 1000

        data = self.md_pca_analysis_data[selection]

        plt.scatter(data[:, 0], data[:, 1], marker='o', s=60, c=time_ns)
        # plt.scatter(self.reduced_cartesian[:, 0], self.reduced_cartesian[:, 1], marker='o', s=60, c=cmap)
        # plt.xlabel('Feature space for the 1st feature')
        # plt.ylabel('Feature space for the 2nd feature')

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        if title is None:
            title_to_use = 'Conformation PCA Analysis: {0}'.format(selection)
        else:
            title_to_use = title

        # plt.title(title_to_use)

        cbar = plt.colorbar()
        cbar.set_label('Time $t$ (ns)')
        fig.savefig(self.simulation_name + '_' + selection + '_simple_PCA_analysis' + '.png', dpi=custom_dpi,
                    bbox_inches='tight',
                    transparent=transparent_alpha)
        if show is True:
            plt.show()
        print("simple PCA plot: {0}  -> created".format(selection))
        print('-----------------------------------\n')

    ####################################################################################################################

    # TODO kmeans cluster analysis on PCA data

    # @profile
    @hlp.timeit
    def md_full_cluster_analysis(self, selection, data_type='PCA', algorithm='kmeans', parallel=True,
                                 num_of_threads=7, show=False, connectivity=True):
        print('MD full cluster analysis is called!!!!! Be Ready \n')
        if data_type == 'PCA':
            # data_to_analyze = self.md_pca_analysis_data[selection]['data']
            # n_dim= self.md_pca_analysis_data[selection]['n_dim']

            data_to_analyze = self.md_pca_analysis_data[selection]

            data_to_analyze_time = self.pca_traj.time
            connectivity = connectivity
            self.simulation_name = self.initial_name + '_type:PCA_'
        elif data_type == 'PairRMSD':
            # This consumes a lot of RAM
            data_to_analyze = self.md_pairwise_rmsd_analysis_data[selection]
            data_to_analyze_time = self.sim_time
            connectivity = False
            self.simulation_name = self.initial_name + '_type:PairRMSD_selection:{0}_'.format(selection)
            # self.range_n_clusters = list(range(2, self.k_clust + 1))

        if parallel is True:
            self.parallel_cluster_proc = []

            # self.simultaneous_run = list(range(0, num_of_threads))

            pool = multiprocessing.Pool(num_of_threads)

            # range_n_clusters = list(range(1, 11))

            function_arguments_to_call = [[x, data_to_analyze, data_to_analyze_time, algorithm, connectivity] for x in
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

            # This will free memory
            pool.close()
            test = 1

            # self.parallel_cluster_proc.append(proc)
            # proc.start()# start now

        elif parallel is False:
            self.clusters_info = md_silhouette_analysis_pca(data_to_analyze,
                                                            data_to_analyze_time,
                                                            range_n_clusters=self.range_n_clusters,
                                                            show_plots=show,
                                                            algorithm=algorithm,
                                                            connectivity=connectivity)

        # self.sil_pca = self.extract_info_cluster_data(self.clusters_info, 'silhouette')
        # self.calinski_pca = self.extract_info_cluster_data(self.clusters_info, 'calinski')
        # self.dunn_pca = self.extract_info_cluster_data(self.clusters_info, 'dunn')
        # self.dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'dbi')
        #
        # self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')
        # self.book_dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'book_dbi')

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()
        test = 1
        # TODO a new version
        self.clust_num, self.clust_analysis_info = select_number_of_clusters_v2(self.clusters_info,
                                                                                self.range_n_clusters)

        print('Number of Clusters for Selection:{0} = {1} \n'.format(selection, self.clust_num))
        self.cluster_selection_analysis_data.update({selection: {'clusterInfo': self.clusters_info,
                                                                 'clustNum': self.clust_num,
                                                                 'clusterAnalysisInfo': self.clust_analysis_info,
                                                                 'overrideClustNum': None}})
        test = 1

        import gc
        gc.collect()
        len(gc.get_objects())

        test = 1

        # Save clustering analysis
        self.save_processed_clustering()
        # self.clust_num = self.clust_analysis_info['clustNum']
        # self.clust_num = self.select_number_of_clusters()

        # self.cluster_list = self.collect_cluster_info()

    def override_clust_num(self, selection, clust_num):
        self.cluster_selection_analysis_data[selection]['overrideClustNum'] = clust_num
        print('Clust num for selection: {0} has been overriden with {1}'.format(selection, clust_num))

    def save_processed_clustering(self):
        save_processed_data_info = {}

        save_processed_data_info.update({'clusterAnalyis': self.cluster_selection_analysis_data})

        save_processed_data_info.update({'simulationName': self.simulation_name})
        import pickle
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        filename = self.simulation_name + '_clusterAnalysis_data.pickle'
        pickle.dump(save_processed_data_info, open(filename, "wb"))

    def load_cluster_analysis(self, filename):
        '''

        :param filename: load pickle file
        :return:
        '''
        print('Load Cluster Analysis has been calledd !!!!! \n')
        import pickle
        analysed_data = pickle.load(open(filename, "rb"))
        self.cluster_selection_analysis_data = analysed_data['clusterAnalyis']
        self.simulation_name = analysed_data['simulationName']
        print('test')

    ####################################################################################################################
    # TODO show cluster scoring
    @hlp.timeit
    def show_all_cluster_scores_analysis_plots(self,
                                               selection,
                                               show_plot=False,
                                               custom_dpi=600):
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)
        sns.set(style="ticks", context='paper')
        sns.set(font_scale=1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                     2)  # sharex='col', sharey='row') TODO this can be used for shared columns
        # fig.set_size_inches(20, 20)
        fig.set_size_inches(plot_tools.cm2inch(17.7, 17.7))

        self.clusters_info = self.cluster_selection_analysis_data[selection]['clusterInfo']
        self.sil_pca = self.extract_info_cluster_data(self.clusters_info, 'silhouette')
        self.calinski_pca = self.extract_info_cluster_data(self.clusters_info, 'calinski')
        self.dunn_pca = self.extract_info_cluster_data(self.clusters_info, 'dunn')
        self.dbi_pca = self.extract_info_cluster_data(self.clusters_info, 'dbi')

        cluster_range = self.range_n_clusters
        score = self.dbi_pca
        criteria_name = 'Davis-Bouldain Index'
        score_text = 'The optimal clustering solution\n' \
                     ' has the smallest\n' \
                     ' Davies-Bouldin index value.'
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
                     "represents the right\n" \
                     " partitioning given the index"
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
                     'score value\n' \
                     ' are considered well clustered'
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

        fig.savefig(self.simulation_name + '_{0}_'.format(selection) + 'MD_cluster_scores.png', dpi=custom_dpi,
                    bbox_inches='tight')
        plt.cla()
        plt.close(fig)

        ####################################################################################################################

    # TODO  compute dssp and plot

    #  only need to select protein for dssp analysis
    @hlp.timeit
    def compute_dssp(self, selection='protein', custom=None, simplified_state=False):

        self.called_compute_dssp = True
        print("DSSP has been called\n")
        print('-----------------------------------\n')

        selection_text = selection

        # self.dssp_traj = self.full_traj[:]
        self.dssp_topology = self.full_traj.topology

        self.dssp_selection = self.dssp_topology.select(selection)

        # self.dssp_traj.restrict_atoms(self.dssp_selection)

        if custom is None:
            self.dssp_traj = self.full_traj.atom_slice(atom_indices=self.dssp_selection, inplace=False)
        else:
            self.dssp_traj = custom

        self.dssp_data = md.compute_dssp(self.dssp_traj, simplified=simplified_state)

        # indexes =  self.dssp_traj.time/1000
        self.dssp_df = pd.DataFrame(self.dssp_data)

        if custom is None:
            self.md_dssp_analysis_data.update({selection_text: self.dssp_df})

        return self.dssp_df

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

    def save_processed_rmsd(self):
        save_processed_data_info = {}

        if self.called_rmsd_analysis is True:
            save_processed_data_info.update({'rmsd': self.rmsd_analysis_data})
            save_processed_data_info.update({'time': self.sim_time})

            save_processed_data_info.update({'simulationName': self.simulation_name})
            import pickle
            # pickle.dump(self.cluster_models, open(filename, "wb"))
            filename = self.simulation_name + '_rmsd_processed_data.pickle'
            pickle.dump(save_processed_data_info, open(filename, "wb"))

            del self.rmsd_analysis_data
            del save_processed_data_info

    def save_processed_rg(self):
        save_processed_data_info = {}

        if self.called_rg_analysis is True:
            self.save_processed_data_info.update({'Rg': self.rg_analysis_data})
            save_processed_data_info.update({'time': self.sim_time})

            save_processed_data_info.update({'simulationName': self.simulation_name})
            import pickle
            # pickle.dump(self.cluster_models, open(filename, "wb"))
            filename = self.simulation_name + '_rg_processed_data.pickle'
            pickle.dump(save_processed_data_info, open(filename, "wb"))

            del self.rg_analysis_data
            del save_processed_data_info

    def save_processed_data(self):
        self.save_processed_data_info = {}

        if self.called_md_pca_analysis is True:
            self.save_processed_data_info.update({'PCA': self.pca_traj})

        if self.called_rmsd_analysis is True:
            self.save_processed_data_info.update({'rmsd': self.rmsd_analysis_data})
            self.save_processed_data_info.update({'time': self.sim_time})

        if self.called_rg_analysis is True:
            self.save_processed_data_info.update({'Rg': self.rg_analysis_data})

        if self.called_rmsf_analysis is True:
            self.save_processed_data_info.update({'rmsf': self.rmsf_analysis_data})

        if self.called_calc_solvent_area is True:
            # self.save_processed_data_info.update({'sasa': self.sasa})
            # self.save_processed_data_info.update({'totalSasa': self.total_sasa})
            self.save_processed_data_info.update({'sasa': self.sasa_analysis_data})

        if self.called_compute_dssp is True:
            self.save_processed_data_info.update({'dssp': self.dssp_df})

        if self.called_compute_best_hummer_q is True:
            self.save_processed_data_info.update({'nativeContacts': self.q})

        if self.called_find_clusters_hbond is True:
            self.save_processed_data_info.update({'clustersHBonds': self.hbond_cluster_analysis_data})

        if self.called_hbond_analysis_count is True:
            self.save_processed_data_info.update({'hbondCount': self.hbond_count})
            self.save_processed_data_info.update({'hbondFrames': self.hbonds_frames})

        if self.called_find_max_cluster_centroid is True:
            self.save_processed_data_info.update({'centroidConf': self.centroid_conf})

        if self.called_ramachandran_centroid_calc is True:
            self.save_processed_data_info.update({'ramachandranCentroid': self.angles_calc_centroid})

        if self.called_find_clusters_centroid is True:
            self.save_processed_data_info.update({'clustersCentroid': self.clusters_centroids})

        self.save_processed_data_info.update({'simulationName': self.simulation_name})
        import pickle
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        filename = self.simulation_name + '_processed_data.pickle'
        pickle.dump(self.save_processed_data_info, open(filename, "wb"))

    # @profile
    def memory_profile_data(self):

        test = self.range_n_clusters

        # This part is for checking which methods were called
        test = self.called_md_pca_analysis

        test = self.called_rmsd_analysis

        test = self.called_rg_analysis

        test = self.called_rmsf_calc

        test = self.called_hbond_analysis_count

        test = self.called_calc_solvent_area

        test = self.called_compute_dssp

        test = self.called_compute_best_hummer_q

        test = self.called_find_max_cluster_centroid

        test = self.called_ramachandran_centroid_calc

        test = self.called_rmsf_analysis

        test = self.called_find_clusters_centroid

        test = self.called_find_clusters_hbond

        # DATA ANALYSIS OBJECTS

        test = self.rmsd_analysis_data
        test = self.rg_analysis_data
        test = self.rmsf_analysis_data

        test = self.sasa_analysis_data

        test = self.hbond_cluster_analysis_data

        test = self.center_of_mass_analysis_data

        test = self.md_pca_analysis_data

        test = self.cluster_selection_analysis_data

        test = self.cluster_selection_color_data

        test = self.k_clust

        test = self.full_traj
