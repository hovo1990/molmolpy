
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



import itertools

import hdbscan
import matplotlib
import matplotlib.cm as cm
import pandas as pd
#from bokeh.core.compat.mplexporter.utils import get_line_style
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

from msmbuilder.preprocessing import RobustScaler

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

matplotlib.rcParams.update({'font.size': 12})
# matplotlib.style.use('ggplot')
sns.set(style="white", context='paper')


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)


class FeatAnalysisObject(object):
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
    def __init__(self, featurized_file,
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

        self.feat_trajectory_file = featurized_file

        self.range_n_clusters = list(range(2, k_clust + 1))

        # This part is for checking which methods were called
        self.called_feat_pca_analysis = False




        self.called_find_max_cluster_centroid = False


        self.called_find_clusters_centroid = False

        self.called_find_clusters_hbond = False

        # DATA ANALYSIS OBJECTS



        self.md_feat_analysis_data = {}


        self.md_pre_feat_analysis_data = {}

        self.cluster_selection_analysis_data = {}

        self.cluster_selection_color_data = {}

        self.k_clust = k_clust


    @hlp.timeit
    def feat_full_load(self):
        # TODO How to prepare for mdtraj
        # mdconvert -t md_0_1.pdb -s 4 -o md_traj.xtc md_0_3_clear.xtc

        print('Featurized file load has been called\n')
        print('-------------------------------\n')
        self.parse_table = pd.read_hdf(self.feat_trajectory_file)

        self.sim_nums = self.parse_table['SimNum'].unique().tolist()

        self.pca_data = self.parse_table.iloc[:,
                     6:]

        self.sim_seqs = []

        for i in self.sim_nums:
            temp_table = self.parse_table.loc[self.parse_table.SimNum == i]
            diheds = temp_table.iloc[:,
                     6:]  # first two columns of data frame with all rows.iloc[:, 0:2] # first two columns of data frame with all rows
            diheds_nd = diheds.values
            self.sim_seqs.append(diheds_nd)

        self.feat_loaded = True
        self.feat_deleted = False
        print("Full feautirized file loaded successfully")
        print('-----------------------------------\n')


    @hlp.timeit
    def scale_data(self, scaler='Robust'):

        print('Scale featurized data been called\n')
        print('-------------------------------\n')
        from msmbuilder.preprocessing import RobustScaler

        if scaler=='Robust':
            scaler = RobustScaler()


        self.scaled_data= scaler.fit_transform(self.sim_seqs)

        print('scaled ', self.scaled_data[0].shape)
        # #
        print("Scaling feautirized data successfully")
        print('-----------------------------------\n')

    @hlp.timeit
    def pca_cum_variance_analysis(self,  show_plot=False, custom_dpi=600,
                                     percentage=70, number_of_components=20):

        self.called_feat_pca_analysis = True
        print('PCA Cumulative Variance analysis has been called\n')
        print('-------------------------------\n')


        sns.set(style="ticks", context='paper')
        # fig = plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        sns.set(font_scale=1)



        if number_of_components is not None:
            pca1 = PCA(n_components=number_of_components)
        else:
            pca1 = PCA(n_components=len(self.selection))

        TEST = 1

        # self.pca_transformed_data = pca1.fit_transform(self.scaled_data)
        self.pca_transformed_data = pca1.fit_transform(self.pca_data)


        # The amount of variance that each PC explains
        var = pca1.explained_variance_ratio_
        print('Explained variance ratio: ', var)

        self.md_pre_feat_analysis_data = {'varExplainedRatio': pca1.explained_variance_ratio_,
                                                               'varExplained': pca1.explained_variance_,
                                                               'mean': pca1.mean_,
                                                               }


        # Cumulative Variance explains
        var1 = np.cumsum(np.round(pca1.explained_variance_ratio_, decimals=4) * 100)

        print("Cumulative Variance explains ", var1)

        # plt.plot(var)
        plt.plot(var1)
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Proportion of Variance Explained")

        fig.savefig(self.simulation_name  + 'PCA_cumsum_analysis_' + '.png',
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
        return self.number_pca
        print("PCA transformation finished successfully")
        print('-----------------------------------\n')

