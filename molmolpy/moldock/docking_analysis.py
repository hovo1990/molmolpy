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

import matplotlib
import seaborn as sns
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import argrelmax
from scipy.signal import argrelmin

from sklearn import mixture

import multiprocessing

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import extra_tools

from molmolpy.utils import helper as hlp

# matplotlib.style.use('ggplot')
sns.set(style="darkgrid", context='paper')


class DockingAnalysisObject(object):
    """



        # Ultra docking analysis
        >>> QRC_dock_analysis.cluster_centroids_and_reshapes()
        >>>
        >>> QRC_dock_analysis.override_clust_num(9, selection='reshape')
        >>>
        >>> QRC_dock_analysis.show_ultra_docking_pca_results(lang='eng', down_region=30)
        >>>
        >>> QRC_dock_analysis.show_ultra_docking_centroid_pca_results(lang='eng', down_region=5)
        >>> QRC_dock_analysis.show_ultra_docking_reshape_pca_results(lang='eng', down_region=5)
        >>>
        >>> QRC_dock_analysis.show_all_custom_cluster_analysis_plots('centroid')
        >>> QRC_dock_analysis.show_all_custom_cluster_analysis_plots('reshape')




        >>> # Exhaustiveness example
        >>> QRC_exhaust_dock_analysis.cluster_centroids_and_reshapes()
        >>>
        >>> QRC_exhaust_dock_analysis.override_clust_num(4, 'centroid')
        >>>
        >>>
        >>>
        >>> QRC_exhaust_dock_analysis.show_ultra_docking_pca_results(lang='eng', down_region=30)
        >>>
        >>> QRC_exhaust_dock_analysis.show_ultra_docking_centroid_pca_results(lang='eng', down_region=5)
        >>> QRC_exhaust_dock_analysis.show_ultra_docking_exhaust_analysis(lang='eng', down_region=5)
        >>> QRC_exhaust_dock_analysis.show_ultra_docking_reshape_pca_results(lang='eng', down_region=5)
        >>>
        >>> QRC_exhaust_dock_analysis.export_custom_cluster_models('centroid')
        >>>
        >>> #
        >>> # QRC_exhaust_dock_analysis.cluster_analysis_pca()
        >>> #
        >>> #
        >>> # QRC_exhaust_dock_analysis.pca_exhaust_plot()
        >>> # QRC_exhaust_dock_analysis.exhaust_n_clusters_plot()
        >>> # QRC_exhaust_dock_analysis.show_all_exhaustive_cluster_analysis_plots()
        >>> # # HSL_exxhaust_dock_analysis.pca_exhaust_plot_plus_hist()
        >>> # # HSL_exxhaust_dock_analysis.pca_exhaust_plot_plus_bar()
        >>> #
        >>> # QRC_exhaust_dock_analysis.show_silhouette_analysis_pca_best()
        >>> # QRC_exhaust_dock_analysis.show_all_cluster_analysis_plots()
        >>>
        >>>
        >>> # # TODO Visualize pymol
        >>> receptor_file = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology/dock_100ns.pdb'
        >>> QRC_exhaust_dock_analysis.add_receptor_for_viz(receptor_file)
        >>>
        >>> # Centroid_analysis.generate_pymol_viz('centroid')
        >>>
        >>> QRC_exhaust_dock_analysis.generate_exhaust_pymol_viz_thread(type='data')
        >>> QRC_exhaust_dock_analysis.generate_exhaust_pymol_viz_thread(type='cluster')



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

    def __init__(self, docking_samples_object,
                 calc_type='mean',
                 load_way='molmolpy',
                 molname='Unknown',
                 receptor_name='Unknown',
                 k_clust=15):
        '''
        #   zdes skin svou papku
            QRC_folder = '/media/Work/MEGA/Programming/LasR_QRC/QRC_dock/MonB/'
            QRC_docking_samples = docking_samples_object.DockSamplesObject(QRC_folder, molname='Quercetin', receptor_name='LasR',
                                                                           color='r', z_order=30, size=100, marker='v')

            # QRC_dock_analysis = docking_analysis.DockingAnalysisObject(QRC_docking_samples, calc_type='mean')
            # QRC_dock_analysis = docking_analysis.DockingAnalysisObject(QRC_docking_samples, calc_type='reshape')


            # QRC_dock_analysis.pca_analysis_reshape()



            QRC_dock_analysis = docking_analysis.DockingAnalysisObject(QRC_docking_samples, calc_type='both')
            # QRC_dock_analysis.plot_both(show_plot=False)


            QRC_dock_analysis.cluster_centroids_and_reshapes()

            QRC_dock_analysis.override_clust_num(9, selection='reshape')

            QRC_dock_analysis.show_ultra_docking_pca_results(lang='eng', down_region=30)

            QRC_dock_analysis.show_ultra_docking_centroid_pca_results(lang='eng', down_region=5)
            QRC_dock_analysis.show_ultra_docking_reshape_pca_results(lang='eng', down_region=5)

            QRC_dock_analysis.show_all_custom_cluster_analysis_plots('centroid')
            QRC_dock_analysis.show_all_custom_cluster_analysis_plots('reshape')



            QRC_dock_analysis.show_cluster_histograms_ultra('centroid')
            QRC_dock_analysis.show_cluster_histograms_ultra('reshape')



            # Temporary for test
            QRC_dock_analysis.export_custom_cluster_models('centroid')
            QRC_dock_analysis.export_custom_cluster_models('reshape')
            #
            #
            # Centroid_analysis = cluster_traj_analyzer.ClusterTrajAnalyzerObject(QRC_dock_analysis)
            # # #
            # # Centroid_analysis.extract_centroids_auto_custom()
            # # Centroid_analysis.extract_centroids_auto()
            # #
            correct_topology = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology/'

            QRC_topology = correct_topology + 'QRC_NEW.pdb'

            working_folder = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology'
            # #
            # # # TODO fix this part
            # # # TODO this buggy
            # # # Centroid_analysis.simplified_cluster_extraction()
            # #
            # Centroid_analysis.simplified_cluster_extraction_mdtraj_custom()
            # #
            # #
            # #
            # # # TODO for fixing centroid Hydrogen molecules and recovering topology
            Centroid_analysis.reconstruct_centroids_topology_custom(working_folder_path=working_folder,
                                                             correct_topology_info=QRC_topology,
                                                             parallel=False)

            receptor_file = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology/dock_100ns.pdb'
            Centroid_analysis.add_receptor_for_viz(receptor_file)

            Centroid_analysis.extract_centroids_auto()



        :param docking_samples_object:
        :param calc_type:
        :param load_way:
        :param molname:
        :param receptor_name:
        :param k_clust:
        '''

        self.docking_samples_object = docking_samples_object

        self.sample_files = docking_samples_object.sample_files

        # Use samles data for pymol
        self.samples_data = docking_samples_object.samples_data

        self.info_type = self.docking_samples_object.info_type

        self.molecule_name = self.docking_samples_object.molecule_name
        self.receptor_name = self.docking_samples_object.receptor_name
        self.color_plot = self.docking_samples_object.color
        self.size = self.docking_samples_object.size
        self.marker = self.docking_samples_object.marker
        self.z_order = self.docking_samples_object.z_order

        self.analysis_structure = self.docking_samples_object.analysis_structure__
        self.analysis_centroid_structure__ = self.docking_samples_object.analysis_centroid_structure__
        self.concatenated_analysis__ = self.docking_samples_object.concatenated_analysis__

        self.analysis_reshape_structure__ = self.docking_samples_object.analysis_reshape_structure__

        self.data_cols = self.docking_samples_object.data_cols
        self.equiv_models = self.docking_samples_object.equivalent_models__

        # histogram plot test

        # self.simulation_name = 'docking_' + self.receptor_name + '_' + self.molecule_name

        self.simulation_name = self.docking_samples_object.simulation_name + '_' + self.info_type

        if calc_type == 'mean':
            self.data_for_analysis = self.analysis_structure[['X', 'Y', 'Z']]
            self.pca_data = self.pca_analysis()

            self.analysis_type = 'mean'

        elif calc_type == 'reshape' or calc_type == 'all':
            # This is reshaped structure
            self.data_for_analysis = self.analysis_reshape_structure__[self.data_cols]
            self.pca_data = self.pca_analysis_reshape()

            self.analysis_type = 'reshape'

        elif calc_type == 'both':
            # self.data_for_analysis_coord = self.analysis_structure[['X', 'Y', 'Z']]
            self.data_for_analysis_centroid = self.analysis_structure[['X', 'Y', 'Z']]
            self.pca_data_centroid = self.pca_analysis(custom=self.data_for_analysis_centroid)
            self.pca_data_centroid_scaled = self.pca_analysis(scale_features=True,
                                                              custom=self.data_for_analysis_centroid)

            self.data_for_analysis_reshape = self.analysis_reshape_structure__[self.data_cols]
            self.pca_data_reshape = self.pca_analysis(custom=self.data_for_analysis_reshape)
            self.pca_data_reshape_scaled = self.pca_analysis(scale_features=True, custom=self.data_for_analysis_reshape)

            self.data_for_analysis_all = self.concatenated_analysis__[self.data_cols]

            self.pca_data_all = self.pca_analysis(custom=self.data_for_analysis_all)
            self.pca_data_all_scaled = self.pca_analysis(scale_features=True, custom=self.data_for_analysis_all)
            test = 1

            self.analysis_type = 'ultra'

        else:
            print('Error no such option try centroid reshape or both')
            sys.exit(1)

        # self.data_for_analysis = self.analysis_structure[['X', 'Y', 'Z']]

        # self.plotHist()

        # self.pca_data = self.pca_analysis()



        # self.iso_analysis()

        self.clusters_info = {}

        self.exhaust_clusters_info = {}
        self.range_n_clusters = list(range(2, k_clust + 1))

        # TODO Save extract files list so its easier for mdtraj analysis in later part
        self.save_extract_files_list = {}

        self.clust_percentage_data = {'centroid':{}, 'reshape':{}}

        # self.silhouette_analysis_pca(show_plots=True)
        # TODO double but why
        # self.cluster_analysis_pca()

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()

        # Then need to extract conformations

        # TODO this is the part of cluster data extraction
        # self.clust_num = self.select_number_of_clusters()
        # self.cluster_models = self.collect_cluster_info()

        data = 2
        # self.calinski_graph_pca()
        # # self.hdbscan_pca()


        # self.MeanShift()
        # self.VBGMM()

    def combine_dataframes(self):
        # self.data_for_analysis_coord
        # self.data_for_analysis_all
        test = 1

        # this doesnt work
        # T1 = pd.merge(self.data_for_analysis_coord, self.data_for_analysis_all, on=self.data_for_analysis_coord.index, how='outer')

        test = 1

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

    @hlp.timeit
    def exhaust_box_plot(self):
        import seaborn as sns
        sns.set(style="ticks")

        ax = sns.boxplot(x="SampleInfoNum", y="BindingEnergy", data=self.analysis_reshape_structure__, palette="PRGn")
        ax.set(xlabel='Exhaustiveness', ylabel='Binding Affinity(kcal/mol)')
        sns.despine(offset=10, trim=True)
        plt.show()

    @hlp.timeit
    def exhaust_dist_plot(self):
        import seaborn as sns
        sns.set(style="ticks")

        ax = sns.distplot(self.analysis_reshape_structure__["BindingEnergy"])
        # ax.set(xlabel='Exhaustiveness', ylabel='Binding Affinity(kcal/mol)')
        # sns.despine(offset=10, trim=True)
        plt.show()

    @hlp.timeit
    def exhaust_joint_plot(self):
        import seaborn as sns
        sns.set(style="white")

        ax = sns.jointplot(x="SampleInfoNum", y="BindingEnergy", data=self.analysis_reshape_structure__, kind='hex',
                           color='k')
        # ax.set(xlabel='Exhaustiveness', ylabel='Binding Affinity(kcal/mol)')
        # sns.despine(offset=10, trim=True)
        plt.show()

    @hlp.timeit
    def exhaust_rmsd(self):
        import seaborn as sns

        import mdtraj as md

    @hlp.timeit
    def pca_analysis_exhaust(self):
        scaleFeatures = False

        df = self.data_for_analysis
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(df)
        T = pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T)

        T.columns = ['component']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T

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
        T = pca.transform(df)

        # ax = self.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
        T = pd.DataFrame(T)

        T.columns = ['component1', 'component2']
        # T.plot.scatter(x='component1', y='component2', marker='o', s=300, alpha=0.75)  # , ax=ax)
        # plt.show()
        return T

    @hlp.timeit
    def pca_analysis_reshape(self, custom=None):
        scaleFeatures = False

        if custom is None:
            df = self.data_for_analysis
        else:
            df = custom
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
    def pseudoF_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.book_pseudoF_pca
        criteria_name = 'pseudoF score'
        score_text = 'Objects with a high pseudo-F score value are considered well clustered'
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
    def book_dbi_graph_pca(self):
        cluster_range = self.range_n_clusters
        score = self.book_dbi_pca
        criteria_name = 'Davis-Bouldain Index'
        score_text = 'The optimal clustering solution has the smallest Davies-Bouldin index value.'
        plot_tools.plot_cluster_analysis(cluster_range, score, criteria_name, score_text)



    # TODO adapt for ultra docking
    @hlp.timeit
    def show_cluster_and_scores_pca_best(self, custom_dpi=600, show_plot=False, lang='rus', trasparent_alpha=True,
                                         order=1):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        n_clusters = self.clust_num

        cluster_labels = self.clusters_info[n_clusters]['labels']
        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        X = self.pca_data

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))
        # fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)

        #


        # Create a subplot with 1 row and 2 columns
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(10, 7)

        plot_list = []

        # GOOD Starting point
        # X = [(1, 2, 1), (4, 2, 2), (4, 2, 4), (4, 2, 6), (4,2,8) ]


        if order == 1:
            plot_order = [(1, 2, 1), (4, 2, 2), (4, 2, 4), (4, 2, 6), (4, 2, 8)]
        elif order == 2:
            plot_order = [(1, 4, 1), (2, 4, 2), (2, 4, 4), (2, 4, 6), (2, 4, 8)]
        elif order == 3:
            # Finally correct Order
            plot_order = [(1, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 5), (2, 3, 6)]
        for nrows, ncols, plot_number in plot_order:
            ax = plt.subplot(nrows, ncols, plot_number)
            plot_list.append(ax)

        test = 1

        # plt.show()

        # fig.tight_layout()
        # sns.set(font_scale=2)

        general_font_size = 20

        if lang == 'eng':

            plot1_title = "The visualization of the clustered data."
            plot1_xlabel = "Feature space for the 1st feature"
            plot1_ylabel = "Feature space for the 2nd feature"

            plot1_xlabel = "PC1"
            plot1_ylabel = "PC2"

            # plt.set_xlabel("PC1 (Å)")
            # plt.set_ylabel("PC2 (Å)")

            plot_whole_titile = "Silhouette analysis for KMeans clustering on docking data with n_clusters = %d" % n_clusters
        else:

            plot1_title = "Визуализация данных кластеризации"
            plot1_xlabel = "Пространство для 1-го признака"
            plot1_ylabel = " Пространство для 2-го признака"

            plot_whole_titile = "Силуэтный анализ на основе данных докинга, используя алгоритм k-средних = %d" % n_clusters

        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot

        colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)
        self.colors_ = colors

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        plot_list[0].scatter(X['component1'], X['component2'], marker='.', s=3000, lw=0, alpha=0.7,
                             c=colors)

        # Labeling the clusters

        # Draw white circles at cluster centers
        plot_list[0].scatter(centers[:, 0], centers[:, 1],
                             marker='o', c="white", alpha=1, s=1200)

        for i, c in enumerate(centers):
            plot_list[0].scatter(c[0], c[1], marker='$%d$' % int(i + 1), alpha=1, s=1000)

        # plot_list[0].set_title(plot1_title, fontweight='bold', fontsize=general_font_size )
        plot_list[0].set_xlabel(plot1_xlabel, fontweight='bold', fontsize=general_font_size)
        plot_list[0].set_ylabel(plot1_ylabel, fontweight='bold', fontsize=general_font_size)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=14, fontweight='bold')
        #
        # plt.show()

        # fig.savefig(self.simulation_name + '_best_PCA_docking_analysis' + '_' + lang + '.png', dpi=custom_dpi,
        #             transparent=trasparent_alpha)


        cluster_range = self.range_n_clusters
        score = self.book_dbi_pca
        score_text = ''
        if lang == 'rus':
            criteria_name = 'Индекс Дэвиса-Болдина'
            num_cluster = 'кол-во кластеров'
            # if order != 1 or order !=3:
            #     score_text = 'Оптимальным решением кластеризации\n будет минимальное значение индекса Дэвиса-Болдина'
        else:
            criteria_name = 'Davis-Bouldain Index'
            if order != 1:
                score_text = 'The optimal clustering solution\n has the smallest Davies-Bouldin index value.'
        plot_list[1].scatter(cluster_range, score, marker='o', c='b', s=200)
        plot_list[1].plot(cluster_range, score, ':k', linewidth=3.0)

        plot_list[1].set_xlim(cluster_range[0], cluster_range[-1])

        plot_list[1].set_title(score_text)
        plot_list[1].set_xlabel(num_cluster, fontweight='bold', fontsize=general_font_size)
        plot_list[1].set_ylabel(criteria_name, fontweight='bold', fontsize=general_font_size)

        cluster_range = self.range_n_clusters
        score = self.dunn_pca
        score_text = ''
        if lang == 'rus':
            criteria_name = 'Индекс Данна'
            num_cluster = 'кол-во кластеров'
            # if order != 1 or order != 3:
            #     score_text = 'Оптимальным решением кластеризации\n будет максимальное значение индекса Данна'
        else:
            criteria_name = "Dunn's Index"
            if order != 1:
                score_text = "Maximum value of the index\n represents the right partitioning given the index"
        plot_list[2].scatter(cluster_range, score, marker='o', c='b', s=200)
        plot_list[2].plot(cluster_range, score, ':k', linewidth=3.0)

        plot_list[2].set_xlim(cluster_range[0], cluster_range[-1])
        plot_list[2].set_title(score_text)
        plot_list[2].set_xlabel(num_cluster, fontweight='bold', fontsize=general_font_size)
        plot_list[2].set_ylabel(criteria_name, fontweight='bold', fontsize=general_font_size)

        cluster_range = self.range_n_clusters
        score = self.sil_pca
        score_text = ''
        if lang == 'rus':
            criteria_name = 'Индекс оценки силуэта'
            num_cluster = 'кол-во кластеров'
            # if order != 1 or order != 3:
            #     score_text = 'Объекты с высоким значением силуэта\n считаются хорошо сгруппированными.'
        else:
            criteria_name = 'Mean Silhouette Coefficient for all samples'
            if order != 1:
                score_text = 'Objects with a high silhouette value\n are considered well clustered'
        plot_list[3].scatter(cluster_range, score, marker='o', c='b', s=200)
        plot_list[3].plot(cluster_range, score, ':k', linewidth=3.0)

        plot_list[3].set_xlim(cluster_range[0], cluster_range[-1])
        plot_list[3].set_title(score_text)
        plot_list[3].set_xlabel(num_cluster, fontweight='bold', fontsize=general_font_size)
        plot_list[3].set_ylabel(criteria_name, fontweight='bold', fontsize=general_font_size)

        cluster_range = self.range_n_clusters
        score = self.calinski_pca
        score_text = ''
        if lang == 'rus':
            criteria_name = 'Индекс Калински-Харабаза'
            num_cluster = 'кол-во кластеров'
            # if order != 1 or order != 3:
            #     score_text = 'Объекты с высоким значением оценки Калински-Харабаз\n считаются хорошо сгруппированными'
        else:
            criteria_name = 'Calinski-Harabaz score'
            if order != 1:
                score_text = 'Objects with a high Calinski-Harabaz score\n value are considered well clustered'
        plot_list[4].scatter(cluster_range, score, marker='o', c='b', s=200)
        plot_list[4].plot(cluster_range, score, ':k', linewidth=3.0)
        plot_list[4].set_xlim(cluster_range[0], cluster_range[-1])
        plot_list[4].set_title(score_text)
        plot_list[4].set_xlabel(num_cluster, fontweight='bold', fontsize=general_font_size)
        plot_list[4].set_ylabel(criteria_name, fontweight='bold', fontsize=general_font_size)

        plt.subplots_adjust(top=0.90)
        if lang == 'rus':
            suptitle_text = 'Определение оптимального количества кластеров'
        else:
            suptitle_text = "Determination of optimal number of Clusters"

        # plt.suptitle((suptitle_text),
        #              fontsize=14, fontweight='bold')

        fig.tight_layout()

        fig.savefig(self.simulation_name + '_{0}_'.format(lang) + 'order:{0}_cluster_determination.png'.format(order),
                    dpi=custom_dpi,
                    transparent=trasparent_alpha)

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_all_custom_cluster_analysis_plots(self, type, custom_dpi=1200, show_plot=False, lang='eng',
                                               trasparent_alpha=False):
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)

        data_to_use = self.ultra_clustering[type]

        n_clusters = data_to_use['clustNum']

        cluster_labels = data_to_use['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old version
        centers = data_to_use['dataClustering']['clusterInfo'][n_clusters]['centers']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                     2)  # sharex='col', sharey='row') TODO this can be used for shared columns

        fig.set_size_inches(plot_tools.cm2inch(17.7, 17.7))

        sns.set(font_scale=1)

        cluster_range = self.range_n_clusters
        score = data_to_use['dataClustering']['clusterAnalysisInfo']['dbiBook']
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
        score = data_to_use['dataClustering']['clusterAnalysisInfo']['dunn']
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
        score = data_to_use['dataClustering']['clusterAnalysisInfo']['silhouette']
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
        score = data_to_use['dataClustering']['clusterAnalysisInfo']['calinski']
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

        if lang == 'rus':
            suptitle_text = 'Определение оптимального количества кластеров'
        else:
            suptitle_text = "Determination of optimal number of Clusters"

        # fig.tight_layout()

        plt.suptitle((suptitle_text),
                     fontsize=14, fontweight='bold')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.90)

        fig.savefig(self.simulation_name + '_{0}_type_{1}'.format(lang, type) + '_cluster_determination.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_all_cluster_analysis_plots(self, custom_dpi=1200, show_plot=False, lang='eng', trasparent_alpha=False):
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                     2)  # sharex='col', sharey='row') TODO this can be used for shared columns

        fig.set_size_inches(16, 10)

        sns.set(font_scale=1)

        cluster_range = self.range_n_clusters
        score = self.book_dbi_pca
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
        score = self.dunn_pca
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
        score = self.sil_pca
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
        score = self.calinski_pca
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

        plt.subplots_adjust(top=0.90)
        if lang == 'rus':
            suptitle_text = 'Определение оптимального количества кластеров'
        else:
            suptitle_text = "Determination of optimal number of Clusters"

        # fig.tight_layout()

        plt.suptitle((suptitle_text),
                     fontsize=14, fontweight='bold')

        fig.savefig(self.simulation_name + '_{0}_'.format(lang) + '_cluster_determination.png', dpi=custom_dpi,
                    transparent=trasparent_alpha)

        if show_plot is True:
            plt.show()

    def sns_test(self):
        import seaborn as sns
        sns.set(style="ticks")

        # Load the example dataset for Anscombe's quartet
        df = sns.load_dataset("anscombe")

        # Show the results of a linear regression within each dataset
        sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
                   col_wrap=2, ci=None, palette="muted", size=4,
                   scatter_kws={"s": 50, "alpha": 1})

    @hlp.timeit
    def select_number_of_clusters(self, cluster_info):
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

    def collect_cluster_info_v2(self, clusters_info, clust_num, pca, original_data):
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

    def collect_cluster_info(self):
        data = self.clusters_info[self.clust_num]
        print(data)

        labels = data['labels']
        # Make more flexible whether pca_data or not
        pca_data = self.pca_data
        original_data = self.analysis_reshape_structure__  # self.pca_data

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


        # TODO need to find a way to extract models correctrly

    def export_custom_cluster_models(self, type, save_pickle_auto=False, folder_name='cluster_traj'):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''

        print('Start export custom cluster models for {0}\n'.format(type))
        print('----------------------------------------------')

        data_to_use = self.ultra_clustering[type]


        if data_to_use['overrideClustNum'] is not None:
            n_clusters = data_to_use['overrideClustNum']
        else:
            n_clusters = data_to_use['clustNum']

        # n_clusters = data_to_use['clustNum']

        data_clustering = data_to_use['dataClustering']

        colors = data_clustering['colors']
        rgb_colors = data_clustering['rgbColors']

        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name + \
                         '_' + self.info_type
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)
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

        for clust in clust_numbers:
            cluster = clust_numbers[clust]
            # print(cluster)
            # Make nomenclatur similar to plots
            filename = "ligBindTraj_type_{0}_{1}_{2}.pdb".format(type_to_write, clust + 1, self.info_type)
            filename_to_write = save_directory + os.sep + filename
            file_to_write = open(filename_to_write, 'w')


            # TODO this needs a new approach
            percentage = self.clust_percentage_data[type][int(clust)]

            self.save_extract_files_list[type].update({clust: {'relativePath': filename_to_write,
                                                               'filename': filename,
                                                               'colors': colors[clust],
                                                               'rgbColors': rgb_colors[clust],
                                                               'currModels': cluster,
                                                               'key': clust,
                                                               'percentage':percentage}})

            res_num = 1
            for model in cluster:
                # curr_model = model
                curr_df = model['molDetail']['dataframe']
                pdb_tools.write_lig(curr_df, res_num, file_to_write)
                # self.write_model_to_file(curr_model, res_num, file_to_write)
                res_num += 1
                file_to_write.write('ENDMDL\n')
            file_to_write.close()

        test = 1

    # TODO need to find a way to extract models correctrly
    def export_cluster_models(self, save_pickle_auto=False, folder_name='cluster_traj'):
        '''
        Save cluster data to pdb files in cluster_traj directory
        :return:
        '''
        save_directory = '.' + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + folder_name
        self.save_cluster_models_dir = save_directory
        folder_utils.create_folder(save_directory)
        if save_pickle_auto is True:
            self.save_analysed_data(save_directory + os.sep + '{0}_cluster_data.pickle'.format(self.molecule_name))
        clust_numbers = self.cluster_models
        for clust in clust_numbers:
            cluster = clust_numbers[clust]
            # print(cluster)
            # Make nomenclatur similar to plots
            file_to_write = open(save_directory + os.sep + "ligBindTraj_{0}.pdb".format(clust + 1), 'w')
            res_num = 1
            for model in cluster:
                # curr_model = model
                curr_df = model['molDetail']['dataframe']
                pdb_tools.write_lig(curr_df, res_num, file_to_write)
                # self.write_model_to_file(curr_model, res_num, file_to_write)
                res_num += 1
                file_to_write.write('ENDMDL\n')
            file_to_write.close()

    def save_analysed_data(self, filename=None):
        '''
        :param filename: Saves clustered data to pickle file
        :return:
        '''
        # import json
        # with open(filename, 'w') as outfile:
        #     json.dump(self.cluster_models, outfile)
        import pickle
        if filename is None:
            filename = self.molecule_name + '_pickleFile.pickle'
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        pickle.dump(self, open(filename, "wb"))

    # TODO should I add json saving of information or not?
    def load_analysed_data(self, filename):
        '''

        :param filename: load pickle file
        :return:
        '''
        self.analysed_data = pickle.load(open(filename, "rb"))
        print('test')

    # TODO create another function that shows only the best plot for kmeans
    @hlp.timeit
    def show_silhouette_analysis_pca_best(self, custom_dpi=1200, show_plot=False, lang='eng', trasparent_alpha=False):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        # import seaborn as sns
        #
        # sns.set(font_scale=2)

        n_clusters = self.clust_num

        cluster_labels = self.clusters_info[n_clusters]['labels']
        sample_silhouette_values = self.clusters_info[n_clusters]['silhouette_values']
        silhouette_avg = self.clusters_info[n_clusters]['silhouette']

        centers = self.clusters_info[n_clusters]['centers']

        X = self.pca_data

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 7)
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

        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        y_lower = 10

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)
        self.colors_ = colors

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            cmap = cm.get_cmap("Spectral")
            color = cmap(float(i) / n_clusters)
            
            #old version
            #color = cm.spectral(float(i) / n_clusters)
            # ax1.fill_betweenx(np.arange(y_lower, y_upper),
            #                   0, ith_cluster_silhouette_values,
            #                   facecolor=color, edgecolor=color, alpha=0.7)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=colors[i], edgecolor=colors[i], alpha=0.98)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(plot1_title)
        ax1.set_xlabel(plot1_xlabel)
        ax1.set_ylabel(plot1_ylabel)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)

        # 2nd Plot showing the actual clusters formed
        colors = converters.convert_to_colordata(cluster_labels, colors)
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #
        # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)

        self.cluster_colors = colors

        ax2.scatter(X['component1'], X['component2'], marker='.', s=250, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % int(i + 1), alpha=1, s=200)

        ax2.set_title(plot2_title)
        ax2.set_xlabel(plot2_xlabel)
        ax2.set_ylabel(plot2_ylabel)

        plot_tools.change_ax_plot_font_size(ax1, 12)
        plot_tools.change_ax_plot_font_size(ax2, 12)

        plt.suptitle(plot_whole_titile,
                     fontsize=16, fontweight='bold')

        # fig.tight_layout()

        fig.savefig(self.simulation_name + '_best_PCA_docking_analysis' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha)

        if show_plot is True:
            plt.show()



            # TODO create another function that shows only the best plot for kmeans

    @hlp.timeit
    def plot_both(self, custom_dpi=1200, show_plot=False, lang='eng',
                  trasparent_alpha=False):

        # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
        #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
        #                                         'labels': cluster_labels, 'centers': centers,
        #                                         'silhouette_values': sample_silhouette_values}})

        # import seaborn as sns
        #
        sns.set(style="white", context='paper')
        sns.set(font_scale=1)

        #         self.data_for_analysis_all

        test = 1
        # print(self.concatenated_analysis__[int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__)+ 1])

        centroid_indexes = self.concatenated_analysis__[
            self.concatenated_analysis__['Type'] == 'centroid'].index.tolist()
        test1 = len(self.concatenated_analysis__) / 2

        # temp_centroid = self.pca_data_all[centroid_indexes[0]:centroid_indexes[-1] + 1]
        temp_centroid = self.pca_data_all[
                        int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]

        reshape_indexes = self.concatenated_analysis__[self.concatenated_analysis__['Type'] == 'reshape'].index.tolist()
        temp_reshape = self.pca_data_all[reshape_indexes[0]:reshape_indexes[-1] + 1]

        test = 1

        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots()
        # fig.set_size_inches(10, 7)
        fig.set_size_inches(plot_tools.cm2inch(17.7, 12))

        if lang == 'eng':

            # plot2_ylabel = "Feature space for the 2nd feature"

            plot2_xlabel = "PC1"
            plot2_ylabel = "PC2"

            # plt.set_xlabel("PC1 (Å)")
            # plt.set_ylabel("PC2 (Å)")

            # plot_whole_titile = "Silhouette analysis for KMeans clustering on docking data with n_clusters = %d" % n_clusters
            plot_whole_titile = "Silhouette analysis for k-means clustering on docking data"
        else:

            plot2_title = "Визуализация данных кластеризации"
            plot2_xlabel = "Пространство для 1-го признака"
            plot2_ylabel = " Пространство для 2-го признака"

            plot_whole_titile = "Силуэтный анализ на основе данных докинга, используя алгоритм k-средних = %d" % n_clusters

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.7, dark=0, light=0.85)
        # self.colors_ = colors
        #
        # # 2nd Plot showing the actual clusters formed
        # colors = converters.convert_to_colordata(cluster_labels, colors)
        # # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        # #
        # #
        # # my_cmap = sns.cubehelix_palette(n_colors=n_clusters)
        #
        # self.cluster_colors = colors




        label = 'Centroid'
        size = 1000
        object = ax.scatter(temp_centroid['component1'], temp_centroid['component2'], marker='o',
                            edgecolors='k', s=size,
                            lw=0.4,
                            c='b', label=label)
        object.set_zorder(1)

        label = 'Reshape'
        size = 80
        object = ax.scatter(temp_reshape['component1'], temp_reshape['component2'], marker='o',
                            edgecolors='k', s=size,
                            lw=0.4,
                            c='r', label=label)
        object.set_zorder(2)

        # Labeling the clusters

        # ax.set_title(plot2_title)
        # ax.set_xlabel(plot2_xlabel)
        # ax.set_ylabel(plot2_ylabel)
        #
        # plot_tools.change_ax_plot_font_size(ax1, 12)
        # plot_tools.change_ax_plot_font_size(ax, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')
        # leg = plt.legend(loc='best')
        leg = plt.legend(loc=9, bbox_to_anchor=(1.1, 0.5), labelspacing=2.0, ncol=1)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)

        fig.tight_layout()
        sns.despine()

        fig.savefig(self.simulation_name + '_centroid_reshape_' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()


            # TODO create another function that shows only the best plot for kmeans

    @hlp.timeit
    def show_ultra_docking_pca_results(self, custom_dpi=1200, show_plot=False, lang='eng',
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



        object_centroid = ax.scatter(X['component1'], X['component2'], marker='.', s=1000, lw=0, alpha=0.7,
                                     c=colors)
        object_centroid.set_zorder(1)

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
                                             marker='o', c="white", alpha=1, s=400)
        object_centroid_circles.set_zorder(2)

        for i, c in enumerate(centers):
            roman_number = extra_tools.write_roman(int(i + 1))
            object_centroid_circles_text = ax.scatter(c[0], c[1] - down, marker='$%s$' % roman_number, alpha=1, s=350,
                                                      c='g')
            object_centroid_circles_text.set_zorder(3)

        # Reshape part
        reshape_data = self.ultra_clustering['reshape']

        if reshape_data['overrideClustNum'] is not None:
            n_clusters = reshape_data['overrideClustNum']
        else:
            n_clusters = reshape_data['clustNum']

        cluster_labels = reshape_data['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old center
        centers = reshape_data['dataClustering']['clusterInfo'][n_clusters]['centers']

        X = reshape_data['dataPCA']

        # New version
        centers = calculate_cluster_center_docking(X, cluster_labels)

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        # colors = reshape_data['dataClustering']['colorData']
        # self.colors_ = colors
        #
        # self.cluster_colors = reshape_data['dataClustering']['rgbColors']



        colors = reshape_data['dataClustering']['clusterInfo'][n_clusters]['colorData']
        self.colors_ = colors

        self.cluster_colors = reshape_data['dataClustering']['clusterInfo'][n_clusters]['rgbColors']

        object_reshape = ax.scatter(X['component1'], X['component2'], marker='.', s=400, lw=0, alpha=0.7,
                                    c=colors)
        object_reshape.set_zorder(4)

        # Labeling the clusters

        # Draw white circles at cluster centers
        object_reshape_circles = ax.scatter(centers[:, 0], centers[:, 1],
                                            marker='o', c="white", alpha=1, s=150)
        object_reshape_circles.set_zorder(5)

        for i, c in enumerate(centers):
            int_number = int(i + 1)
            object_reshape_circles_text = ax.scatter(c[0], c[1], marker='$%d$' % int_number, alpha=1, s=100, c='b')
            object_reshape_circles_text.set_zorder(6)

        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        fig.tight_layout()

        sns.set(style="white", context='paper', font_scale=1)

        fig.savefig(self.simulation_name + '_best_ultra_docking_PCA_analysis' + '_' + lang + '.png', dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_ultra_docking_exhaust_analysis(self, custom_dpi=1200, show_plot=False, lang='eng',
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

        X = centroid_data['dataPCA']

        # New Version
        centers = calculate_cluster_center_docking(X, cluster_labels)

        sampleInfoNum = centroid_data['dataOriginal']['SampleInfoNum']

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

        colors = centroid_data['dataClustering']['colorData']
        self.colors_ = colors

        cluster_colors = centroid_data['dataClustering']['colors']

        self.cluster_colors = centroid_data['dataClustering']['rgbColors']

        object_centroid = ax.scatter(X['component1'], X['component2'], marker='.', s=1000, lw=0, alpha=0.7,
                                     c=colors)
        object_centroid.set_zorder(1)

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
                                             marker='o', c="white", alpha=1, s=400)
        object_centroid_circles.set_zorder(2)

        unique_exhaust_values = {}
        unique_exhaust_values_list = []
        unique_exhaust_values_list_str = []

        whole_size = len(centroid_data['dataOriginal']['SampleInfoNum'])

        self.clust_percentage_data = {}

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            # clust_label =X[cluster_labels == i]
            exhaust_values = centroid_data['dataOriginal']['SampleInfoNum'][cluster_labels == i]

            exhaust_unique = exhaust_values.unique()

            unique_exhaust_values.update({str(i): exhaust_unique})

            unique_exhaust_values_list.append(sorted(exhaust_unique))

            curr_exhaust_text = ''
            for i_exhaust in sorted(exhaust_unique):
                curr_exhaust_text += '%s ' % i_exhaust

            clust_size = len(exhaust_values)

            percentage = (clust_size*100)/whole_size

            self.clust_percentage_data.update({int(i):percentage})

            curr_exhaust_text += '({0}%)'.format(percentage)
            unique_exhaust_values_list_str.append(curr_exhaust_text)
            test = 1

        test = 1

        object_centroid_circles_text_list = []
        for i_exhaust, c in enumerate(centers):
            roman_number = extra_tools.write_roman(int(i_exhaust + 1))
            object_centroid_circles_text = ax.scatter(c[0], c[1] - down, marker='$%s$' % roman_number, alpha=1,
                                                      s=350,
                                                      c=cluster_colors[i_exhaust])
            object_centroid_circles_text.set_zorder(3)

            object_centroid_circles_text_list.append(object_centroid_circles_text)

        test = 1

        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        test = 1

        lgd = ax.legend(tuple(object_centroid_circles_text_list),
                        tuple(unique_exhaust_values_list_str),
                        scatterpoints=1,
                        bbox_to_anchor=(0.5, -0.12),
                        loc=9,
                        ncol=1,
                        fontsize=12)
        ax.grid('on')

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        fig.tight_layout()

        sns.set(style="white", context='paper', font_scale=1)

        fig.savefig(self.simulation_name + '_best_ultra_docking_centroid_exhaust_analysis' + '_' + lang + '.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_ultra_cluster_quality_analysis_plots(self, data_type, custom_dpi=1200, show_plot=False, lang='eng', trasparent_alpha=False):
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
    def show_ultra_docking_centroid_pca_results(self,  custom_dpi=1200, show_plot=False, lang='eng',
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

        colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['colorData']
        self.colors_ = colors

        self.cluster_colors = centroid_data['dataClustering']['clusterInfo'][n_clusters]['rgbColors']

        object_centroid = ax.scatter(X['component1'], X['component2'], marker='.', s=1000, lw=0, alpha=0.7,
                                     c=colors)
        object_centroid.set_zorder(1)

        # Labeling the clusters

        down = down_region

        # Draw white circles at cluster centers
        object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
                                             marker='o', c="white", alpha=1, s=400)
        object_centroid_circles.set_zorder(2)


        #self.clust_percentage_data = {}

        whole_size = len(centroid_data['dataOriginal']['SampleInfoNum'])

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            # clust_label =X[cluster_labels == i]
            cluster_values = centroid_data['dataOriginal']['SampleInfoNum'][cluster_labels == i]

            # exhaust_unique = exhaust_values.unique()
            #
            # unique_exhaust_values.update({str(i): exhaust_unique})
            #
            # unique_exhaust_values_list.append(sorted(exhaust_unique))
            #
            # curr_exhaust_text = ''
            # for i_exhaust in sorted(exhaust_unique):
            #     curr_exhaust_text += '%s ' % i_exhaust

            clust_size = len(cluster_values)

            percentage = (clust_size*100)/whole_size

            self.clust_percentage_data['centroid'].update({int(i):percentage})

            # curr_exhaust_text += '({0}%)'.format(percentage)
            # unique_exhaust_values_list_str.append(curr_exhaust_text)
            # test = 1



        for i, c in enumerate(centers):
            roman_number = extra_tools.write_roman(int(i + 1))
            object_centroid_circles_text = ax.scatter(c[0], c[1] - down, marker='$%s$' % roman_number, alpha=1,
                                                      s=350,
                                                      c='g')
            object_centroid_circles_text.set_zorder(3)


        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        fig.tight_layout()

        sns.set(style="white", context='paper', font_scale=1)

        fig.savefig(self.simulation_name + '_best_ultra_docking_centroid_PCA_analysis' + '_' + lang + '.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()

    @hlp.timeit
    def show_ultra_docking_reshape_pca_results(self, custom_dpi=1200, show_plot=False, lang='eng',
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

        # colors = centroid_data['dataClustering']['colorData']
        # self.colors_ = colors
        #
        # self.cluster_colors = centroid_data['dataClustering']['rgbColors']
        #
        # object_centroid = ax.scatter(X['component1'], X['component2'], marker='.', s=1000, lw=0, alpha=0.7,
        #                              c=colors)
        # object_centroid.set_zorder(1)
        #
        # # Labeling the clusters
        #
        # down = down_region
        #
        # # Draw white circles at cluster centers
        # object_centroid_circles = ax.scatter(centers[:, 0], centers[:, 1] - down,
        #                                      marker='o', c="white", alpha=1, s=400)
        # object_centroid_circles.set_zorder(2)
        #
        # for i, c in enumerate(centers):
        #     roman_number = extra_tools.write_roman(int(i + 1))
        #     object_centroid_circles_text = ax.scatter(c[0], c[1] - down, marker='$%s$' % roman_number, alpha=1,
        #                                               s=350,
        #                                               c='g')
        #     object_centroid_circles_text.set_zorder(3)

        #
        #
        # Reshape part
        reshape_data = self.ultra_clustering['reshape']

        if reshape_data['overrideClustNum'] is not None:
            n_clusters = reshape_data['overrideClustNum']
        else:
            n_clusters = reshape_data['clustNum']

        cluster_labels = reshape_data['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old center
        centers = reshape_data['dataClustering']['clusterInfo'][n_clusters]['centers']

        X = reshape_data['dataPCA']

        # New version
        centers = calculate_cluster_center_docking(X, cluster_labels)

        # TODO docking cluster colors maybe a better version
        # colors = sns.cubehelix_palette(n_colors=n_clusters, rot=-.4)
        colors = reshape_data['dataClustering']['clusterInfo'][n_clusters]['colorData']
        self.colors_ = colors

        self.cluster_colors = reshape_data['dataClustering']['clusterInfo'][n_clusters]['rgbColors']

        object_reshape = ax.scatter(X['component1'], X['component2'], marker='.', s=400, lw=0, alpha=0.7,
                                    c=colors)
        object_reshape.set_zorder(4)

        # Labeling the clusters

        # Draw white circles at cluster centers
        object_reshape_circles = ax.scatter(centers[:, 0], centers[:, 1],
                                            marker='o', c="white", alpha=1, s=150)
        object_reshape_circles.set_zorder(5)

        whole_size = len(reshape_data['dataOriginal']['SampleInfoNum'])

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            # clust_label =X[cluster_labels == i]
            cluster_values = reshape_data['dataOriginal']['SampleInfoNum'][cluster_labels == i]

            # exhaust_unique = exhaust_values.unique()
            #
            # unique_exhaust_values.update({str(i): exhaust_unique})
            #
            # unique_exhaust_values_list.append(sorted(exhaust_unique))
            #
            # curr_exhaust_text = ''
            # for i_exhaust in sorted(exhaust_unique):
            #     curr_exhaust_text += '%s ' % i_exhaust

            clust_size = len(cluster_values)

            percentage = (clust_size * 100) / whole_size

            self.clust_percentage_data['reshape'].update({int(i): percentage})



        for i, c in enumerate(centers):
            int_number = int(i + 1)
            object_reshape_circles_text = ax.scatter(c[0], c[1], marker='$%d$' % int_number, alpha=1, s=100,
                                                     c='b')
            object_reshape_circles_text.set_zorder(6)

        # ax.set_title(plot2_title)
        ax.set_xlabel(plot2_xlabel)
        ax.set_ylabel(plot2_ylabel)

        # plot_tools.change_ax_plot_font_size(ax2, 12)

        # plt.suptitle(plot_whole_titile,
        #              fontsize=16, fontweight='bold')

        fig.tight_layout()

        sns.set(style="white", context='paper', font_scale=1)

        fig.savefig(self.simulation_name + '_best_ultra_docking_reshape_PCA_analysis' + '_' + lang + '.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha, bbox_inches='tight')

        if show_plot is True:
            plt.show()
            # TODO create another function that shows only the best plot for kmeans

    @hlp.timeit
    def exhaust_cluster_analysis_pca(self, data, show_silhouette_plots=False):

        self.exhaust_sil_pca = []
        self.exhaust_calinski_pca = []
        self.exhaust_dunn_pca = []
        self.exhaust_dbi_pca = []

        # PseudoF and DBI from DRABAS book
        self.exhaust_book_dbi_pca = []
        self.exhaust_book_pseudoF_pca = []

        X = data

        for n_clusters in self.range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            try:
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(X)

                centers = clusterer.cluster_centers_
            except Exception as e:
                print('Error in exhaust_cluster_analysis_pca: ', e)
                return

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            try:
                silhouette_avg = silhouette_score(X, cluster_labels)

                calinski_avg = calinski_harabaz_score(X, cluster_labels)

                # looks like this is ok
                dunn_avg = dunn_fast(X, cluster_labels)

                converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)
                david_bouldain = davisbouldin(converted_values, centers)

                # pseudo_f = pseudoF_permanova(X, cluster_labels)
                # print("For n_clusters =", n_clusters,
                #       "The pseudo_f is :", pseudo_f)


                # BOOK implementation of pseudoF and DBI
                book_dbi = davis_bouldin(X, cluster_labels, centers)
                book_pseudoF = pseudo_F(X, cluster_labels, centers)

                sample_silhouette_values = silhouette_samples(X, cluster_labels)
            except:
                if n_clusters == 2:
                    silhouette_avg = 1.0
                    calinski_avg = 1.0
                    dunn_avg = 1.0
                    david_bouldain = 0.0
                    book_dbi = 0.0
                    book_pseudoF = 0.0
                    sample_silhouette_values = [1, 1]

                else:
                    silhouette_avg = 0.0
                    calinski_avg = 0.0
                    dunn_avg = 0.0
                    david_bouldain = 1.0
                    book_dbi = 1.0
                    book_pseudoF = 0.0
                    sample_silhouette_values = [0, 0]

            print("For n_clusters =", n_clusters,
                  "The average dunn is :", dunn_avg)

            print("For n_clusters =", n_clusters,
                  "The average dbd is :", david_bouldain)

            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            print("For n_clusters =", n_clusters,
                  "The average calinski_harabaz_score is :", calinski_avg)
            print('-*-' * 10)

            print("For n_clusters =", n_clusters,
                  "The average book pseudoF is :", book_pseudoF)
            print("For n_clusters =", n_clusters,
                  "The average book DBI  is :", book_dbi)

            # Store info for each n_clusters
            # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
            #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
            #                                         'labels': cluster_labels, 'centers': centers}})

            # Make decision based on average and then round value that would be your cluster quanity

            print('------------------------------------------------------------')

            self.exhaust_sil_pca.append(silhouette_avg)
            self.exhaust_calinski_pca.append(calinski_avg)
            self.exhaust_dunn_pca.append(dunn_avg)
            self.exhaust_dbi_pca.append(david_bouldain)

            # TODO test cluster analysis using book algorithms
            self.exhaust_book_dbi_pca.append(book_dbi)
            self.exhaust_book_pseudoF_pca.append(book_pseudoF)

            # Compute the silhouette scores for each sample


            self.exhaust_clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                                            'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                                            'labels': cluster_labels, 'centers': centers,
                                                            'silhouette_values': sample_silhouette_values,
                                                            'dbiBook': book_dbi,
                                                            'pseudoF_book': book_pseudoF}})

            return self.exhaust_clusters_info

            if show_silhouette_plots is True:
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

        test = 1

    @hlp.timeit
    def select_exhaust_number_of_clusters(self, exhaustiveness=0):
        # ["foo", "bar", "baz"].index("bar")
        max_silhouette = max(self.exhaust_sil_pca)
        max_dunn = max(self.exhaust_dunn_pca)
        min_dbi = min(self.exhaust_dbi_pca)
        max_pseudoF = max(self.exhaust_calinski_pca)

        sil_index = self.exhaust_sil_pca.index(max_silhouette)
        dunn_index = self.exhaust_dunn_pca.index(max_dunn)
        dbi_index = self.exhaust_dbi_pca.index(min_dbi)
        pseudoF_index = self.exhaust_calinski_pca.index(max_pseudoF)

        cluster_quantity = []
        cluster_quantity.append(self.range_n_clusters[sil_index])
        cluster_quantity.append(self.range_n_clusters[dunn_index])
        cluster_quantity.append(self.range_n_clusters[dbi_index])
        # cluster_quantity.append(self.range_n_clusters[pseudoF_index])

        print('------------------------------------------------')
        print('verify yolo', cluster_quantity)

        algorithms = ['Silhouette', 'Dunn', 'DBI']  # , 'pseudoF']

        print('Current exhaustiveness ', exhaustiveness)
        for algo in range(len(algorithms)):
            print('Algorith {0} {1}'.format(algorithms[algo], cluster_quantity[algo]))

        cluster_set = set(cluster_quantity)

        cluster_dict = {}
        for n_set in cluster_set:
            count = cluster_quantity.count(n_set)
            cluster_dict.update({n_set: count})

        print('verify yolo ', cluster_dict)

        import operator

        # TODO how to select middle yay
        whole_stuff = max(cluster_dict.items(), key=operator.itemgetter(1))
        # clust_num = max(cluster_dict.iterkeys(), key=lambda k: cluster_dict[k])
        clust_num_pre = [key for key, val in cluster_dict.items() if val == max(cluster_dict.values())]

        import numpy

        def median(lst):
            return numpy.median(numpy.array(lst))

        clust_num = sorted(clust_num_pre)[len(clust_num_pre) // 2]

        print("number of clusters is ", clust_num)

        # return clust_num

        # Use davis bouldin index
        return clust_num

    @hlp.timeit
    def pca_exhaust_plot(self, title='Exhaustiveness Analysis'):

        # Make more flexible whether pca_data or not
        pca_data = self.pca_data
        original_data = self.analysis_reshape_structure__  # self.pca_data

        sample_info_num = original_data['SampleInfoNum']

        unique_values = original_data['SampleInfoNum'].unique()

        unique_values = np.sort(unique_values, axis=None)
        test = 1

        # colors = iter(sns.color_palette("dark", len(unique_values)))

        # This is better for printing
        # look http://seaborn.pydata.org/tutorial/color_palettes.html
        colors = iter(sns.color_palette("cubehelix", len(unique_values)))

        ncols_num = 2
        fig, ax = plt.subplots(nrows=(len(unique_values) // ncols_num), ncols=ncols_num, sharex=True, sharey=True)

        fig.set_size_inches(plot_tools.cm2inch(17.7, 20))

        plt.suptitle(title, fontsize=14, fontweight='bold')

        # TODO very important for exhaust analysis
        self.exhaust_best_n_clusters = []
        self.unique_exhaust_val = []

        # self.exhaust_metrics_data = []
        self.exhaust_metrics_data = {}

        for x in range(0, len(unique_values) // ncols_num):  # Need to modify WORKS
            for y in range(ncols_num):

                # TODO not an elegant solution
                if x > 0:
                    k = x + x + y
                else:
                    k = x + y

                indexes = original_data[sample_info_num == unique_values[k]]
                print('Curr exhaust ', unique_values[k])
                self.curr_exhaust_plot = unique_values[k]

                modelNum = indexes['ModelNum']
                modelNum = modelNum.apply(np.int64) - 1
                # Select by index
                xyz = pca_data.ix[modelNum]

                # TODO VIP
                print('CLuster analysis on Exhaust Samples')
                # clusters_info = self.exhaust_cluster_analysis_pca(xyz)

                clusters_info = md_silhouette_analysis_pca(xyz,
                                                           None,
                                                           range_n_clusters=self.range_n_clusters,
                                                           show_plots=False,
                                                           algorithm='kmeans',
                                                           data_type='docking')

                self.exhaust_clusters_info = clusters_info

                type = 'centroid'
                # New version
                n_clusters, clust_analysis_info = select_number_of_clusters_v2_docking(clusters_info,
                                                                                       type,
                                                                                       self.range_n_clusters)

                # Old Version
                # n_clusters = self.select_exhaust_number_of_clusters(unique_values[k])
                self.exhaust_best_n_clusters.append(n_clusters)

                self.unique_exhaust_val.append(unique_values[k])

                # Save all metrics data to list
                # self.exhaust_metrics_data.append(self.exhaust_clusters_info)

                self.exhaust_sil_pca = self.extract_info_cluster_data(clusters_info, 'silhouette')
                self.exhaust_calinski_pca = self.extract_info_cluster_data(clusters_info, 'calinski')
                self.exhaust_dunn_pca = self.extract_info_cluster_data(clusters_info, 'dunn')
                self.exhaust_dbi_pca = self.extract_info_cluster_data(clusters_info, 'dbi')

                self.exhaust_book_dbi_pca = self.extract_info_cluster_data(clusters_info, 'book_dbi')
                self.exhaust_book_pseudoF_pca = self.extract_info_cluster_data(self.clusters_info, 'book_pseudoF')

                temp_data = {'sil': self.exhaust_sil_pca, 'calinski': self.exhaust_calinski_pca,
                             'dunn': self.exhaust_dunn_pca, 'dbi': self.exhaust_dbi_pca,
                             'dbiBook': self.exhaust_book_dbi_pca,
                             'pseudoF': self.exhaust_book_pseudoF_pca}

                self.exhaust_metrics_data.update({unique_values[k]: temp_data})

                print('\n')
                print('Exhaustiveness number ', unique_values[k])
                print('Best number of clusters is ', n_clusters)
                print('\n')

                cluster_labels = self.exhaust_clusters_info[n_clusters]['labels']
                sample_silhouette_values = self.exhaust_clusters_info[n_clusters]['silhouette_values']
                silhouette_avg = self.exhaust_clusters_info[n_clusters]['silhouette']

                centers = self.exhaust_clusters_info[n_clusters]['centers']

                # plot_color = next(colors)

                unique_labels = list(set(cluster_labels))

                # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)

                colors = iter(sns.color_palette("cubehelix", len(unique_labels)))
                for z in unique_labels:  # Need to modify WORKS
                    # print('k is ',k)
                    # k == -1 then it is an outlier
                    if z != -1:
                        data = xyz[cluster_labels == z]
                        ax[x][y].scatter(data['component1'], data['component2'], marker='o', s=20, lw=0, alpha=0.75,
                                         c=next(colors))
                        # binding_energy = cluster_energy['BindingEnergy']
                        # sns.distplot(binding_energy,  ax=ax[k][1])
                        # ax[k][1].hist(binding_energy, normed=False, color=colors[z], alpha=0.3)

                # ax[k].scatter(xyz['component1'], xyz['component2'], marker='o', s=10, lw=0, alpha=0.5,
                #                  c=colors)

                # colors = iter(cm.spectral(cluster_labels.astype(float) / n_clusters))


                for i, c in enumerate(centers):
                    ax[x][y].scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=160)

                ax[x][y].set_title(("Exhaustiveness {0}".format(int(unique_values[k]))),
                                   fontsize=12, fontweight='bold')

        # fig, ax = plt.subplots(len(unique_values), ncols=1, sharex=True, sharey=True)
        #
        # fig.set_size_inches(7, 28)
        #
        # for k in range(len(unique_values)):  # Need to modify WORKS
        #     indexes = original_data[sample_info_num == unique_values[k]]
        #
        #     modelNum = indexes['ModelNum']
        #     modelNum = modelNum.apply(np.int64) - 1
        #     # Select by index
        #     xyz = pca_data.ix[modelNum]
        #
        #     print('CLuster analysis on Exhaust Samples')
        #     self.exhaust_cluster_analysis_pca(xyz)
        #
        #     n_clusters = self.select_exhaust_number_of_clusters(unique_values[k])
        #
        #     print('\n')
        #     print('Exhaustiveness number ', unique_values[k])
        #     print('Best number of clusters is ', n_clusters)
        #     print('\n')
        #
        #     cluster_labels = self.exhaust_clusters_info[n_clusters]['labels']
        #     sample_silhouette_values = self.exhaust_clusters_info[n_clusters]['silhouette_values']
        #     silhouette_avg = self.exhaust_clusters_info[n_clusters]['silhouette']
        #
        #     centers = self.exhaust_clusters_info[n_clusters]['centers']
        #
        #     # plot_color = next(colors)
        #
        #     unique_labels = list(set(cluster_labels))
        #
        #     # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #     colors = iter(sns.color_palette("cubehelix", len(unique_labels)))
        #     for z in unique_labels:  # Need to modify WORKS
        #         # print('k is ',k)
        #         # k == -1 then it is an outlier
        #         if z != -1:
        #             data = xyz[cluster_labels == z]
        #             ax[k].scatter(data['component1'], data['component2'], marker='o', s=10, lw=0, alpha=0.75,
        #                           c=next(colors))
        #             # binding_energy = cluster_energy['BindingEnergy']
        #             # sns.distplot(binding_energy,  ax=ax[k][1])
        #             # ax[k][1].hist(binding_energy, normed=False, color=colors[z], alpha=0.3)
        #
        #     # ax[k].scatter(xyz['component1'], xyz['component2'], marker='o', s=10, lw=0, alpha=0.5,
        #     #                  c=colors)
        #
        #     # colors = iter(cm.spectral(cluster_labels.astype(float) / n_clusters))
        #
        #
        #     for i, c in enumerate(centers):
        #         ax[k].scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=160)
        #
        #     ax[k].set_title(("Exhaustiveness {0}".format(int(unique_values[k]))),
        #                     fontsize=12, fontweight='bold')

        # plt.show()
        fig.savefig(self.simulation_name + '_exhaust_pca_new.png', dpi=1200)
        plt.clf()

    @hlp.timeit
    # TODO still buggy
    def show_all_exhaustive_cluster_analysis_plots(self):
        '''
        Call cluster_analysis_pca(show_silhouette_plots=False) before calling this function
        :return:
        '''
        # Create a subplot with 2 row and 2 columns
        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)
        plt.clf()

        fig, ax = plt.subplots(nrows=len(self.unique_exhaust_val), ncols=4,
                               sharex=True)  # sharex='col', ) TODO this can be used for shared columns
        fig.set_size_inches(plot_tools.cm2inch(17.7, 20))

        # fig.set_size_inches(20,20)

        # fig.tight_layout()

        # sns.set(style="whitegrid")
        sns.set(font_scale=0.5)

        size_marker = 25

        xmargin = 0.1
        ymargin = 0.1

        for x in range(0, len(self.unique_exhaust_val)):
            best_n_clusters = self.exhaust_best_n_clusters[x]
            # self.unique_exhaust_val.append(unique_values[k])

            # Save all metrics data to list
            print('Curr exhaust ', self.unique_exhaust_val[x])
            metrics_data = self.exhaust_metrics_data[self.unique_exhaust_val[x]]

            cluster_range = self.range_n_clusters
            score = metrics_data['dbi']
            criteria_name = 'Davis-Bouldain\n Index'
            # score_text = 'The optimal clustering solution has the smallest Davies-Bouldin index value.'
            score_text = 'Exhaustiveness: {0}'.format(self.unique_exhaust_val[x])
            ax[x][0].scatter(cluster_range, score, marker='o', c='b', s=size_marker)
            ax[x][0].plot(cluster_range, score, ':k', linewidth=3.0)

            ax[x][0].set_xlim(cluster_range[0], cluster_range[-1])

            # ax[x][0].set_title(score_text)
            ax[x][0].set_title(score_text)
            ax[x][0].set_xlabel('n of clusters')
            ax[x][0].set_ylabel(criteria_name)

            # ax[x][0].autoscale()
            # ax[x][0].autoscale_view()
            # ax[x][0].margins(x=xmargin, y=ymargin, tight=True)

            cluster_range = self.range_n_clusters
            score = metrics_data['dunn']
            criteria_name = "Dunn's Index"
            # score_text = "Maximum value of the index represents the right partitioning given the index"
            score_text = 'Exhaustiveness: {0}'.format(self.unique_exhaust_val[x])
            ax[x][1].scatter(cluster_range, score, marker='o', c='b', s=size_marker)
            ax[x][1].plot(cluster_range, score, ':k', linewidth=3.0)

            ax[x][1].set_xlim(cluster_range[0], cluster_range[-1])
            ax[x][1].set_title(score_text)
            ax[x][1].set_xlabel('n of clusters')
            ax[x][1].set_ylabel(criteria_name)

            # ax[x][1].autoscale()
            # ax[x][1].autoscale_view()
            # ax[x][1].margins(x=xmargin, y=ymargin, tight=True)

            cluster_range = self.range_n_clusters
            score = metrics_data['sil']
            criteria_name = 'Mean Silhouette\n Score'
            # score_text = 'Objects with a high silhouette value are considered well clustered'
            score_text = 'Exhaustiveness: {0}'.format(self.unique_exhaust_val[x])
            ax[x][2].scatter(cluster_range, score, marker='o', c='b', s=size_marker)
            ax[x][2].plot(cluster_range, score, ':k', linewidth=3.0)

            ax[x][2].set_xlim(cluster_range[0], cluster_range[-1])
            ax[x][2].set_title(score_text)
            ax[x][2].set_xlabel('n of clusters')
            ax[x][2].set_ylabel(criteria_name)

            # ax[x][2].autoscale()
            # ax[x][2].autoscale_view()
            # ax[x][2].margins(x=xmargin, y=ymargin, tight=True)

            cluster_range = self.range_n_clusters
            score = metrics_data['calinski']
            criteria_name = 'Calinski-Harabaz\n score'
            # score_text = 'Objects with a high Calinski-Harabaz score value are considered well clustered'
            score_text = 'Exhaustiveness: {0}'.format(self.unique_exhaust_val[x])
            ax[x][3].scatter(cluster_range, score, marker='o', c='b', s=size_marker)
            ax[x][3].plot(cluster_range, score, ':k', linewidth=3.0)
            ax[x][3].set_xlim(cluster_range[0], cluster_range[-1])
            ax[x][3].set_title(score_text)
            ax[x][3].set_xlabel('n of clusters')
            ax[x][3].set_ylabel(criteria_name)

            # ax[x][3].autoscale()
            # ax[x][3].autoscale_view()
            # ax[x][3].margins(x=xmargin, y=ymargin, tight=True)

            plot_tools.change_ax_plot_font_size(ax[x][0], 4)
            plot_tools.change_ax_plot_font_size(ax[x][1], 4)
            plot_tools.change_ax_plot_font_size(ax[x][2], 4)
            plot_tools.change_ax_plot_font_size(ax[x][3], 4)

        #
        plt.suptitle(("Exhaustiveness determination by cluster analysis"),
                     fontsize=12, fontweight='bold')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.93)

        # plt.show()
        fig.savefig(self.simulation_name + '_exhaust_scores_all.png', dpi=1200)

    @hlp.timeit
    def exhaust_n_clusters_plot(self, custom_dpi=1200):
        '''
        Before using this function, call pca_exhaust_plot() method
        :return:
        '''

        import seaborn as sns
        plt.clf()

        sns.set(style="whitegrid")
        sns.set(font_scale=0.73)
        # plt.figure(figsize=(14, 7))
        plt.figure(figsize=plot_tools.cm2inch(8.4, 8.4))

        print('verify yolo')
        print(self.unique_exhaust_val)

        self.unique_exhaust_val = [int(x) for x in self.unique_exhaust_val]

        dataframe = converters.convert_data_to_pandas(self.unique_exhaust_val, self.exhaust_best_n_clusters,
                                                      x_axis_name='Exhaustiveness Values',
                                                      y_axis_name='Number of Clusters')

        # g = sns.factorplot(x="Exhaustiveness Values", y="Number of Clusters", data=dataframe,
        #                    capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)


        zolo = sns.pointplot(x="Exhaustiveness Values", y="Number of Clusters", data=dataframe)
        zolo.set_ylabel('Number of Clusters')
        # zolo.savefig("Exhaust_n_clusterPlot.png", dpi=custom_dpi)
        fig = zolo.get_figure()
        fig.tight_layout()
        fig.savefig(self.simulation_name + "_exhaust_n_clusterPlot.png", dpi=custom_dpi)
        # g.despine(left=True)

        # plt.show()

        plt.clf()

    @hlp.timeit
    def pca_exhaust_plot_plus_hist(self):

        # Make more flexible whether pca_data or not
        pca_data = self.pca_data
        original_data = self.analysis_reshape_structure__  # self.pca_data

        sample_info_num = original_data['SampleInfoNum']

        unique_values = original_data['SampleInfoNum'].unique()

        unique_values = np.sort(unique_values, axis=None)
        test = 1

        # colors = iter(sns.color_palette("dark", len(unique_values)))

        # This is better for printing
        # look http://seaborn.pydata.org/tutorial/color_palettes.html
        colors = iter(sns.color_palette("cubehelix", len(unique_values)))

        fig, ax = plt.subplots(len(unique_values), ncols=2)

        fig.set_size_inches(12, 30)

        for k in range(len(unique_values)):  # Need to modify WORKS
            indexes = original_data[sample_info_num == unique_values[k]]

            modelNum = indexes['ModelNum']
            modelNum = modelNum.apply(np.int64) - 1
            # Select by index
            xyz = pca_data.ix[modelNum]

            print('CLuster analysis on Exhaust Samples')
            self.exhaust_cluster_analysis_pca(xyz)

            # TODO select number of clusters
            n_clusters = self.select_exhaust_number_of_clusters(unique_values[k])

            print('\n')
            print('Exhaustiveness number ', unique_values[k])
            print('Best number of clusters is ', n_clusters)
            print('\n')

            cluster_labels = self.exhaust_clusters_info[n_clusters]['labels']
            sample_silhouette_values = self.exhaust_clusters_info[n_clusters]['silhouette_values']
            silhouette_avg = self.exhaust_clusters_info[n_clusters]['silhouette']

            centers = self.exhaust_clusters_info[n_clusters]['centers']

            # plot_color = next(colors)

            unique_labels = list(set(cluster_labels))

            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            # colors = sns.color_palette("cubehelix", len(unique_labels))
            ax[k][0].scatter(xyz['component1'], xyz['component2'], marker='o', s=150, lw=0, alpha=0.7,
                             c=colors)

            # colors = iter(cm.spectral(cluster_labels.astype(float) / n_clusters)  )
            for z in unique_labels:  # Need to modify WORKS
                # print('k is ',k)
                # k == -1 then it is an outlier
                if z != -1:
                    cluster_data = []
                    cluster_energy = indexes[cluster_labels == z]
                    binding_energy = cluster_energy['BindingEnergy']
                    # sns.distplot(binding_energy,  ax=ax[k][1])
                    ax[k][1].hist(binding_energy, normed=False, color=colors[z], alpha=0.3)

            ax[k][1].set(xlabel='Binding Affinity(kcal/mol)', ylabel='Frequency')

            for i, c in enumerate(centers):
                ax[k][0].scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=160)

            ax[k][0].set_title(("Exhaustiveness {0}".format(int(unique_values[k]))),
                               fontsize=12, fontweight='bold')

        # plt.show()
        fig.savefig(self.simulation_name + '_exhaust_pca_hist.png', dpi=600)

    @hlp.timeit
    def pca_exhaust_plot_plus_bar(self):

        # Make more flexible whether pca_data or not
        pca_data = self.pca_data
        original_data = self.analysis_reshape_structure__  # self.pca_data

        sample_info_num = original_data['SampleInfoNum']

        unique_values = original_data['SampleInfoNum'].unique()

        unique_values = np.sort(unique_values, axis=None)
        test = 1

        # colors = iter(sns.color_palette("dark", len(unique_values)))

        # This is better for printing
        # look http://seaborn.pydata.org/tutorial/color_palettes.html
        colors = iter(sns.color_palette("cubehelix", len(unique_values)))

        fig, ax = plt.subplots(len(unique_values), ncols=2)

        fig.set_size_inches(12, 30)

        fig.subplots_adjust(hspace=.5)

        for k in range(len(unique_values)):  # Need to modify WORKS
            indexes = original_data[sample_info_num == unique_values[k]]

            modelNum = indexes['ModelNum']
            modelNum = modelNum.apply(np.int64) - 1
            # Select by index
            xyz = pca_data.ix[modelNum]

            print('CLuster analysis on Exhaust Samples')
            self.exhaust_cluster_analysis_pca(xyz)

            n_clusters = self.select_exhaust_number_of_clusters(unique_values[k])

            print('\n')
            print('Exhaustiveness number ', unique_values[k])
            print('Best number of clusters is ', n_clusters)
            print('\n')

            cluster_labels = self.exhaust_clusters_info[n_clusters]['labels']
            sample_silhouette_values = self.exhaust_clusters_info[n_clusters]['silhouette_values']
            silhouette_avg = self.exhaust_clusters_info[n_clusters]['silhouette']

            centers = self.exhaust_clusters_info[n_clusters]['centers']

            # plot_color = next(colors)

            unique_labels = list(set(cluster_labels))

            # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            #
            # ax[k][0].scatter(xyz['component1'], xyz['component2'], marker='o', s=150, lw=0, alpha=0.7,
            #                  c=colors)

            colors = sns.color_palette("cubehelix", len(unique_labels))
            sns.regplot('component1', 'component2', data=xyz, fit_reg=False, ax=ax[k][0],
                        scatter_kws={"marker": "D",
                                     "s": 100,
                                     'c': colors})

            # colors = iter(cm.spectral(cluster_labels.astype(float) / n_clusters)  )
            label_num = []
            count_data = []
            for z in unique_labels:  # Need to modify WORKS
                # print('k is ',k)
                # k == -1 then it is an outlier
                if z != -1:
                    label_num.append(z)
                    count = len(indexes[cluster_labels == z])
                    count_data.append(count)
                    # binding_energy = cluster_energy['BindingEnergy']
                    # sns.distplot(binding_energy,  ax=ax[k][1])
                    # ax[k][1].hist(binding_energy, normed=False, color=colors[z], alpha=0.3)

            label_num = np.array(label_num)
            count_data = np.array(count_data)

            sns.barplot(label_num, count_data, ax=ax[k][1])

            ax[k][1].set(xlabel='Cluster', ylabel='Conformation Count')

            for i, c in enumerate(centers):
                ax[k][0].scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=160)

            ax[k][0].set_title(("Exhaustiveness {0}".format(int(unique_values[k]))),
                               fontsize=12, fontweight='bold')

            # fig[k].suptitle(("Exhaustiveness {0}".format(int(unique_values[k]))),
            #                 fontsize=12, fontweight='bold')

        # plt.show()
        fig.savefig(self.simulation_name + '_exhaust_pca_bar.png', dpi=600)


        #
        # ax2.scatter(X['component1'], X['component2'], marker='.', s=250, lw=0, alpha=0.7,
        #             c=colors)

    ####################################################################################################################

    @hlp.timeit
    def show_cluster_histograms_ultra(self, type, custom_dpi=1200, show_plot=False, trasparent_alpha=False):

        # n_clusters = self.select_number_of_clusters()
        #
        # cluster_labels = self.clusters_info[n_clusters]['labels']

        data_to_use = self.ultra_clustering[type]

        n_clusters = data_to_use['clustNum']

        cluster_labels = data_to_use['dataClustering']['clusterInfo'][n_clusters]['labels']
        # Old version
        centers = data_to_use['dataClustering']['clusterInfo'][n_clusters]['centers']

        # data = self.clusters_info[self.clust_num]
        data = data_to_use['dataClustering']['clusterInfo'][n_clusters]
        print(data)

        labels = data['labels']

        pca_data = data_to_use['dataPCA']
        original_data = data_to_use['dataOriginal']  # self.pca_data

        cluster_list = {}
        unique_labels = list(set(labels))
        # colors = iter(plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))))
        # colors = iter(plt.cm.rainbow(np.linspace(0,1, len(unique_labels))))


        import seaborn as sns
        sns.set(style="white", context='paper')
        # colors = iter(sns.color_palette("dark", len(unique_labels)))

        # Better for printing
        # look http://seaborn.pydata.org/tutorial/color_palettes.html
        # colors = iter(sns.color_palette("cubehelix", len(unique_labels)))
        colors = iter(self.colors_)

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                cluster_data = []
                xyz = original_data[labels == k]
                model_num = xyz['BindingEnergy']

                fig, axs = plt.subplots(figsize=plot_tools.cm2inch(17.7, 8), ncols=2, nrows=1)

                plot_color = next(colors)
                for col in (0, 1):
                    sns.distplot(model_num, kde=col, norm_hist=col, ax=axs[col], color=plot_color,
                                 hist_kws={"linewidth": 3})

                # axs = sns.distplot(model_num, norm_hist=False, kde=False)
                axs[0].set(xlabel='Binding Affinity(kcal/mol)', ylabel='Frequency with No kernel density')
                axs[1].set(xlabel='Binding Affinity(kcal/mol)', ylabel='Frequency with KDE')
                # sns.plt.title('Cluster {0}'.format(str(k)))


                cluster_percentage = round((len(xyz) / len(original_data)) * 100, 3)

                total_sample_size = len(original_data)

                mean = np.mean(model_num)
                std = np.std(model_num)

                if type == 'centroid':
                    num = extra_tools.write_roman(k + 1)
                    text_to_add = 'Region {0}'.format(num)
                else:
                    num = str(k + 1)
                    text_to_add = 'Cluster {0}'.format(num)

                # TODO  plt.suptitle
                plt.suptitle(
                    '{0} {1}% of {2} $_\Delta$G = {3} ± {4} kcal/mol'.format(text_to_add, cluster_percentage,
                                                                             total_sample_size,
                                                                             round(mean, 3),
                                                                             round(std, 3)),
                    fontsize=14, fontweight='bold')
                # sns.despine(offset=10, trim=True)
                fig.tight_layout()
                fig.subplots_adjust(top=0.89)
                fig.savefig(self.simulation_name + '_cluster_energy_{0}_type_{1}.png'.format(str(k + 1), type),
                            dpi=custom_dpi,
                            transparent=trasparent_alpha, bbox_inches='tight')

                if show_plot is True:
                    plt.show()

    @hlp.timeit
    def show_cluster_histograms(self, custom_dpi=1200, show_plot=False, trasparent_alpha=False):

        # n_clusters = self.select_number_of_clusters()
        #
        # cluster_labels = self.clusters_info[n_clusters]['labels']
        data = self.clusters_info[self.clust_num]
        print(data)

        labels = data['labels']

        pca_data = self.pca_data
        original_data = self.analysis_structure  # self.pca_data

        cluster_list = {}
        unique_labels = list(set(labels))
        # colors = iter(plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))))
        # colors = iter(plt.cm.rainbow(np.linspace(0,1, len(unique_labels))))


        import seaborn as sns
        sns.set(style="white", context='paper')
        # colors = iter(sns.color_palette("dark", len(unique_labels)))

        # Better for printing
        # look http://seaborn.pydata.org/tutorial/color_palettes.html
        # colors = iter(sns.color_palette("cubehelix", len(unique_labels)))
        colors = iter(self.colors_)

        for k in unique_labels:  # Need to modify WORKS
            # print('k is ',k)
            # k == -1 then it is an outlier
            if k != -1:
                cluster_data = []
                xyz = original_data[labels == k]
                model_num = xyz['BindingEnergy']

                fig, axs = plt.subplots(figsize=(6, 4), ncols=2, nrows=1)

                plot_color = next(colors)
                for col in (0, 1):
                    sns.distplot(model_num, kde=col, norm_hist=col, ax=axs[col], color=plot_color,
                                 hist_kws={"linewidth": 3})

                # axs = sns.distplot(model_num, norm_hist=False, kde=False)
                axs[0].set(xlabel='Binding Affinity(kcal/mol)', ylabel='Frequency with No kernel density')
                axs[1].set(xlabel='Binding Affinity(kcal/mol)', ylabel='Frequency with KDE')
                # sns.plt.title('Cluster {0}'.format(str(k)))


                cluster_percentage = round((len(xyz) / len(original_data)) * 100, 3)

                total_sample_size = len(original_data)

                mean = np.mean(model_num)
                std = np.std(model_num)

                plt.suptitle(
                    'Cluster {0} {1}% of {2} $_\Delta$G = {3} ± {4} kcal/mol'.format(str(k + 1), cluster_percentage,
                                                                                     total_sample_size,
                                                                                     round(mean, 3),
                                                                                     round(std, 3)),
                    fontsize=14, fontweight='bold')
                # sns.despine(offset=10, trim=True)

                fig.savefig(self.simulation_name + '_cluster_energy_{0}.png'.format(str(k + 1)), dpi=custom_dpi,
                            transparent=trasparent_alpha)

                if show_plot is True:
                    plt.show()

    @hlp.timeit
    def extract_info_cluster_data(self, cluster_data, key):
        temp_data = []
        for clust_num in self.range_n_clusters:
            temp_data.append(cluster_data[clust_num][key])
        return temp_data

    @hlp.timeit
    def cluster_centroids_and_reshapes(self):

        #         self.data_for_analysis_all

        test = 1
        # print(self.concatenated_analysis__[int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__)+ 1])


        # TODO CENTROID ANALYSIS

        centroid_indexes = self.concatenated_analysis__[
            self.concatenated_analysis__['Type'] == 'centroid'].index.tolist()
        test1 = len(self.concatenated_analysis__) / 2

        # temp_centroid = self.pca_data_all[centroid_indexes[0]:centroid_indexes[-1] + 1]

        data_centroid = self.concatenated_analysis__[
                        int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]

        self.data_centroid = data_centroid

        # self.concatenated_analysis__[self.data_cols]

        temp_centroid_pca = self.pca_data_all[
                            int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]

        temp_centroid_pre = self.concatenated_analysis__[
                            int(len(self.concatenated_analysis__) / 2):len(self.concatenated_analysis__) + 1]

        temp_centroid = temp_centroid_pre[self.data_cols]

        self.temp_centroid = temp_centroid

        type = 'centroid'
        centroid_cluster_analysis = self.cluster_analysis_custom(temp_centroid, type)
        print('Centroid clustering Finished \n')

        # centroid_cluster_analysis_pca = self.cluster_analysis_custom(temp_centroid_pca)
        # print('Centroid PCA clustering Finished \n')

        clusters_info_centroid = centroid_cluster_analysis['clusterInfo']

        clust_num_centroid = centroid_cluster_analysis['clustNum']

        # data_to_return = {
        #                   'colors':colors,
        #                   'labels':cluster_labels,
        #                   'colorData':cluster_colors,
        #                   'clusterInfo': clusters_info,
        #                   'clustNum': clust_num,
        #                   'clusterAnalysisInfo': clust_analysis_info,
        #                   'overrideClustNum': None}

        centroid_cluster_models = self.collect_cluster_info_v2(clusters_info_centroid, clust_num_centroid,
                                                               temp_centroid, data_centroid)
        test = 1

        # cluster_labels = centroid_data['dataClustering']['clusterInfo'][n_clusters]['labels']




        # TODO RESHAPE ANALYSIS
        reshape_indexes = self.concatenated_analysis__[self.concatenated_analysis__['Type'] == 'reshape'].index.tolist()
        data_reshape = self.concatenated_analysis__[reshape_indexes[0]:reshape_indexes[-1] + 1]

        self.data_reshape = data_reshape

        temp_reshape_pca = self.pca_data_all[reshape_indexes[0]:reshape_indexes[-1] + 1]

        temp_reshape_pre = self.concatenated_analysis__[reshape_indexes[0]:reshape_indexes[-1] + 1]

        temp_reshape = temp_reshape_pre[self.data_cols]

        self.temp_reshape = temp_reshape

        type = 'reshape'
        reshape_cluster_analysis = self.cluster_analysis_custom(temp_reshape, type)
        print('Reshape clustering Finished \n')

        # reshape_cluster_analysis_pca = self.cluster_analysis_custom(temp_reshape_pca)
        # print('Reshape PCA clustering Finished \n')

        clusters_info_reshape = reshape_cluster_analysis['clusterInfo']

        clust_num_reshape = reshape_cluster_analysis['clustNum']

        # data_to_return = {
        #                   'colors':colors,
        #                   'labels':cluster_labels,
        #                   'colorData':cluster_colors,
        #                   'clusterInfo': clusters_info,
        #                   'clustNum': clust_num,
        #                   'clusterAnalysisInfo': clust_analysis_info,
        #                   'overrideClustNum': None}

        reshape_cluster_models = self.collect_cluster_info_v2(clusters_info_reshape, clust_num_reshape,
                                                              temp_reshape, data_reshape)

        # converters.convert_seaborn_color_to_rgb(self.cluster_colors_pre_rgb)

        # TODO update this part
        self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid_pca,
                                              'dataPre': temp_centroid_pre,
                                              'dataToCluster': temp_centroid,
                                              'dataOriginal': data_centroid,
                                              'clusterModels': centroid_cluster_models,
                                              'dataClustering': centroid_cluster_analysis,
                                              'clustNum': clust_num_centroid,
                                              'overrideClustNum':None},
                                 'reshape': {'dataPCA': temp_reshape_pca,
                                             'dataPre': temp_reshape_pre,
                                             'dataToCluster': temp_reshape,
                                             'dataOriginal': data_reshape,
                                             'clusterModels': reshape_cluster_models,
                                             'dataClustering': reshape_cluster_analysis,
                                             'clustNum': clust_num_reshape,
                                             'overrideClustNum': None}}

        # self.ultra_clustering = {'centroid': {'dataPCA': temp_centroid_pca,
        #                                       'dataPre':temp_centroid_pre,
        #                                       'dataToCluster':temp_centroid,
        #                                       'dataOriginal': data_centroid,
        #                                       'clusterModels': centroid_cluster_models,
        #                                       'dataClustering': centroid_cluster_analysis,
        #                                       'dataClusteringOCA': centroid_cluster_analysis_pca,
        #                                       'clustNum': clust_num_centroid},
        #                          'reshape': {'dataPCA': temp_reshape_pca,
        #                                      'dataPre':temp_reshape_pre,
        #                                      'dataToCluster':temp_reshape,
        #                                      'dataOriginal': data_reshape,
        #                                      'clusterModels': reshape_cluster_models,
        #                                      'dataClustering': reshape_cluster_analysis,
        #                                      'dataClusteringPCA': reshape_cluster_analysis_pca,
        #                                      'clustNum': clust_num_reshape}}

        print('Finished ultra clustering')
        test = 1

        # TODO fix parallel implementation

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


        if clusters_info is not None:
            clust_num, clust_analysis_info = select_number_of_clusters_v2_docking(clusters_info, type,
                                                                                  self.range_n_clusters)

            test = 1

            # cluster_models = self.collect_cluster_info()

            # Backup

            # TODO color part for ultra docking analysis
        else:
            clust_num =1


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

        # self.cluster_list = self.collect_cluster_info()

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

            # ax2.set_xlabel("PC1 (Å)")
            # ax2.set_ylabel("PC2 (Å)")

            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.show()

    def plotHist(self):
        self.analysis_structure['BindingEnergy'].plot.hist()
        plt.show()

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

    def transform_data(self):
        mol_data = {}
        for model, model_info in zip(self.object, self.info):
            # print(model_info)
            pandas_model = self.pandas_transformation(model)
            mol_data.update({model_info[0]: {'dataframe': pandas_model, 'vina_info': model_info[1:]}})

        return mol_data

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

    def add_receptor_for_viz(self, receptor_file):
        self.receptor_file_viz = receptor_file

    def generate_pymol_viz_for_thread(self, receptor, exhaust_data, save_name):
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

        self.data_to_open = {}
        for exhaust_mol in exhaust_data:
            molecule = exhaust_data[exhaust_mol]
            filepath = molecule.file_path
            data_type = molecule.sample_info_type
            num = molecule.sample_info_num

            self.data_to_open.update({num: filepath})

        data_keys = list(self.data_to_open.keys())

        int_keys = sorted([int(x) for x in data_keys])

        for i in int_keys:
            filepath = self.data_to_open[str(i)]
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


    def generate_pymol_viz_clust_exhaust_for_thread(self, receptor, exhaust_data, save_name,
                                                    percentage=False, label_atom='C12'):
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
        for exhaust_mol in exhaust_data:
            # molecule = exhaust_data[exhaust_mol]
            # filepath = molecule.file_path
            # data_type = molecule.sample_info_type
            # num = molecule.sample_info_num

            test = 1

            curr_data = exhaust_data[exhaust_mol]

            curr_index = exhaust_mol
            correct_file_pymol_name = 'exhaust_clust_{0}'.format(curr_index+1)

            correct_topol_filepath = curr_data['relativePath']
            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            curr_color = 'exhaus_cluster_color_{0}'.format(curr_index+1)
            pymol.cmd.set_color(curr_color, curr_data['colors'])
            pymol.cmd.color(curr_color, correct_file_pymol_name)

            if percentage is True:
                curr_percentage = curr_data['percentage']
                curr_label= 'exhaus_cluster_label_{0}'.format(curr_index + 1)
                curr_expression = '"{0}%"'.format(curr_percentage)
                #pymol.cmd.set_label(curr_color, curr_data['colors'])

                curr_selection = '{0} and n. {1]'.format(correct_file_pymol_name,label_atom)

                # pymol.cmd.label(correct_file_pymol_name, expression=curr_expression)
                pymol.cmd.label(selection=curr_selection, expression=curr_expression)

                # select test, exhaust_clust_1//CA
                # WORKS select exhaust_clust_1 and n. C12



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


    def generate_pymol_viz_clust_exhaust_percent_for_thread(self, receptor, exhaust_data, save_name):
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
        for exhaust_mol in exhaust_data:
            # molecule = exhaust_data[exhaust_mol]
            # filepath = molecule.file_path
            # data_type = molecule.sample_info_type
            # num = molecule.sample_info_num

            test = 1

            curr_data = exhaust_data[exhaust_mol]

            curr_index = exhaust_mol
            correct_file_pymol_name = 'exhaust_clust_{0}'.format(curr_index+1)

            correct_topol_filepath = curr_data['relativePath']
            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            curr_color = 'exhaus_cluster_color_{0}'.format(curr_index+1)
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




    def generate_exhaust_pymol_viz_thread(self, type='data'):
        '''This is to make sure that pymol methods run separately'''
        import threading, time

        # self.sample_files = self.obtain_samples()
        # self.samples_data = self.load_samples()


        print('Start of Pymol Exhaust show smethod --->  ')
        save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + type  + '_exaustiveness_pymolViz.pse'

        if type == 'data':
            self.generate_pymol_viz_for_thread(self.receptor_file_viz, self.samples_data, save_state_name)
        elif type== 'cluster_percentage':
            save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + type + 'percentage_exaustiveness_pymolViz.pse'
            data_to_pass = self.save_extract_files_list['centroid']
            self.generate_pymol_viz_clust_exhaust_for_thread(self.receptor_file_viz,
                                                             data_to_pass, save_state_name,percentage=True)
        else:
            data_to_pass = self.save_extract_files_list['centroid']
            self.generate_pymol_viz_clust_exhaust_for_thread(self.receptor_file_viz, data_to_pass, save_state_name)


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


