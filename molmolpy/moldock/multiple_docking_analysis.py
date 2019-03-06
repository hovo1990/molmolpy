# -*- coding: utf-8 -*-

#!/usr/bin/env python
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

import matplotlib
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters

matplotlib.style.use('ggplot')

from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


class MultipleDockingAnalysisObject(object):
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

    def __init__(self, multiple_docking_samples_object, load_way='molmolpy', k_clust=10):

        self.multiple_docking_samples_object = multiple_docking_samples_object

        self.all_data = self.concat_dataframes()
        print(self.all_data)

        self.data_for_analysis = self.all_data[['X', 'Y', 'Z']]

        self.pca_data = self.pca_analysis()

        self.clusters_info = {}
        self.range_n_clusters = list(range(2, k_clust + 1))
        # self.silhouette_analysis_pca(show_plots=True)
        self.silhouette_analysis_pca()

        # self.silhouette_graph_pca()
        # self.dunn_graph_pca()
        # self.dbi_graph_pca()

        # Then need to extract conformations
        self.clust_num = self.select_number_of_clusters()
        print('self clust num ', self.clust_num)
        # self.cluster_models = self.collect_cluster_info()

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

    def collect_cluster_info(self):
        data = self.clusters_info[self.clust_num]
        print(data)

        labels = data['labels']
        # Make more flexible whether pca_data or not
        pca_data = self.pca_data
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

    def export_cluster_models(self):
        clust_numbers = self.cluster_models
        for clust in clust_numbers:
            cluster = clust_numbers[clust]
            # print(cluster)
            for model in cluster:
                print(model['modelNum'])

                # for vmd write dummy molecules


                # Molecule pd row -> row = next(df.iterrows())[1]

    #TODO create another function that shows only the best plot for kmeans
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
            self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                                    'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                                    'labels': cluster_labels, 'centers': centers}})

            # Make decision based on average and then round value that would be your cluster quanity

            print('------------------------------------------------------------')

            self.sil_pca.append(silhouette_avg)
            self.calinski_pca.append(calinski_avg)
            self.dunn_pca.append(dunn_avg)
            self.dbi_pca.append(david_bouldain)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

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

    def plot_pca(self, show_plots=False, clust_num=None):

        fignum = 1
        fig = plt.figure(fignum)
        plt.clf()

        if show_plots is True:
            # Create a subplot with 1 row and 2 columns
            X = self.pca_data

            fig.set_size_inches(18, 7)

            fig.set_size_inches(18, 7)

            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(len(self.multiple_docking_samples_object))
            # plt.scatter(X['component1'], X['component2'], marker='.', s=50, lw=0, alpha=0.7,
            #             c=colors)


            # colors = plt.cm.Spectral(np.linspace(0, 1, len(self.multiple_docking_samples_object)))

            if clust_num is None:
                cluster_info = self.clusters_info[self.clust_num]
            else:
                cluster_info = self.clusters_info[clust_num]


            labels = cluster_info['labels']
            unique_labels = list(set(labels))

            # for k in unique_labels:  # Need to modify WORKS
            #     # print('k is ',k)
            #     xyz = X[labels == k]
            #     if k != -1:

            for frame_n in range(0, len(self.multiple_docking_samples_object)):
                frame = self.multiple_docking_samples_object[frame_n].get_data_for_analysis()

                frame_mol = self.multiple_docking_samples_object[frame_n].molecule_name
                color_mol = self.multiple_docking_samples_object[frame_n].color_plot
                size = self.multiple_docking_samples_object[frame_n].size
                marker = self.multiple_docking_samples_object[frame_n].marker
                zorder_mol = self.multiple_docking_samples_object[frame_n].z_order

                indexes = self.all_data[self.all_data['MolName'] == frame_mol].index.tolist()
                dataframe_mol = self.all_data[self.all_data['MolName'] == frame_mol]

                energy_mol = dataframe_mol['BindingEnergy'].describe()

                mean = energy_mol['mean']
                std = energy_mol['std']

                label = r'{0} $\Delta$G = {1}±{2} kcal/mol'.format(frame_mol, round(mean, 3), round(std, 3))

                print('mol ', frame_mol)
                print('energy ', energy_mol)
                print('---------------------')

                tempX = X[indexes[0]:indexes[-1] + 1]
                # object = plt.scatter(tempX['component1'], tempX['component2'], marker=marker,
                #                      edgecolors='k', s=size,
                #                      lw=0.4, alpha=0.5,
                #                      c=color_mol, label=label)
                object = plt.scatter(tempX['component1'], tempX['component2'], marker=marker,
                                     edgecolors='k', s=size,
                                     lw=0.4,
                                     c=color_mol, label=label)
                object.set_zorder(zorder_mol)

            # Labeling the clusters
            centers = cluster_info['centers']
            # Draw white circles at cluster centers
            object1 = plt.scatter(centers[:, 0], centers[:, 1],
                                  marker='o', c="white", alpha=1, s=300)
            object1.set_zorder(100)

            for i, c in enumerate(centers):
                object2 = plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=300, c='k')
                object2.set_zorder(100)

            # plt.title("The visualization of the clustered data.")
            # plt.xlabel("Feature space for the 1st feature")
            # plt.ylabel("Feature space for the 2nd feature")

            plt.title("Визуализация анализа данных кластеризации")
            plt.xlabel("Компонент 1")
            plt.ylabel("Компонент 2")

            leg = plt.legend(loc='upper right', fontsize=20)
            # leg.get_frame().set_alpha(0.5)

            plt.suptitle(("K-cредний анализ данных с количеством "
                          "кластеров = %d" % clust_num),
                         fontsize=14, fontweight='bold')



                # Labeling the clusters
                # Draw white circles at cluster centers
                # plt.scatter(centers[:, 0], centers[:, 1],
                #             marker='o', c="white", alpha=1, s=100)

                # for i, c in enumerate(centers):
                #     plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=100)

        plt.savefig('docking_analysis.png', dpi=800)  #, transparent=True)
        plt.show()

    def concat_dataframes(self):
        frame_0 = self.multiple_docking_samples_object[0].get_data_for_analysis()
        for frame_n in range(1, len(self.multiple_docking_samples_object)):
            frame_0 = frame_0.append(self.multiple_docking_samples_object[frame_n].get_data_for_analysis(),
                                     ignore_index=True)

        return frame_0

    def multiple_pca(self, show_plots=False):

        fignum = 1
        fig = plt.figure(fignum)
        plt.clf()

        for molecule_dock_samples in self.multiple_docking_samples_object:

            n_clusters = molecule_dock_samples.clust_num

            cluster_labels = molecule_dock_samples.clusters_info[n_clusters]['labels']

            centers = molecule_dock_samples.clusters_info[n_clusters]['centers']

            if show_plots is True:
                # Create a subplot with 1 row and 2 columns
                X = molecule_dock_samples.pca_data

                fig.set_size_inches(18, 7)

                fig.set_size_inches(18, 7)

                # 2nd Plot showing the actual clusters formed
                # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)

                colors = plt.cm.Spectral(np.linspace(0, 1, len(self.multiple_docking_samples_object)))

                for frame_n in range(0, len(self.multiple_docking_samples_object)):
                    frame = self.multiple_docking_samples_object[frame_n].get_data_for_analysis()

                    frame_mol = frame['MolName']
                    tempX = X[X['MolName'] == frame_mol]
                    plt.scatter(tempX['component1'], tempX['component2'], marker='.', s=30, lw=0, alpha=0.7,
                                c=colors)

                    # Labeling the clusters
                    # Draw white circles at cluster centers
                    # plt.scatter(centers[:, 0], centers[:, 1],
                    #             marker='o', c="white", alpha=1, s=100)

                    # for i, c in enumerate(centers):
                    #     plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=100)

        plt.show()

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
