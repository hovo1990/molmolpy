# -*- coding: utf-8 -*-
__author__ = "Joaquim Viegas"

'''This code was taken from https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py, thanks for awesome code'''

""" JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices

dunn(k_list):
    Slow implementation of Dunn index that depends on numpy
    -- basec.pyx Cython implementation is much faster but flower than dunn_fast()
dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
davisbouldin(k_list, k_centers):
    Implementation of Davis Boulding index that depends on numpy
    -- basec.pyx Cython implementation is much faster
"""

import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from molmolpy.utils import helper as hlp


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(k_list):
    """ Dunn index [CVI]

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di


def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)


def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    # values = values[np.nonzero(values)]

    return np.max(values)


# This is good and works well
def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)

    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di


def big_s(x, center):
    len_x = len(x)
    total = 0

    for i in range(len_x):
        total += np.linalg.norm(x[i] - center)

    return total / len_x


def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k] - k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list - 1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l]) / d_eucs[k, l]
        for l in range(k + 1, len_k_list):
            values[l - 1] = (big_ss[k] + big_ss[l]) / d_eucs[k, l]

        db += np.max(values)
    res = db / len_k_list
    return res


# author:  Hovakim Grabski
import skbio
from skbio.stats.distance import permanova


# Taken from Practical Data Analysis Cookbook Tomasz Drabas
def pseudo_F(X, labels, centroids):
    '''
        The pseudo F statistic :
        pseudo F = [( [(T - PG)/(G - 1)])/( [(PG)/(n - G)])]
        The pseudo F statistic was suggested by
        Calinski and Harabasz (1974) in
        Calinski, T. and J. Harabasz. 1974.
            A dendrite method for cluster analysis.
            Commun. Stat. 3: 1-27.
            http://dx.doi.org/10.1080/03610927408827101

        We borrowed this code from
        https://github.com/scampion/scikit-learn/blob/master/
        scikits/learn/cluster/__init__.py

        However, it had an error so we altered how B is
        calculated.
    '''
    center = np.mean(X, axis=0)
    u, count = np.unique(labels, return_counts=True)

    B = np.sum([count[i] * ((cluster - center) ** 2)
                for i, cluster in enumerate(centroids)])

    X = X.as_matrix()
    W = np.sum([(x - centroids[labels[i]]) ** 2
                for i, x in enumerate(X)])

    k = len(centroids)
    n = len(X)

    return (B / (k - 1)) / (W / (n - k))


# Another version taken from Drabas book
def davis_bouldin(X, labels, centroids):
    '''
        The Davis-Bouldin statistic is an internal evaluation
        scheme for evaluating clustering algorithms. It
        encompasses the inter-cluster heterogeneity and
        intra-cluster homogeneity in one metric.

        The measure was introduced by
        Davis, D.L. and Bouldin, D.W. in 1979.
            A Cluster Separation Measure
            IEEE Transactions on Pattern Analysis and
            Machine Intelligence, PAMI-1: 2, 224--227

            http://dx.doi.org/10.1109/TPAMI.1979.4766909
    '''
    distance = np.array([
        np.sqrt(np.sum((x - centroids[labels[i]]) ** 2))
        for i, x in enumerate(X.as_matrix())])

    u, count = np.unique(labels, return_counts=True)

    Si = []

    for i, group in enumerate(u):
        Si.append(distance[labels == group].sum() / count[i])

    Mij = []

    for centroid in centroids:
        Mij.append([
            np.sqrt(np.sum((centroid - x) ** 2))
            for x in centroids])

    Rij = []
    for i in range(len(centroids)):
        Rij.append([
            0 if i == j
            else (Si[i] + Si[j]) / Mij[i][j]
            for j in range(len(centroids))])

    Di = [np.max(elem) for elem in Rij]

    return np.array(Di).sum() / len(centroids)


# PseudoF permanova calculation
# WIP
def pseudoF_permanova(points, labels):
    """ Statistical significance is assessed via a permutation test.
     The assignment of objects to groups (grouping) is randomly permuted a number of times
     (controlled via permutations). A pseudo-F statistic is computed for each permutation and the
     p-value is the proportion of
    permuted pseudo-F statisics that are equal to or greater than the original
     (unpermuted) pseudo-F statistic. (using sklearn pairwise euclidean_distance function)

    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = skbio.DistanceMatrix(points.as_matrix())
    ks = np.sort(np.unique(labels))

    pseudo_f = permanova(distances, labels)
    print(pseudo_f)
    return pseudo_f


# TODO part of silhuette analysis
import itertools

import hdbscan
import matplotlib
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from molmolpy.utils import converters

import os
import sys
import pickle

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture



def calculate_cluster_center_msm(original_data, cluster_labels):
    centers = []

    labels = cluster_labels
    unique_labels = list(set(cluster_labels))
    print('Unique labels ', unique_labels)

    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        # k == -1 then it is an outlier
        if k != -1:
            cluster_data = []
            xyz_pandas = original_data[labels == k]

            xyz = xyz_pandas.values
            shape_xyz = xyz.shape
            n_dim = shape_xyz[-1]

            temp_center = []
            for i in range(n_dim):
                temp = xyz[:, i]
                temp_center.append(np.mean(temp))
            # x = xyz[:, 0]
            # y = xyz[:, 1]

            # temp_center = [np.mean(x), np.mean(y)]
            centers.append(temp_center)

    return np.array(centers)



def calculate_cluster_center(original_data, cluster_labels):
    centers = []

    labels = cluster_labels
    unique_labels = list(set(cluster_labels))
    print('Unique labels ', unique_labels)

    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        # k == -1 then it is an outlier
        if k != -1:
            cluster_data = []
            xyz = original_data[labels == k]


            shape_xyz = xyz.shape
            n_dim = shape_xyz[-1]

            temp_center = []
            for i in range(n_dim):
                temp = xyz[:, i]
                temp_center.append(np.mean(temp))
            # x = xyz[:, 0]
            # y = xyz[:, 1]

            # temp_center = [np.mean(x), np.mean(y)]
            centers.append(temp_center)

    return np.array(centers)



def calculate_cluster_center_only_docking(original_data):
    centers = []

    cluster_data = []
    xyz = original_data


    shape_xyz = xyz.shape
    n_dim = shape_xyz[-1]

    temp_center = []
    temp1 = xyz[:,0]
    temp_center.append(np.mean(temp1))

    temp2 = xyz[:,1]
    temp_center.append(np.mean(temp2))
    # x = xyz[:, 0]
    # y = xyz[:, 1]

    # temp_center = [np.mean(x), np.mean(y)]
    centers = temp_center

    return centers


# TODO For docking
def calculate_cluster_center_docking(original_data, cluster_labels):
    centers = []

    labels = cluster_labels
    unique_labels = list(set(cluster_labels))
    print('Unique labels ', unique_labels)

    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        # k == -1 then it is an outlier
        if k != -1:
            cluster_data = []
            xyz = original_data[labels == k]


            shape_xyz = xyz.shape
            n_dim = shape_xyz[-1]

            temp_center = []
            temp1 = xyz['component1']
            temp_center.append(np.mean(temp1))

            temp2 = xyz['component2']
            temp_center.append(np.mean(temp2))
            # x = xyz[:, 0]
            # y = xyz[:, 1]

            # temp_center = [np.mean(x), np.mean(y)]
            centers.append(temp_center)

    return np.array(centers)



@hlp.timeit
def md_silhouette_analysis_pca(data, trajectory_time, range_n_clusters=list(range(1, 11)),
                               show_plots=False, algorithm='kmeans', connectivity=True,
                                   k_neighb=10, data_type='mdtraj', type='centroid'):
    sil_pca = []
    calinski_pca = []
    dunn_pca = []
    dbi_pca = []

    exhaust_book_dbi_pca = []
    exhaust_book_pseudoF_pca = []

    X = data
    traj = trajectory_time

    clusters_info = {}

    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.

        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        else:

            if connectivity is True:
                # connectivity matrix for structured Ward
                knn_graph = kneighbors_graph(X, k_neighb, include_self=False)
                # make connectivity symmetric
                knn_graph = 0.5 * (knn_graph + knn_graph.T)

                clusterer = AgglomerativeClustering(linkage='ward',
                                                    connectivity=knn_graph,
                                                    n_clusters=n_clusters)
            else:
                clusterer = AgglomerativeClustering(linkage='ward',
                                                    n_clusters=n_clusters)

        try:
            cluster_labels = clusterer.fit_predict(X)
        except Exception as e:
            print('Error in parallel_data_cluster_analysis: ', e)
            return


        if type == 'centroid':
            colors = sns.cubehelix_palette(n_colors=n_clusters, rot=.5, dark=0, light=0.85)
        elif type == 'reshape':
            colors = sns.cubehelix_palette(n_colors=n_clusters, start=2.8, rot=.1)

        colors_rgb = converters.convert_seaborn_color_to_rgb(colors)

        colors_ = colors
        colors_data = converters.convert_to_colordata(cluster_labels, colors)
        cluster_colors = colors_data


        test = 1

        if algorithm == 'kmeans':
            centers = clusterer.cluster_centers_
        else:
            # TODO this part needs to be modified
            centers = calculate_cluster_center(X, cluster_labels)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        test = 1

        # TODO serious bug here
        try:
            silhouette_avg = silhouette_score(X, cluster_labels)

            calinski_avg = calinski_harabaz_score(X, cluster_labels)

            # looks like this is ok
            dunn_avg = dunn_fast(X, cluster_labels)

            if data_type == 'mdtraj':
                converted_values = converters.convert_mdtraj_for_dbi_analysis(X, cluster_labels)
            elif data_type == 'docking':
                converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)

            david_bouldain = davisbouldin(converted_values, centers)

            # pseudo_f = pseudoF_permanova(X, cluster_labels)
            # print("For n_clusters =", n_clusters,
            #       "The pseudo_f is :", pseudo_f)


            # BOOK implementation of pseudoF and DBI
            try:
                book_dbi = davis_bouldin(X, cluster_labels, centers)
                book_pseudoF = pseudo_F(X, cluster_labels, centers)
            except Exception as e:
                # print('Error is book dbi and pseudoF')
                book_dbi = davis_bouldin(pd.DataFrame(X), cluster_labels, centers)
                book_pseudoF = pseudo_F(pd.DataFrame(X), cluster_labels, centers)

            sample_silhouette_values = silhouette_samples(X, cluster_labels)
        except Exception as e:
            print('Error in single clusterization is', e)
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

        # silhouette_avg = silhouette_score(X, cluster_labels)
        #
        # calinski_avg = calinski_harabaz_score(X, cluster_labels)
        #
        # # looks like this is ok
        # dunn_avg = dunn_fast(X, cluster_labels)
        #
        # if data_type == 'mdtraj':
        #     converted_values = converters.convert_mdtraj_for_dbi_analysis(X, cluster_labels)
        # elif data_type == 'docking':
        #     converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)
        # david_bouldain = davisbouldin(converted_values, centers)
        #
        # # pseudo_f = pseudoF_permanova(X, cluster_labels)
        # # print("For n_clusters =", n_clusters,
        # #       "The pseudo_f is :", pseudo_f)
        #
        # # BOOK implementation of pseudoF and DBI
        # book_dbi = davis_bouldin(pd.DataFrame(X), cluster_labels, centers)
        # book_pseudoF = pseudo_F(pd.DataFrame(X), cluster_labels, centers)

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

        sil_pca.append(silhouette_avg)
        calinski_pca.append(calinski_avg)
        dunn_pca.append(dunn_avg)
        dbi_pca.append(david_bouldain)

        # TODO test cluster analysis using book algorithms
        exhaust_book_dbi_pca.append(book_dbi)
        exhaust_book_pseudoF_pca.append(book_pseudoF)

        # Compute the silhouette scores for each sample
        # sample_silhouette_values = silhouette_samples(X, cluster_labels)

        #
        # colors_rgb = converters.convert_seaborn_color_to_rgb(colors)
        #
        # colors_ = colors
        # cluster_labels = clusters_info[n_clusters]['labels']
        # colors_data = converters.convert_to_colordata(cluster_labels, colors)
        # cluster_colors = colors_data




        clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                           'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                           'labels': cluster_labels, 'centers': centers,
                                           'silhouette_values': sample_silhouette_values,
                                           'book_dbi': book_dbi,
                                           'book_pseudoF': book_pseudoF,
                                           'colors': colors,
                                           'rgbColors': colors_rgb,
                                           'colorData': cluster_colors,
                                           }})

        # if show_plots is True:
        #     sns.set(style="ticks", context='paper')
        #
        #     # Create a subplot with 1 row and 2 columns
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     fig.set_size_inches(18, 7)
        #
        #     fig.set_size_inches(18, 7)
        #
        #     # The 1st subplot is the silhouette plot
        #     # The silhouette coefficient can range from -1, 1 but in this example all
        #     # lie within [-0.1, 1]
        #     ax1.set_xlim([-1, 1])
        #     # The (n_clusters+1)*10 is for inserting blank space between silhouette
        #     # plots of individual clusters, to demarcate them clearly.
        #     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        #
        #     unique_labels = list(set(cluster_labels))
        #
        #     # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #
        #     colors = iter(sns.color_palette("cubehelix", len(unique_labels)))
        #
        #     y_lower = 10
        #
        #     for i in range(n_clusters):
        #         # Aggregate the silhouette scores for samples belonging to
        #         # cluster i, and sort them
        #         ith_cluster_silhouette_values = \
        #             sample_silhouette_values[cluster_labels == i]
        #
        #         ith_cluster_silhouette_values.sort()
        #
        #         size_cluster_i = ith_cluster_silhouette_values.shape[0]
        #         y_upper = y_lower + size_cluster_i
        #
        #         color = cm.spectral(float(i) / n_clusters)
        #         ax1.fill_betweenx(np.arange(y_lower, y_upper),
        #                           0, ith_cluster_silhouette_values,
        #                           facecolor=color, edgecolor=color, alpha=0.7)
        #
        #         # Label the silhouette plots with their cluster numbers at the middle
        #         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #
        #         # Compute the new y_lower for next plot
        #         y_lower = y_upper + 10  # 10 for the 0 samples
        #
        #     ax1.set_title("The silhouette plot for the various clusters.")
        #     ax1.set_xlabel("The silhouette coefficient values")
        #     ax1.set_ylabel("Cluster label")
        #
        #     # The vertical line for average silhouette score of all the values
        #     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        #
        #     ax1.set_yticks([])  # Clear the yaxis labels / ticks
        #     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        #
        #     # 2nd Plot showing the actual clusters formed
        #     # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #     # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        #     #             c=colors)
        #
        #
        #
        #     for z in unique_labels:  # Need to modify WORKS
        #         # print('k is ',k)
        #         # k == -1 then it is an outlier
        #         if z != -1:
        #             data = X[cluster_labels == z]
        #             ax2.scatter(data[:, 0], data[:, 1], marker='o', s=10, lw=0, alpha=0.75,
        #                         c=next(colors))
        #             # binding_energy = cluster_energy['BindingEnergy']
        #             # sns.distplot(binding_energy,  ax=ax[k][1])
        #             # ax[k][1].hist(binding_energy, normed=False, color=colors[z], alpha=0.3)
        #
        #     # cbar = ax2.colorbar()
        #     # cbar.set_label('Time [ps]')
        #     # fig.subplots_adjust(right=0.8)
        #     # fig.colorbar(traj, ax=ax2)
        #
        #     # Labeling the clusters
        #     # Draw white circles at cluster centers
        #     ax2.scatter(centers[:, 0], centers[:, 1],
        #                 marker='o', c="white", alpha=1, s=100)
        #
        #     for i, c in enumerate(centers):
        #         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=100)
        #
        #     ax2.set_title("The visualization of the clustered data.")
        #     ax2.set_xlabel("Feature space for the 1st feature")
        #     ax2.set_ylabel("Feature space for the 2nd feature")
        #
        #     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
        #                   "with n_clusters = %d" % n_clusters),
        #                  fontsize=14, fontweight='bold')
        #
        #     plt.show()

    return clusters_info


@hlp.timeit
def extract_info_cluster_data(cluster_data, key, range_n_clusters):
    temp_data = []
    if cluster_data is None:
        temp_data.append(1)
        return temp_data

    for clust_num in range_n_clusters:
        temp_data.append(cluster_data[clust_num][key])
    return temp_data


@hlp.timeit
def select_number_of_clusters_v2(clusters_info,
                                 range_n_clusters):
    silhouette_data = extract_info_cluster_data(clusters_info, 'silhouette', range_n_clusters)
    calinski_data = extract_info_cluster_data(clusters_info, 'calinski', range_n_clusters)
    dunn_data = extract_info_cluster_data(clusters_info, 'dunn', range_n_clusters)
    dbi_data = extract_info_cluster_data(clusters_info, 'dbi', range_n_clusters)

    book_dbi_data = extract_info_cluster_data(clusters_info, 'book_dbi', range_n_clusters)

    import scipy

    from scipy import signal

    import heapq

    max_num_list = 3
    silhouette_array = np.array(silhouette_data)
    top_sil = heapq.nlargest(max_num_list, range(len(silhouette_array)), silhouette_array.take)
    print('Sil ', top_sil)

    calinski_array = np.array(calinski_data)
    top_calinski = heapq.nlargest(max_num_list, range(len(calinski_array)), calinski_array.take)

    dunn_array = np.array(dunn_data)
    top_dunn = heapq.nlargest(max_num_list, range(len(dunn_array)), dunn_array.take)

    dbi_array = np.array(dbi_data)
    top_dbi = heapq.nsmallest(max_num_list, range(len(dbi_array)), dbi_array.take)

    p = [top_dbi, top_calinski, top_dunn, top_sil]

    print('Sil ', top_sil)
    print('Calinski ', top_calinski)
    print('Dunn ', top_dunn)
    print('DBI ', top_dbi)

    # TODO very important score values
    # score_values = [1, 0.85, 0.35]
    score_values = [1, 0.75, 0.5]

    cluster_scores = {str(x): 0.0 for x in range_n_clusters}

    # TODO this is checked up so now what?

    for temp_index in range(len(score_values)):
        clust_index = range_n_clusters[top_sil[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_calinski[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_dunn[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_dbi[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

    test = 1
    print('Cluster scores is ', cluster_scores)

    # # TODO things don't work as expected
    # result = set(p[0])
    # for s in p[1:]:
    #     result.intersection_update(s)
    # print(result)
    #
    #
    #
    # cluster_quantity = []
    # for i in result:
    #     cluster_quantity.append(range_n_clusters[i])

    # TODO don't forget
    cluster_quantity = []
    for i in p:
        for x in i:
            cluster_quantity.append(range_n_clusters[x])

    print('------------------------------------------------')
    print('verify yolo', cluster_quantity)

    cluster_set = set(cluster_quantity)

    cluster_dict = {}
    for n_set in cluster_set:
        count = cluster_quantity.count(n_set)
        cluster_dict.update({n_set: count})

    print('verify yolo ', cluster_dict)

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
    print('Frequent indexes is ', clust_num_pre)

    import numpy
    def median(lst):
        return numpy.median(numpy.array(lst))

    clust_num = sorted(clust_num_pre)[len(clust_num_pre) // 2]

    max_index = 0
    max_score = 0
    # This one is for most frequent
    # for index_yay in clust_num_pre:

    #This one is for best score :)
    for index_yay in range_n_clusters:
        score = cluster_scores[str(index_yay)]
        if score > max_score:
            max_score = score
            max_index = index_yay

    clust_num = max_index

    print("V2 Number of clusters is ", clust_num)

    clust_info = {'clustNum': clust_num,
                  'silhouette': silhouette_data,
                  'calinski': calinski_data,
                  'dbi': dbi_data,
                  'dunn': dunn_data,
                  'dbiBook': book_dbi_data}

    return clust_num, clust_info



@hlp.timeit
def select_number_of_clusters_v2_docking(clusters_info, type, range_n_clusters):
    silhouette_data = extract_info_cluster_data(clusters_info, 'silhouette', range_n_clusters)
    calinski_data = extract_info_cluster_data(clusters_info, 'calinski', range_n_clusters)
    dunn_data = extract_info_cluster_data(clusters_info, 'dunn', range_n_clusters)
    dbi_data = extract_info_cluster_data(clusters_info, 'dbi', range_n_clusters)

    book_dbi_data = extract_info_cluster_data(clusters_info, 'book_dbi', range_n_clusters)

    import scipy

    from scipy import signal

    import heapq

    max_num_list = 3
    silhouette_array = np.array(silhouette_data)
    top_sil = heapq.nlargest(max_num_list, range(len(silhouette_array)), silhouette_array.take)
    print('Sil ', top_sil)

    calinski_array = np.array(calinski_data)
    top_calinski = heapq.nlargest(max_num_list, range(len(calinski_array)), calinski_array.take)

    dunn_array = np.array(dunn_data)
    top_dunn = heapq.nlargest(max_num_list, range(len(dunn_array)), dunn_array.take)

    dbi_array = np.array(dbi_data)
    top_dbi = heapq.nsmallest(max_num_list, range(len(dbi_array)), dbi_array.take)

    p = [top_dbi, top_calinski, top_dunn, top_sil]

    print('Sil ', top_sil)
    print('Calinski ', top_calinski)
    print('Dunn ', top_dunn)
    print('DBI ', top_dbi)

    # TODO very important score values
    # score_values = [1, 0.85, 0.35]
    score_values = [1, 0.75, 0.5]

    cluster_scores = {str(x): 0.0 for x in range_n_clusters}

    for temp_index in range(len(score_values)):
        clust_index = range_n_clusters[top_sil[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_calinski[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_dunn[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

        clust_index = range_n_clusters[top_dbi[temp_index]]
        cluster_scores[str(clust_index)] += score_values[temp_index]

    test = 1
    print('Cluster scores is ', cluster_scores)

    # # TODO things don't work as expected
    # result = set(p[0])
    # for s in p[1:]:
    #     result.intersection_update(s)
    # print(result)
    #
    #
    #
    # cluster_quantity = []
    # for i in result:
    #     cluster_quantity.append(range_n_clusters[i])

    # TODO don't forget
    cluster_quantity = []
    for i in p:
        for x in i:
            cluster_quantity.append(range_n_clusters[x])

    print('------------------------------------------------')
    print('verify yolo', cluster_quantity)

    cluster_set = set(cluster_quantity)

    cluster_dict = {}
    for n_set in cluster_set:
        count = cluster_quantity.count(n_set)
        cluster_dict.update({n_set: count})

    print('verify yolo ', cluster_dict)

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
    print('Frequent indexes is ', clust_num_pre)

    import numpy
    def median(lst):
        return numpy.median(numpy.array(lst))

    clust_num = sorted(clust_num_pre)[len(clust_num_pre) // 2]

    max_index = 0
    max_score = 0
    # This one is for most frequent
    # for index_yay in clust_num_pre:

    # # TODO VERY TRICKY PART?
    if type=='centroid':

        #This one is for best score :)
        for index_yay in range_n_clusters:
            score = cluster_scores[str(index_yay)]
            if score >= max_score:
                max_score = score
                max_index = index_yay

        clust_num = max_index

        clust_info = {'clustNum': clust_num,
                      'silhouette': silhouette_data,
                      'calinski': calinski_data,
                      'dbi': dbi_data,
                      'dunn': dunn_data,
                      'dbiBook': book_dbi_data}
    elif type=='reshape':

        from collections import Counter

        d = Counter(cluster_scores)
        print(d.most_common())

        for k, v in d.most_common(3):
            print('%s: %i' % (k, v))

        top_three = d.most_common()[0:3]



        # clust_num = max_index
        clust_num = min([int(i[0]) for i in top_three])


    print("V2 Number of clusters is ", clust_num)

    clust_info = {'clustNum': clust_num,
                  'silhouette': silhouette_data,
                  'calinski': calinski_data,
                  'dbi': dbi_data,
                  'dunn': dunn_data,
                  'dbiBook': book_dbi_data}

    return clust_num, clust_info



@hlp.timeit
def select_number_of_clusters_v2_old(clusters_info, range_n_clusters):
    silhouette_data = extract_info_cluster_data(clusters_info, 'silhouette', range_n_clusters)
    calinski_data = extract_info_cluster_data(clusters_info, 'calinski', range_n_clusters)
    dunn_data = extract_info_cluster_data(clusters_info, 'dunn', range_n_clusters)
    dbi_data = extract_info_cluster_data(clusters_info, 'dbi', range_n_clusters)

    book_dbi_data = extract_info_cluster_data(clusters_info, 'book_dbi', range_n_clusters)

    import scipy

    from scipy import signal

    import heapq

    max_num_list = 3
    silhouette_array = np.array(silhouette_data)
    top_sil = heapq.nlargest(max_num_list, range(len(silhouette_array)), silhouette_array.take)
    print('Sil ', top_sil)

    calinski_array = np.array(calinski_data)
    top_calinski = heapq.nlargest(max_num_list, range(len(calinski_array)), calinski_array.take)

    dunn_array = np.array(dunn_data)
    top_dunn = heapq.nlargest(max_num_list, range(len(dunn_array)), dunn_array.take)

    dbi_array = np.array(dbi_data)
    top_dbi = heapq.nsmallest(max_num_list, range(len(dbi_array)), dbi_array.take)

    p = [top_dbi, top_calinski, top_dunn, top_sil]

    print('Sil ', top_sil)
    print('Calinski ', top_calinski)
    print('Dunn ', top_dunn)
    print('DBI ', top_dbi)

    # # TODO things don't work as expected
    # result = set(p[0])
    # for s in p[1:]:
    #     result.intersection_update(s)
    # print(result)
    #
    #
    #
    # cluster_quantity = []
    # for i in result:
    #     cluster_quantity.append(range_n_clusters[i])

    # TODO don't forget
    cluster_quantity = []
    for i in p:
        for x in i:
            cluster_quantity.append(range_n_clusters[x])

    print('------------------------------------------------')
    print('verify yolo', cluster_quantity)

    cluster_set = set(cluster_quantity)

    cluster_dict = {}
    for n_set in cluster_set:
        count = cluster_quantity.count(n_set)
        cluster_dict.update({n_set: count})

    print('verify yolo ', cluster_dict)

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

    print("V2 Number of clusters is ", clust_num)

    clust_info = {'clustNum': clust_num,
                  'silhouette': silhouette_data,
                  'calinski': calinski_data,
                  'dbi': dbi_data,
                  'dunn': dunn_data,
                  'dbiBook': book_dbi_data}

    return clust_num, clust_info


@hlp.timeit
def select_number_of_clusters(clusters_info, range_n_clusters):
    silhouette_data = extract_info_cluster_data(clusters_info, 'silhouette', range_n_clusters)
    calinski_data = extract_info_cluster_data(clusters_info, 'calinski', range_n_clusters)
    dunn_data = extract_info_cluster_data(clusters_info, 'dunn', range_n_clusters)
    dbi_data = extract_info_cluster_data(clusters_info, 'dbi', range_n_clusters)

    book_dbi_data = extract_info_cluster_data(clusters_info, 'book_dbi', range_n_clusters)

    # ["foo", "bar", "baz"].index("bar")
    max_silhouette = max(silhouette_data)
    max_dunn = max(dunn_data)
    min_dbi = min(dbi_data)

    sil_index = silhouette_data.index(max_silhouette)
    dunn_index = dunn_data.index(max_dunn)
    dbi_index = dbi_data.index(min_dbi)

    cluster_quantity = []
    cluster_quantity.append(range_n_clusters[sil_index])
    cluster_quantity.append(range_n_clusters[dunn_index])
    cluster_quantity.append(range_n_clusters[dbi_index])

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

    clust_info = {'clustNum': clust_num,
                  'silhouette': silhouette_data,
                  'calinski': calinski_data,
                  'dbi': dbi_data,
                  'dunn': dunn_data,
                  'dbiBook': book_dbi_data}

    return clust_num, clust_info


# @hlp.timeit
def parallel_data_cluster_analysis(n_clusters, data, trajectory_time=None,
                                   algorithm='kmeans', connectivity=True,
                                   k_neighb=10, data_type='mdtraj'):
    sil_pca = []
    calinski_pca = []
    dunn_pca = []
    dbi_pca = []

    exhaust_book_dbi_pca = []
    exhaust_book_pseudoF_pca = []

    X = data

    traj = trajectory_time

    clusters_info = {}

    if algorithm == 'kmeans':
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10  )
        clusterer = KMeans(n_clusters=n_clusters, random_state=10 , n_init=300, max_iter=2000)
    else:

        # TODO connectivity requires knn graph
        if connectivity is True:
            # connectivity matrix for structured Ward
            knn_graph = kneighbors_graph(X, k_neighb, include_self=False)
            # make connectivity symmetric
            knn_graph = 0.5 * (knn_graph + knn_graph.T)

            clusterer = AgglomerativeClustering(linkage='ward',
                                                connectivity=knn_graph,
                                                n_clusters=n_clusters)
        else:
            clusterer = AgglomerativeClustering(linkage='ward',
                                                n_clusters=n_clusters)

    try:
        cluster_labels = clusterer.fit_predict(X)
    except Exception as e:
        print('verify yolo Error in parallel_data_cluster_analysis: ', e)
        return

    if algorithm == 'kmeans':
        centers = clusterer.cluster_centers_
    else:
        # TODO this part needs to be modified
        centers = calculate_cluster_center(X, cluster_labels)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # TODO serious bug here
    try:
        silhouette_avg = silhouette_score(X, cluster_labels)

        calinski_avg = calinski_harabaz_score(X, cluster_labels)

        # looks like this is ok
        dunn_avg = dunn_fast(X, cluster_labels)

        if data_type == 'mdtraj':
            converted_values = converters.convert_mdtraj_for_dbi_analysis(X, cluster_labels)
        elif data_type == 'docking':
            converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)

        david_bouldain = davisbouldin(converted_values, centers)

        # pseudo_f = pseudoF_permanova(X, cluster_labels)
        # print("For n_clusters =", n_clusters,
        #       "The pseudo_f is :", pseudo_f)


        # BOOK implementation of pseudoF and DBI
        try:
            book_dbi = davis_bouldin(X, cluster_labels, centers)
            book_pseudoF = pseudo_F(X, cluster_labels, centers)
        except Exception as e:
            # print('Error is book dbi and pseudoF')
            book_dbi = davis_bouldin(pd.DataFrame(X), cluster_labels, centers)
            book_pseudoF = pseudo_F(pd.DataFrame(X), cluster_labels, centers)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    except Exception as e:
        print('BADOOM Error in parallel clusterization scores is', e)
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

    # pseudo_f = pseudoF_permanova(X, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The pseudo_f is :", pseudo_f)

    # BOOK implementation of pseudoF and DBI

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

    sil_pca.append(silhouette_avg)
    calinski_pca.append(calinski_avg)
    dunn_pca.append(dunn_avg)
    dbi_pca.append(david_bouldain)

    # TODO test cluster analysis using book algorithms
    exhaust_book_dbi_pca.append(book_dbi)
    exhaust_book_pseudoF_pca.append(book_pseudoF)

    # Compute the silhouette scores for each sample
    # sample_silhouette_values = silhouette_samples(X, cluster_labels)

    clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                       'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                       'labels': cluster_labels, 'centers': centers,
                                       'silhouette_values': sample_silhouette_values,
                                       'book_dbi': book_dbi,
                                       'book_pseudoF': book_pseudoF}})

    return clusters_info





# @hlp.timeit
def parallel_data_cluster_analysis_msm(n_clusters, data, trajectory_time=None,
                                   algorithm='kmeans', connectivity=True,
                                   k_neighb=10, data_type='mdtraj'):
    sil_pca = []
    calinski_pca = []
    dunn_pca = []
    dbi_pca = []

    exhaust_book_dbi_pca = []
    exhaust_book_pseudoF_pca = []

    X = data

    traj = trajectory_time

    clusters_info = {}

    if algorithm == 'kmeans':
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10  )
        clusterer = KMeans(n_clusters=n_clusters, random_state=10 , n_init=300, max_iter=2000)
    else:

        # TODO connectivity requires knn graph
        if connectivity is True:
            # connectivity matrix for structured Ward
            knn_graph = kneighbors_graph(X, k_neighb, include_self=False)
            # make connectivity symmetric
            knn_graph = 0.5 * (knn_graph + knn_graph.T)

            clusterer = AgglomerativeClustering(linkage='ward',
                                                connectivity=knn_graph,
                                                n_clusters=n_clusters)
        else:
            clusterer = AgglomerativeClustering(linkage='ward',
                                                n_clusters=n_clusters)

    try:
        cluster_labels = clusterer.fit_predict(X)
    except Exception as e:
        print('verify yolo Error in parallel_data_cluster_analysis: ', e)
        return

    if algorithm == 'kmeans':
        centers = clusterer.cluster_centers_
    else:
        # TODO this part needs to be modified
        centers = calculate_cluster_center_msm(X, cluster_labels)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # TODO serious bug here
    try:
        silhouette_avg = silhouette_score(X, cluster_labels)

        calinski_avg = calinski_harabaz_score(X, cluster_labels)

        # looks like this is ok
        dunn_avg = dunn_fast(X, cluster_labels)

        if data_type == 'mdtraj':
            converted_values = converters.convert_mdtraj_for_dbi_analysis(X, cluster_labels)
        elif data_type == 'docking':
            converted_values = converters.convert_pandas_for_dbi_analysis(X, cluster_labels)

        david_bouldain = davisbouldin(converted_values, centers)

        # pseudo_f = pseudoF_permanova(X, cluster_labels)
        # print("For n_clusters =", n_clusters,
        #       "The pseudo_f is :", pseudo_f)


        # BOOK implementation of pseudoF and DBI
        try:
            book_dbi = davis_bouldin(X, cluster_labels, centers)
            book_pseudoF = pseudo_F(X, cluster_labels, centers)
        except Exception as e:
            # print('Error is book dbi and pseudoF')
            book_dbi = davis_bouldin(pd.DataFrame(X), cluster_labels, centers)
            book_pseudoF = pseudo_F(pd.DataFrame(X), cluster_labels, centers)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    except Exception as e:
        print('BADOOM Error in parallel clusterization scores is', e)
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

    # pseudo_f = pseudoF_permanova(X, cluster_labels)
    # print("For n_clusters =", n_clusters,
    #       "The pseudo_f is :", pseudo_f)

    # BOOK implementation of pseudoF and DBI

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

    sil_pca.append(silhouette_avg)
    calinski_pca.append(calinski_avg)
    dunn_pca.append(dunn_avg)
    dbi_pca.append(david_bouldain)

    # TODO test cluster analysis using book algorithms
    exhaust_book_dbi_pca.append(book_dbi)
    exhaust_book_pseudoF_pca.append(book_pseudoF)

    # Compute the silhouette scores for each sample
    # sample_silhouette_values = silhouette_samples(X, cluster_labels)

    clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
                                       'calinski': calinski_avg, 'silhouette': silhouette_avg,
                                       'labels': cluster_labels, 'centers': centers,
                                       'silhouette_values': sample_silhouette_values,
                                       'book_dbi': book_dbi,
                                       'book_pseudoF': book_pseudoF}})

    return clusters_info



