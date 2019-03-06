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

# from . import molecule_object
import seaborn as sns
import matplotlib.pyplot as plt
from molmolpy.utils import helper as hlp

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import protein_analysis
from molmolpy.utils import nucleic_analysis

import webcolors

sns.set(style="darkgrid")


# import pylab as plt

def custom_palplot_vertical(pal, size=1, ylabel = [], xlabels=[]):
    """Plot the MMPBSA residue energy values in vertical array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot

    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(3*size, n*size))

    working_cmap =mpl.colors.ListedColormap(list(pal))

    #my_cmap2 = mpl.colors.LinearSegmentedColormap('my_colormap2', list(pal), 256)

    # ax.imshow(np.arange(n).reshape(n, 1),
    #           cmap=working_cmap,
    #           interpolation="nearest", aspect="auto", origin='lower')

    ax.imshow(np.arange(n).reshape(n, 1),
              cmap=working_cmap,
              interpolation="nearest", origin='lower')

    ax.set_aspect(1.5)
    # ax.set_xticks(np.arange(n) - .5)
    # ax.set_yticks([-.5, .5])

    ax.set_xticks([-.5, .5])
    # ax.set_yticks(np.arange(n) - .5)
    ax.set_yticks(np.arange(n))


    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabel)

    ax.yaxis.tick_right()

    f.savefig('vertical_color.png', dpi=600)



def custom_palplot(pal, size=1):
    """Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot

    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])



def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def get_cluster_color_name(values):
    new_values = []
    for i in values:
        new_values.append(int(i))

    actual_name, closest_name = get_colour_name(new_values)

    print('Actual Name ', actual_name)
    print('Closest Name ', closest_name)

    return actual_name, closest_name


def plot_cluster_analysis(cluster_range, criteria_data, criteria_name, score_text='Test'):
    plt.clf()
    plt.scatter(cluster_range, criteria_data, marker='o', c='b', s=200)
    plt.plot(cluster_range, criteria_data, ':k', linewidth=3.0)
    plt.ylabel(criteria_name)
    plt.xlim(cluster_range[0], cluster_range[-1])
    plt.xlabel('n of clusters\n{0}'.format(score_text))
    # plt.figtext(.4, .02, score_text, fontsize=22)
    plt.show()


def change_ax_plot_font_size(ax, fontsize=10):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)



@hlp.timeit
def show_best_data_analysis(object_to_work,
                            clusters_info,
                            clust_num_analysis,
                            title='contact_maps_plot.png',
                            cust_clust=None,
                            show_plot=False,
                            custom_dpi=600):

    print('Show best data analysis is called')
    # self.clusters_info.update({n_clusters: {'dunn': dunn_avg, 'dbi': david_bouldain,
    #                                         'calinski': calinski_avg, 'silhouette': silhouette_avg,
    #                                         'labels': cluster_labels, 'centers': centers,
    #                                         'silhouette_values': sample_silhouette_values}})

    if cust_clust is not None:
        n_clusters = cust_clust
    else:
        n_clusters = clust_num_analysis['clustNum']

    cluster_labels = clusters_info[n_clusters]['labels']
    sample_silhouette_values = clusters_info[n_clusters]['silhouette_values']
    silhouette_avg = clusters_info[n_clusters]['silhouette']

    centers = clusters_info[n_clusters]['centers']

    indexes = object_to_work['indexes_final']
    distances = object_to_work['distancesFinal']
    X = indexes

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
    # self.colors_ = colors

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

    # self.cluster_colors = colors

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

    fig.savefig(title, dpi=custom_dpi,
                bbox_inches='tight')
    if show_plot is True:
        plt.show()



@hlp.timeit
def show_data_cluster_analysis_plots(clust_num_analysis,
                                    range_n_clusters,
                                    show_plot=False,
                                     title='CLuster_quality.png',
                                    custom_dpi=600):
    # Create a subplot with 2 row and 2 columns
    # fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)




    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                 2)  # sharex='col', sharey='row') TODO this can be used for shared columns
    fig.set_size_inches(20, 20)

    cluster_range = range_n_clusters
    score =  clust_num_analysis['dbi']
    criteria_name = 'Davis-Bouldain Index'
    score_text = 'The optimal clustering solution\n' \
                 ' has the smallest Davies-Bouldin index value.'
    ax1.scatter(cluster_range, score, marker='o', c='b', s=200)
    ax1.plot(cluster_range, score, ':k', linewidth=3.0)

    ax1.set_xlim(cluster_range[0], cluster_range[-1])

    ax1.set_title(score_text)
    ax1.set_xlabel('n of clusters')
    ax1.set_ylabel(criteria_name)

    cluster_range = range_n_clusters
    score = clust_num_analysis['dunn']
    criteria_name = "Dunn's Index"
    score_text = "Maximum value of the index\n" \
                 "represents the right partitioning given the index"
    ax2.scatter(cluster_range, score, marker='o', c='b', s=200)
    ax2.plot(cluster_range, score, ':k', linewidth=3.0)

    ax2.set_xlim(cluster_range[0], cluster_range[-1])
    ax2.set_title(score_text)
    ax2.set_xlabel('n of clusters')
    ax2.set_ylabel(criteria_name)

    cluster_range = range_n_clusters
    score = clust_num_analysis['silhouette']
    criteria_name = 'Mean Silhouette Coefficient for all samples'
    score_text = 'Objects with a high silhouette\n' \
                 'value are considered well clustered'
    ax3.scatter(cluster_range, score, marker='o', c='b', s=200)
    ax3.plot(cluster_range, score, ':k', linewidth=3.0)

    ax3.set_xlim(cluster_range[0], cluster_range[-1])
    ax3.set_title(score_text)
    ax3.set_xlabel('n of clusters')
    ax3.set_ylabel(criteria_name)

    cluster_range = range_n_clusters
    score = clust_num_analysis['calinski']
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

    fig.savefig(title, dpi=custom_dpi, bbox_inches='tight')