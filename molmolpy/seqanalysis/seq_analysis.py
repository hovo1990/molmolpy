# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    seq_analysis.py
# @brief   sequence analysis object (identify cnserved regions)
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
import math
import random


# this imports biopython
import Bio

import  Bio.PDB.Polypeptide as bio_prot_tools

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
import multiprocessing

import rpy2.robjects.packages as rpackages
import rpy2.robjects.functions as rfunctions
import rpy2.robjects.vectors as rvectors
from rpy2.robjects import r, pandas2ri


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

from molmolpy.utils import helper as hlp

from itertools import combinations

import seaborn as sns

# import numba

matplotlib.rcParams.update({'font.size': 12})
# matplotlib.style.use('ggplot')
sns.set(style="white", context='paper')


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)



class SeqAnalysisObject(object):
    """

    Usage example


        >>> from molmolpy.seqanalysis import seq_analysis

        >>> from itertools import combinations
        >>>
        >>> import seaborn as sns
        >>>
        >>> import numba
        >>>
        >>>
        >>>
        >>>
        >>> # filename = "/media/Work/MEGA/Programming/molmolpy/molmolpy/seqanalysis/luxR_bacteria.fasta"
        >>> # filename = "/media/Work/MEGA/Programming/molmolpy/molmolpy/seqanalysis/luxR_bacteria_shortNames.fasta"
        >>>
        >>> # filename = "luxR_bacteria_uberTest.fasta"
        >>> #filename = "/media/Work/MEGA/Programming/molmolpy/molmolpy/seqanalysis/test1.fasta"
        >>>
        >>> # filename = "/media/Work/MEGA/Programming/molmolpy/molmolpy/seqanalysis/pseudomonas.fasta"
        >>>
        >>> # filename = "luxR_bacteria_uberShortNames.fasta"
        >>> filename = "luxR_bacteria_superShortNames.fasta"
        >>>
        >>>
        >>> # filename = "uniprot-lasr+and+length%3A239.fasta"
        >>> HSL_analysis_object = seq_analysis.SeqAnalysisObject(filename)
        >>>
        >>>
        >>> filename_1 = 'HSL_simulation_cluster2.pickle'
        >>> filename_2 = 'HSL_simulation_cluster3.pickle'
        >>>
        >>>
        >>> HSL_analysis_object.pickle_load_mmpbsa_analysis(filename_1)
        >>> HSL_analysis_object.pickle_load_mmpbsa_analysis(filename_2)
        >>>
        >>>
        >>> # HSL_analysis_object.pretty_print_basic_all()
        >>>
        >>> HSL_analysis_object.pretty_print_ligand_all()
        >>>
        >>> HSL_analysis_object.pretty_print_ligand_specific()
        >>> test = 1


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
    def __init__(self, msa_sequence,
                 load_way='molmolpy',
                 molname='Unknown',
                 receptor_name='Unknown'):


        self.msa_sequence = msa_sequence

        self.alignment_method_dict = {}


        self.analyzed_mmpbsa_data = {}

        self.prep_msa_library()

        self.colors_texshade = {1:'Violet',
                                2:'Blue',
                                3:'Black',
                                4:'Red',
                                5:'Green'}


        # DO now dynamics coloring?
        # TODO NEED TO MAKE THIS DYNAMICS
        # color_list = ['Black', 'Red', 'Lime', 'Blue', 'Yellow','Cyan', 'Magenta','Silver', 'Gray','Maroon','Olive',
        #               'Green','Purple','Teal','Navy']

        # USE LATEX COLORS
        color_list = ['Apricot', 'Bittersweet', 'Blue', 'Yellow','BlueViolet', 'Brown','CadetBlue', 'DarkOrchid','Green','OliveGreen',
                      'Green','Purple','RoyalPurple','RedOrange']


        color_shuffle = color_list[:]
        random.shuffle(color_shuffle)

        self.colors_texshade = {ind:color_shuffle[ind] for ind in range(len(color_shuffle))}

        test = 1

        #self.colors_texshade = self.generate_texshade_colors()



    def generate_texshade_colors(self):
        test = 1

        colors = sns.color_palette("Paired", 8)

        rgb_colors = converters.convert_seaborn_color_to_rgb(colors)

        for rgb in rgb_colors:
            actual_name, closest_name = plot_tools.get_cluster_color_name(rgb)

        test = 1


    def pickle_load_mmpbsa_analysis(self, filename):
        analysed_data = pickle.load(open(filename, "rb"))

        key = list(analysed_data.keys())[0]

        if key not in self.analyzed_mmpbsa_data:
            test = 1
            self.analyzed_mmpbsa_data.update(analysed_data)
        print('test')

    def custom_change_key_pickle_load_mmpbsa_analysis(self, filename, change_to):
        analysed_data = pickle.load(open(filename, "rb"))

        key = list(analysed_data.keys())[0]

        new_key = change_to

        new_analyzed_data = {new_key:analysed_data[key]}


        self.analyzed_mmpbsa_data.update(new_analyzed_data)
        print('test')



    def prep_msa_library(self, lower_bound=65):
        '''
        Use rpy to perform msa using MSA library from R.

        MSA using ClustalW, ClustalOmega and Muscle

        Obtain Consensus sequences.

        :return:
        '''


        self.r_utils = rpackages.importr('utils')
        self.r_base = rpackages.importr('base')
        self.r_msa = rpackages.importr('msa')
        self.r_tools = rpackages.importr('tools')
        self.r_biostrings = rpackages.importr('Biostrings')

        test = 1

        self.r_sequences = self.r_biostrings.readAAStringSet(self.msa_sequence)


        # Threshold boundary for consensus sequence
        r_upperlower_int = rvectors.FloatVector([100, lower_bound])

        # pandas2ri.activate()

        print('------------------------------------------------------------------------------')
        print('ClustalW alignment->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->\n')

        self.clustal_w_alignment = self.r_msa.msa(self.r_sequences, method='ClustalW')

        # TODO this part is very sensitive
        r_upperlower_int = rvectors.FloatVector([100, lower_bound])

        # self.clustal_w_consensus = pandas2ri.ri2py(self.r_msa.msaConsensusSequence(self.clustal_w_alignment,
        #                                                                type="upperlower",
        #                                                                thresh=r_upperlower_int
        #                                                                ))

        self.clustal_w_consensus_no_conv = self.r_msa.msaConsensusSequence(self.clustal_w_alignment,
                                                                       type="upperlower",
                                                                       thresh=r_upperlower_int
                                                                       )

        self.clustal_w_consensus = pandas2ri.ri2py(self.clustal_w_consensus_no_conv)


        test = 1

        self.r_msa.print(self.clustal_w_alignment, show="complete")

        print('------------------------------------------------------------------------------')
        print('ClustalOmega alignment->->->->->->->->->->->->->->->->->->->->->->->->->->->->\n')

        self.clustal_omega_alignment = self.r_msa.msa(self.r_sequences, method='ClustalOmega')

        self.clustal_omega_consensus_no_conv = self.r_msa.msaConsensusSequence(self.clustal_omega_alignment,
                                                                           type="upperlower",
                                                                          thresh=r_upperlower_int )

        self.clustal_omega_consensus = pandas2ri.ri2py(self.clustal_omega_consensus_no_conv)


        self.r_msa.print(self.clustal_omega_alignment, show="complete")

        print('------------------------------------------------------------------------------')
        print('Muscle alignment->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->\n')

        self.muscle_alignment = self.r_msa.msa(self.r_sequences, method='Muscle')

        self.muscle_consensus_no_conv =self.r_msa.msaConsensusSequence(self.muscle_alignment,
                                                                    type="upperlower",
                                                                    thresh=r_upperlower_int)


        self.muscle_consensus = pandas2ri.ri2py(self.muscle_consensus_no_conv)

        self.r_msa.print(self.clustal_omega_alignment, show="complete")

        print('------------------------------------------------------------------------------\n')

        print('Alignment with Clustal, ClustalOmega and Muscle Finished\n')

        test = 1


    def pretty_print_basic_all(self):
        '''

        Pretty print all aligmnent into tex files


        msa library takes pape size in inches
        :return:
        '''
        r_vector_further_code = rvectors.StrVector([

            "\\showlogoscale{leftright}",

            "\\showruler{1}{top}"])

        r_vector_further_code.r_repr()

        # self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
        #                           file='testus_all.tex',
        #                showNames="left", showLogo="top",
        #                showNumbering='none',
        #                logoColors="chemical", shadingMode="functional",
        #                shadingModeArg="structure",
        #                showLegend=True, askForOverwrite=False, verbose=False,
        #                           furtherCode=r_vector_test)



        # Paper width
        # paper_width = 11
        # paper_height= 8.5


        paper_width_cm = 21
        paper_height_cm = 29

        to_inches = plot_tools.cm2inch(paper_width_cm, paper_height_cm )


        paper_width_inch = to_inches[0]
        paper_height_inch = to_inches[1]

        margins_def = 0.1

        r_margin_int = rvectors.FloatVector([margins_def, margins_def])

        test = 1


        self.r_msa.msaPrettyPrint(self.clustal_omega_alignment,

                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch ,
                                  margins = r_margin_int,
                                  output="tex",
                                  file='basic_clustalOmega_alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,
                                  alFile = 'basic_clustalOmega_alignment.fasta',

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)
        self.r_tools.texi2tex("basic_clustalOmega_alignment.tex", clean=True)


        test = 1
        self.r_msa.msaPrettyPrint(self.clustal_w_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch ,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='basic_clustalW_alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        self.r_msa.msaPrettyPrint(self.muscle_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch ,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='basic_muscle_alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)


        print('----------------------------------------------------------------------\n')
        print('Basic Pretty print has finished\n')

    def pretty_print_ligand_all(self):
        '''

        Pretty print all aligmnent into tex files
        ligand are drawn as boxes
        helix and beta strands are drawn as well


        all show interacting residues with colored boxes

        msa library takes pape size in inches
        :return:
        '''





        further_code_list = [

            "\\showlogoscale{leftright}",

            "\\bottomspace{2in}",

            "\\showruler{1}{top}"]




        key_data = list(self.analyzed_mmpbsa_data.keys())[0]
        helix_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['helix']
        strands_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['strands']

        test = 1


        for curr_helix_index in helix_data:
            test = 1
            h_res_start =str(helix_data[curr_helix_index][0]+1)
            h_res_end = str(helix_data[curr_helix_index][-1]+1)
            h_index = str(curr_helix_index )
            temp_str = "\\feature{top}{1}{" + h_res_start + ".." + h_res_end + "}{helix}{H" + h_index + "}"
            test =1
            further_code_list.append(temp_str)


        for curr_strand_index in strands_data:
            test = 1
            s_res_start =str(strands_data[curr_strand_index][0]+1)
            s_res_end = str(strands_data[curr_strand_index][-1]+1)
            s_index = str(curr_strand_index)
            temp_str = "\\feature{top}{1}{" + s_res_start + ".." + s_res_end + "}{o->}{S" + s_index + "}"
            test =1
            further_code_list.append(temp_str)


        # Draw boxes of interacting ligands
        key_data = list(self.analyzed_mmpbsa_data.keys())


        #  test working "\\frameblock{1}{19..19}{Black[0.5pt]}"

        self.conserved_residues = {}

        for curr_key in self.analyzed_mmpbsa_data:
            curr_data = self.analyzed_mmpbsa_data[curr_key]

            curr_most_contrib = curr_data['mostAllContrib']

            residue_nums = curr_most_contrib['ResidueNum']
            residues = curr_most_contrib['Residue']

            test = 1

            self.conserved_residues.update({curr_key:[]})

            for resid_index, residue in zip(residue_nums,residues):

                # convert aminoacid 3 letter code to 1 and check if it is in the consensus sequence
                curr_residue = residue[0:3]
                one_letter_code = bio_prot_tools.three_to_one(curr_residue)


                muscle_letter = str.upper(self.muscle_consensus[0][resid_index-1])
                omega_letter = str.upper(self.clustal_omega_consensus[0][resid_index-1])
                w_letter = str.upper(self.clustal_w_consensus[0][resid_index-1])
                if muscle_letter  == one_letter_code and omega_letter == one_letter_code and w_letter == one_letter_code:
                    # str.upper(self.muscle_consensus[0][17])
                    test = 1

                    self.conserved_residues[curr_key].append(resid_index)

                    temp_str = "\\frameblock{1}{" + str(resid_index) + ".." + str(resid_index) + "}{" + \
                               self.colors_texshade[curr_key] +"[0.75pt]}"


                    # arrow_str = "\\feature{bbottom}{2}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{" + \
                    #            self.colors_texshade[curr_key] +"[1pt]}"

                    arrow_str = "\\feature{bbbottom}{1}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{}"

                    further_code_list.append(temp_str)

                    further_code_list.append(arrow_str)



            test = 1


        r_vector_further_code = rvectors.StrVector(further_code_list)

        r_vector_further_code.r_repr()





        # Pretty print part

        # Paper width
        # paper_width = 11
        # paper_height= 8.5


        paper_width_cm = 21
        paper_height_cm = 29

        to_inches = plot_tools.cm2inch(paper_width_cm, paper_height_cm)

        paper_width_inch = to_inches[0]
        paper_height_inch = to_inches[1]

        margins_def = 0.5

        r_margin_int = rvectors.FloatVector([margins_def, margins_def])

        test = 1

        self.r_msa.msaPrettyPrint(self.clustal_omega_alignment,

                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,
                                  output="tex",
                                  file='ligand-all-clustalOmega-alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  alFile = 'ligand-all-clustalOmega-alignment.fasta',

                                  showLegend=True,
                                  askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        #self.r_tools.texi2pdf("ligand-all-clustalOmega-alignment.tex", clean=True)


        self.r_msa.msaPrettyPrint(self.clustal_w_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='ligand-all-clustalW-alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,
                                  alFile = 'ligand-all-clustalW-alignment.fasta',

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        self.r_msa.msaPrettyPrint(self.muscle_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='ligand-all-muscle-alignment.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,
                                  alFile = 'ligand-all-muscle-alignment.fasta',

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        print('----------------------------------------------------------------------\n')
        print('Total ligand aa  Pretty print has finished\n')



    def pretty_print_ligand_all_for_all_residues(self):
        '''

        Pretty print all aligmnent into tex files
        ligand are drawn as boxes
        helix and beta strands are drawn as well


        all show interacting residues with colored boxes

        msa library takes pape size in inches
        :return:
        '''


        print('Start for pretty_print_ligand_all_for_all_residues')


        further_code_list = [

            "\\showlogoscale{leftright}",

            "\\bottomspace{2in}",

            "\\showruler{1}{top}"]




        key_data = list(self.analyzed_mmpbsa_data.keys())[0]
        helix_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['helix']
        strands_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['strands']

        test = 1


        for curr_helix_index in helix_data:
            test = 1
            h_res_start =str(helix_data[curr_helix_index][0]+1)
            h_res_end = str(helix_data[curr_helix_index][-1]+1)
            h_index = str(curr_helix_index )
            temp_str = "\\feature{top}{1}{" + h_res_start + ".." + h_res_end + "}{helix}{H" + h_index + "}"
            test =1
            further_code_list.append(temp_str)


        for curr_strand_index in strands_data:
            test = 1
            s_res_start =str(strands_data[curr_strand_index][0]+1)
            s_res_end = str(strands_data[curr_strand_index][-1]+1)
            s_index = str(curr_strand_index)
            temp_str = "\\feature{top}{1}{" + s_res_start + ".." + s_res_end + "}{o->}{S" + s_index + "}"
            test =1
            further_code_list.append(temp_str)


        # Draw boxes of interacting ligands
        key_data = list(self.analyzed_mmpbsa_data.keys())


        #  test working "\\frameblock{1}{19..19}{Black[0.5pt]}"

        self.conserved_residues = {}

        for curr_key in self.analyzed_mmpbsa_data:
            curr_data = self.analyzed_mmpbsa_data[curr_key]

            curr_most_contrib = curr_data['mostAllContrib']

            residue_nums = curr_most_contrib['ResidueNum']
            residues = curr_most_contrib['Residue']

            test = 1

            self.conserved_residues.update({curr_key:[]})

            for resid_index, residue in zip(residue_nums,residues):

                # convert aminoacid 3 letter code to 1 and check if it is in the consensus sequence
                curr_residue = residue[0:3]
                one_letter_code = bio_prot_tools.three_to_one(curr_residue)


                muscle_letter = str.upper(self.muscle_consensus[0][resid_index-1])
                omega_letter = str.upper(self.clustal_omega_consensus[0][resid_index-1])
                w_letter = str.upper(self.clustal_w_consensus[0][resid_index-1])


                # str.upper(self.muscle_consensus[0][17])
                test = 1

                self.conserved_residues[curr_key].append(resid_index)

                temp_str = "\\frameblock{1}{" + str(resid_index) + ".." + str(resid_index) + "}{" + \
                           self.colors_texshade[curr_key] +"[0.75pt]}"


                # arrow_str = "\\feature{bbottom}{2}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{" + \
                #            self.colors_texshade[curr_key] +"[1pt]}"

                arrow_str = "\\feature{bbbottom}{1}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{}"

                further_code_list.append(temp_str)

                further_code_list.append(arrow_str)



            test = 1


        r_vector_further_code = rvectors.StrVector(further_code_list)

        r_vector_further_code.r_repr()





        # Pretty print part

        # Paper width
        # paper_width = 11
        # paper_height= 8.5


        paper_width_cm = 21
        paper_height_cm = 29

        to_inches = plot_tools.cm2inch(paper_width_cm, paper_height_cm)

        paper_width_inch = to_inches[0]
        paper_height_inch = to_inches[1]

        margins_def = 0.5

        r_margin_int = rvectors.FloatVector([margins_def, margins_def])

        test = 1

        self.r_msa.msaPrettyPrint(self.clustal_omega_alignment,

                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,
                                  output="tex",
                                  file='ligand_all_clustalOmega_alignment_all_residues.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  showLegend=True,
                                  askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        self.r_msa.msaPrettyPrint(self.clustal_w_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='ligand_all_clustalW_alignment_all_residues.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        self.r_msa.msaPrettyPrint(self.muscle_alignment,
                                  paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                  margins=r_margin_int,

                                  output="tex",
                                  file='ligand_all_muscle_alignment_all_residues.tex',
                                  showNames="left", showLogo="top",
                                  showNumbering='none',
                                  logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,

                                  showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_further_code)

        print('----------------------------------------------------------------------\n')
        print('Total ligand aa  Pretty print has finished\n')






    def pretty_print_ligand_specific(self):
        '''

        Pretty print all aligmnent into tex files
        ligand are drawn as boxes
        helix and beta strands are drawn as well


        all show interacting residues with colored boxes

        msa library takes pape size in inches
        :return:
        '''

        further_code_list = [

            "\\showlogoscale{leftright}",

            "\\bottomspace{2in}",

            "\\rulersteps{1}",

            "\\featuressmall",
            "\\domaingapcolors{Yellow}{White}",
            "\\domaingaprule{1.5pt}",

            "\\showruler{bottom}{1}"]

        key_data = list(self.analyzed_mmpbsa_data.keys())[0]
        helix_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['helix']
        strands_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['strands']

        test = 1



        # Draw boxes of interacting ligands
        key_data = list(self.analyzed_mmpbsa_data.keys())

        #  test working "\\frameblock{1}{19..19}{Black[0.5pt]}"

        self.conserved_residues = {}

        for curr_key in self.analyzed_mmpbsa_data:

            further_code_list_curr = further_code_list[:]


            curr_data = self.analyzed_mmpbsa_data[curr_key]

            curr_most_contrib = curr_data['mostAllContrib']

            residue_nums = curr_most_contrib['ResidueNum']
            residues = curr_most_contrib['Residue']

            test = 1

            self.conserved_residues.update({curr_key: []})

            for resid_index, residue in zip(residue_nums, residues):

                # convert aminoacid 3 letter code to 1 and check if it is in the consensus sequence
                curr_residue = residue[0:3]
                one_letter_code = bio_prot_tools.three_to_one(curr_residue)

                muscle_letter = str.upper(self.muscle_consensus[0][resid_index - 1])
                omega_letter = str.upper(self.clustal_omega_consensus[0][resid_index - 1])
                w_letter = str.upper(self.clustal_w_consensus[0][resid_index - 1])
                if muscle_letter == one_letter_code and omega_letter == one_letter_code and w_letter == one_letter_code:
                    # str.upper(self.muscle_consensus[0][17])
                    test = 1

                    self.conserved_residues[curr_key].append(resid_index)

                    temp_str = "\\frameblock{1}{" + str(resid_index) + ".." + str(resid_index) + "}{" + \
                               self.colors_texshade[curr_key] + "[0.75pt]}"

                    # arrow_str = "\\feature{bbottom}{2}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{" + \
                    #            self.colors_texshade[curr_key] +"[1pt]}"

                    arrow_str = "\\feature{bbbottom}{1}{" + str(resid_index) + ".." + str(
                        resid_index) + "}{fill:$\\uparrow$}{}"

                    further_code_list_curr.append(temp_str)

                    further_code_list_curr.append(arrow_str)

            # sorted(self.conserved_residues[2])

            test = 1

            set_domains_list = []

            sorted_conserved_residues = sorted(self.conserved_residues[curr_key])

            try:
                temp = []
                for curr_index in range(len(sorted_conserved_residues)):
                    if sorted_conserved_residues[curr_index] not in temp:
                        temp.append(sorted_conserved_residues[curr_index])

                    if sorted_conserved_residues[curr_index+1] - sorted_conserved_residues[curr_index] <4:
                        temp.append(sorted_conserved_residues[curr_index+1])
                    else:
                        set_domains_list.append(temp)
                        temp = []

            except:
                pass

            set_domains_list.append(temp)

            #"\\setdomain{1}{15..20,169..240}"

            set_domains_further_code = []

            for domain in set_domains_list:
                domain_str = ''
                if len(domain) == 1:
                    domain_str += str(domain[0]-1) + '..' + str(domain[0]+1)
                else:
                    domain_str += str(domain[0] - 1) + '..' + str(domain[-1] + 1)
                set_domains_further_code.append(domain_str)


            main_domain_str = "\\setdomain{1}{"
            for str_domain_index in range(len(set_domains_further_code)):
                if str_domain_index == len(set_domains_further_code)-1:
                    main_domain_str += set_domains_further_code[str_domain_index] + '}'
                else:
                    main_domain_str += set_domains_further_code[str_domain_index] + ','




            further_code_list_curr.append(main_domain_str)

            test = 1

            # for curr_helix_index in helix_data:
            #     test = 1
            #     h_res_start = str(helix_data[curr_helix_index][0] + 1)
            #     h_res_end = str(helix_data[curr_helix_index][-1] + 1)
            #     h_index = str(curr_helix_index)
            #     temp_str = "\\feature{top}{1}{" + h_res_start + ".." + h_res_end + "}{helix}{H" + h_index + "}"
            #     test = 1
            #     further_code_list.append(temp_str)
            #
            # for curr_strand_index in strands_data:
            #     test = 1
            #     s_res_start = str(strands_data[curr_strand_index][0] + 1)
            #     s_res_end = str(strands_data[curr_strand_index][-1] + 1)
            #     s_index = str(curr_strand_index)
            #     temp_str = "\\feature{top}{1}{" + s_res_start + ".." + s_res_end + "}{o->}{S" + s_index + "}"
            #     test = 1
            #     further_code_list.append(temp_str)


            r_vector_further_code = rvectors.StrVector(further_code_list_curr)

            r_vector_further_code.r_repr()

            # Pretty print part

            # Paper width
            # paper_width = 11
            # paper_height= 8.5


            paper_width_cm = 22
            paper_height_cm = 22

            to_inches = plot_tools.cm2inch(paper_width_cm, paper_height_cm)

            paper_width_inch = to_inches[0]
            paper_height_inch = to_inches[1]

            margins_def = 0.1

            r_margin_int = rvectors.FloatVector([margins_def, margins_def])

            test = 1

            self.r_msa.msaPrettyPrint(self.clustal_omega_alignment,

                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,
                                      output="tex",
                                      file='ligand-all-clustalOmega-alignment_num:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="identical",
                                      shadingModeArg=100,
                                      shadingColors='greens',
                                      showConsensus='none',
                                      consensusThreshold=75,
                                      alFile='ligand-all-clustalOmega-alignment_num:{0}.fasta'.format(curr_key),

                                      showLegend=True,
                                      askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

            self.r_msa.msaPrettyPrint(self.clustal_w_alignment,
                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,

                                      output="tex",
                                      file='ligand-all-clustalW-alignment_num:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="identical",
                                      shadingModeArg=100,
                                      shadingColors='greens',
                                      showConsensus='none',
                                      consensusThreshold=75,
                                      alFile='ligand-all-clustalW-alignment_num:{0}.fasta'.format(curr_key),

                                      showLegend=True, askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

            self.r_msa.msaPrettyPrint(self.muscle_alignment,
                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,

                                      output="tex",
                                      file='ligand-all-muscle-alignment_num:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="identical",
                                      shadingModeArg=100,
                                      shadingColors='greens',
                                      showConsensus='none',
                                      consensusThreshold=75,
                                      alFile='ligand-all-muscle-alignment_num:{0}.fasta'.format(curr_key),

                                      showLegend=True, askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

        print('----------------------------------------------------------------------\n')
        print('Total ligand aa  Pretty print specific has finished\n')




    def pretty_print_ligand_specific_all_residues(self,paper_width=22, paper_height=22):
        '''

        Pretty print all aligmnent into tex files
        ligand are drawn as boxes
        helix and beta strands are drawn as well


        all show interacting residues with colored boxes

        msa library takes pape size in inches
        :return:
        '''

        further_code_list = [

            "\\showlogoscale{leftright}",

            "\\bottomspace{2in}",

            "\\rulersteps{1}",

            "\\featuressmall",
            "\\domaingapcolors{Yellow}{White}",
            "\\domaingaprule{1.5pt}",

            "\\showruler{bottom}{1}"]

        key_data = list(self.analyzed_mmpbsa_data.keys())[0]
        helix_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['helix']
        strands_data = self.analyzed_mmpbsa_data[key_data]['dsspStructures']['strands']

        test = 1



        # Draw boxes of interacting ligands
        key_data = list(self.analyzed_mmpbsa_data.keys())

        #  test working "\\frameblock{1}{19..19}{Black[0.5pt]}"

        self.conserved_residues = {}

        for curr_key in self.analyzed_mmpbsa_data:

            further_code_list_curr = further_code_list[:]


            curr_data = self.analyzed_mmpbsa_data[curr_key]

            curr_most_contrib = curr_data['mostAllContrib']

            residue_nums = curr_most_contrib['ResidueNum']
            residues = curr_most_contrib['Residue']

            test = 1

            self.conserved_residues.update({curr_key: []})

            for resid_index, residue in zip(residue_nums, residues):

                # convert aminoacid 3 letter code to 1 and check if it is in the consensus sequence
                curr_residue = residue[0:3]
                one_letter_code = bio_prot_tools.three_to_one(curr_residue)

                muscle_letter = str.upper(self.muscle_consensus[0][resid_index - 1])
                omega_letter = str.upper(self.clustal_omega_consensus[0][resid_index - 1])
                w_letter = str.upper(self.clustal_w_consensus[0][resid_index - 1])


                self.conserved_residues[curr_key].append(resid_index)

                temp_str = "\\frameblock{1}{" + str(resid_index) + ".." + str(resid_index) + "}{" + \
                           self.colors_texshade[curr_key] + "[0.75pt]}"

                # arrow_str = "\\feature{bbottom}{2}{" + str(resid_index) + ".." + str(resid_index) + "}{fill:$\\uparrow$}{" + \
                #            self.colors_texshade[curr_key] +"[1pt]}"

                arrow_str = "\\feature{bbbottom}{1}{" + str(resid_index) + ".." + str(
                    resid_index) + "}{fill:$\\uparrow$}{}"

                further_code_list_curr.append(temp_str)

                further_code_list_curr.append(arrow_str)

            # sorted(self.conserved_residues[2])

            test = 1

            set_domains_list = []

            sorted_conserved_residues = sorted(self.conserved_residues[curr_key])

            try:
                temp = []
                for curr_index in range(len(sorted_conserved_residues)):
                    if sorted_conserved_residues[curr_index] not in temp:
                        temp.append(sorted_conserved_residues[curr_index])

                    if sorted_conserved_residues[curr_index+1] - sorted_conserved_residues[curr_index] <4:
                        temp.append(sorted_conserved_residues[curr_index+1])
                    else:
                        set_domains_list.append(temp)
                        temp = []

            except:
                pass

            set_domains_list.append(temp)

            #"\\setdomain{1}{15..20,169..240}"

            set_domains_further_code = []

            for domain in set_domains_list:
                domain_str = ''
                if len(domain) == 1:
                    domain_str += str(domain[0]-1) + '..' + str(domain[0]+1)
                else:
                    domain_str += str(domain[0] - 1) + '..' + str(domain[-1] + 1)
                set_domains_further_code.append(domain_str)


            main_domain_str = "\\setdomain{1}{"
            for str_domain_index in range(len(set_domains_further_code)):
                if str_domain_index == len(set_domains_further_code)-1:
                    main_domain_str += set_domains_further_code[str_domain_index] + '}'
                else:
                    main_domain_str += set_domains_further_code[str_domain_index] + ','




            further_code_list_curr.append(main_domain_str)

            test = 1



            r_vector_further_code = rvectors.StrVector(further_code_list_curr)

            r_vector_further_code.r_repr()

            # Pretty print part

            # Paper width
            # paper_width = 11
            # paper_height= 8.5


            paper_width_cm = paper_width
            paper_height_cm = paper_height

            to_inches = plot_tools.cm2inch(paper_width_cm, paper_height_cm)

            paper_width_inch = to_inches[0]
            paper_height_inch = to_inches[1]

            margins_def = 0.1

            r_margin_int = rvectors.FloatVector([margins_def, margins_def])

            test = 1

            self.r_msa.msaPrettyPrint(self.clustal_omega_alignment,

                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,
                                      output="tex",
                                      file='ligand_all_clustalOmega_alignment_num_all_resid:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="similar",
                                      shadingColors='blues',
                                      showConsensus='none',
                                      consensusThreshold=50,

                                      showLegend=True,
                                      askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

            self.r_msa.msaPrettyPrint(self.clustal_w_alignment,
                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,

                                      output="tex",
                                      file='ligand_all_clustalW_alignment_num_all_resid:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="similar",
                                      shadingColors='blues',
                                      showConsensus='none',
                                      consensusThreshold=50,

                                      showLegend=True, askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

            self.r_msa.msaPrettyPrint(self.muscle_alignment,
                                      paperWidth=paper_width_inch, paperHeight=paper_height_inch,
                                      margins=r_margin_int,

                                      output="tex",
                                      file='ligand_all_muscle_alignment_num_all_resid:{0}.tex'.format(curr_key),
                                      showNames="left", showLogo="top",
                                      showNumbering='none',
                                      logoColors="chemical", shadingMode="similar",
                                      shadingColors='blues',
                                      showConsensus='none',
                                      consensusThreshold=50,

                                      showLegend=True, askForOverwrite=False, verbose=False,
                                      furtherCode=r_vector_further_code)

        print('----------------------------------------------------------------------\n')
        print('Total ligand aa  Pretty print specific all residues not just conservative has finished\n')



    def working_example(self):

        # TODO original
        r_vector_test =rvectors.StrVector(["\\defconsensus{.}{lower}{upper}",
                                                "\\showruler{1}{top}"])

        # # TODO this works
        # r_vector_test =rvectors.StrVector(["\\defconsensus{.}{lower}{upper}",
        #                                    "\\setdomain{1}{15..20,169..240}",
        #                                    "\\feature{top}{1}{177..181}{brace}{tinted}",
        #
        #                                    "\\feature{top}{1}{19..19}{fill:$\downarrow$}{19 SLR binding mode}",
        #                                    "\\feature{top}{1}{171..175}{box[Blue,Red][0.5pt]:H1[Yellow]}{transmembrane domain 4}",
        #                                    "\\frameblock{1}{19..19}{Black[1pt]}"
        #
        #                                         "\\showruler{1}{top}"])


        # "\\defconsensus{.}{lower}{upper}",

        # TODO this works
        r_vector_test = rvectors.StrVector([

                                            "\\showlogoscale{leftright}",

                                            "\\setdomain{1}{15..20,169..240}",
                                            "\\feature{bottom}{1}{177..181}{brace}{tinted}",

                                            "\\feature{bottom}{1}{19..19}{fill:$\downarrow$}{19 SLR binding mode}",

                                            "\\feature{top}{1}{169..186}{helix}{H1}",
                                            "\\feature{top}{1}{190..238}{helix}{H2}",

                                            "\\frameblock{1}{19..19}{Black[0.5pt]}"

                                            "\\showruler{1}{top}"])


        r_vector_test.r_repr()

        # self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
        #                           file='testus_all.tex',
        #                showNames="left", showLogo="top",
        #                showNumbering='none',
        #                logoColors="chemical", shadingMode="functional",
        #                shadingModeArg="structure",
        #                showLegend=True, askForOverwrite=False, verbose=False,
        #                           furtherCode=r_vector_test)

        self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
                                  file='testus_all.tex',
                       showNames="left", showLogo="top",
                       showNumbering='none',
                       logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='none',
                                  consensusThreshold=75,


                       showLegend=True, askForOverwrite=False, verbose=False,
                                  furtherCode=r_vector_test)

        # showConsensus = 'bottom',
        # consensusColors = 'RedGreen',
        # consensusThreshold = 75,


        my_first_alignment = self.r_msa.msa(curr_sequence, method='Muscle')

        self.r_msa.print(my_first_alignment, show="complete")

        r_vector_test = rvectors.StrVector(["\\defconsensus{.}{lower}{upper}",
                                            "\\showruler{1}{top}"])
        r_vector_test.r_repr()

        self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
                                  file='testus_all_muscle.tex',
                                  showNames="none", showLogo="top",
                                  showNumbering='left',
                                  logoColors="chemical", shadingMode="functional",
                                  shadingModeArg="structure",
                                  showLegend=True, askForOverwrite=False, verbose=True,
                                  furtherCode=r_vector_test)


        r_vector_test_int=rvectors.IntVector([169,239])
        r_vector_test_int.r_repr()

        # self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
        #                           file='testus_region.tex',
        #                           y=r_vector_test_int,
        #                showNames="none", showLogo="top",
        #                showNumbering='left',
        #                logoColors="chemical", shadingMode="functional",
        #                consensusColor="ColdHot",
        #                shadingModeArg="structure",
        #                showLegend=True, askForOverwrite=False,
        #                           furtherCode=r_vector_test)


        self.r_msa.msaPrettyPrint(my_first_alignment, output="tex",
                                  file='testus_region.tex',
                                  y=r_vector_test_int,
                       showNames="none", showLogo="top",
                       showNumbering='left',
                       logoColors="chemical", shadingMode="identical",
                                  shadingModeArg=100,
                                  shadingColors='greens',
                                  showConsensus='bottom',
                                  consensusColors='RedGreen',
                                  consensusThreshold=100,
        showLegend=True, askForOverwrite=False,furtherCode=r_vector_test)

        test = 1


    ####################################################################################################################


