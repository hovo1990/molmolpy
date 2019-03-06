#!/usr/bin/env python
#
# Inspired by g_mmpbsa code.
# #
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

from __future__ import absolute_import, division, print_function
from builtins import range
from builtins import object

import re
import numpy as np
import argparse
import sys
import os
import math
import time

from copy import deepcopy

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from matplotlib.colors import ListedColormap

import mdtraj as md

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import converters
from molmolpy.utils import plot_tools
from molmolpy.utils import pdb_tools
from molmolpy.utils import folder_utils
from molmolpy.utils import extra_tools
from molmolpy.utils import pymol_tools
from molmolpy.utils import protein_analysis


class EnergyAnalysisObject(object):
    """

    Usage Example

        >>> from molmolpy.moldyn import md_analysis
        >>> from molmolpy.g_mmpbsa import mmpbsa_analyzer
        >>>
        >>> import os
        >>>
        >>> # In[3]:
        >>>
        >>> folder_to_sim = '/media/Work/SimData/g_mmpbsa/HSL/HSL_1_backbone/Cluster1/'
        >>>
        >>> molmech = folder_to_sim + 'contrib_MM.dat'
        >>> polar = folder_to_sim + 'contrib_pol.dat'
        >>> apolar = folder_to_sim + 'contrib_apol.dat'
        >>>
        >>> LasR_energy_object = mmpbsa_analyzer.EnergyAnalysisObject(molmech, polar, apolar,
        >>>                                                           sim_num=3)
        >>>
        >>> LasR_energy_object.plot_bar_energy_residues()
        >>> LasR_energy_object.plot_most_contributions()
        >>> LasR_energy_object.plot_sorted_contributions()
        >>>
        >>>
        >>> centroid_file = '/media/Work/MEGA/Programming/docking_LasR/HSL_1_v8/centroid.pdb'
        >>>
        >>>
        >>> LasR_energy_object.add_centroid_pdb_file(centroid_file)
        >>> LasR_energy_object.save_mmpbsa_analysis_pickle('HSL_simulation_cluster3.pickle')
        >>> #LasR_energy_object.visualize_interactions_pymol()
        >>>
        >>>
        >>> test = 1
        >>> # simulation_name = 'LasR_Ligand_simulation'
        >>> #



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
    def __init__(self,
                 energymm_xvg,
                 polar_xvg,
                 apolar_xvg,
                 molmech,
                 polar,
                 apolar,
                 bootstrap=True,
                 bootstrap_steps=5000,
                 sim_num=1,
                 receptor_name='LasR',
                 molecule_name='HSL',
                 meta_file=None
                 ):
        self.receptor_name = receptor_name
        self.molecule_name = molecule_name
        self.sim_num = sim_num

        self.simulation_name = self.receptor_name + '_' + self.molecule_name + '_num:' + str(self.sim_num)

        self.meta_file = meta_file

        # molmech = folder_to_sim + 'contrib_MM.dat'
        # polar = folder_to_sim + 'contrib_pol.dat'
        # apolar = folder_to_sim + 'contrib_apol.dat'







        # Complex Energy
        c = []
        if meta_file is not None:
            MmFile, PolFile, APolFile = ReadMetafile(meta_file)
            for i in range(len(MmFile)):
                cTmp = Complex(MmFile[i], PolFile[i], APolFile[i], K[i])
                cTmp.CalcEnergy(args, frame_wise, i)
                c.append(cTmp)
        else:
            cTmp = Complex(energymm_xvg, polar_xvg, apolar_xvg)
            self.cTmp = cTmp

            self.full_copy_original = deepcopy(cTmp)
            self.full_copy_bootstrap = deepcopy(cTmp)


            # cTmp.CalcEnergy(frame_wise, 0, bootstrap=bootstrap, bootstrap_steps=bootstrap_steps)
            # c.append(cTmp)
            # Summary in output files => "--outsum" and "--outmeta" file options


        # TODO adapt to make able to use bootstrap as well, multiple analysis modes?



        self.c = c



        # summary_output_filename = self.simulation_name + '_binding_summary.log'
        # Summary_Output_File(c, summary_output_filename, meta_file)
        #
        # corr_outname =  self.simulation_name + '_correllation_distance.log'
        # corr_plot = self.simulation_name + '_correllation_plot.png'




        test = 1
        # This won't work it needs K, read paper again
        #FitCoef_all = PlotCorr(c, corr_outname, corr_plot, bootstrap_steps)
        #PlotEnrgy(c, FitCoef_all, args, args.enplot)

        # RESIDUE analysis part
        self.MMEnData, self.resnameA = ReadData_Residue_Parse(molmech)
        self.polEnData, self.resnameB = ReadData_Residue_Parse(polar)
        self.apolEnData, self.resnameC = ReadData_Residue_Parse(apolar)

        self.resname = CheckResname(self.resnameA, self.resnameB, self.resnameC)
        self.sim_num = sim_num
        Residues = []

        data = []

        columns_residue_energy = ['index', 'ResidueNum', 'Residue', 'TotalEnergy', 'TotalEnergySD']

        for i in range(len(self.resname)):
            CheckEnData_residue(self.MMEnData[i], self.polEnData[i], self.apolEnData[i])
            r = Residue()
            r.CalcEnergy(self.MMEnData[i], self.polEnData[i], self.apolEnData[i], bootstrap, bootstrap_steps)
            Residues.append(r)
            # print(' %8s %8.4f %8.4f' % (self.resname[i], r.TotalEn[0], r.TotalEn[1]))
            data.append([i, i + 1, self.resname[i], r.TotalEn[0], r.TotalEn[1]])

        self.pandas_residue_energy_data = pd.DataFrame(data)
        self.pandas_residue_energy_data.columns = columns_residue_energy
        test = 1

        self.most_contributions = self.pandas_residue_energy_data[:-1]

        self.most_contributions = self.most_contributions.sort_values(['TotalEnergy'])

        test = 1



    def calculate_binding_energy_full(self, idx=0,jump_data=1, bootstrap=False, bootstrap_steps=5000):
        '''
        Calculate full binding energy then analyze autocorrelation and partial correlation

        :param idx: from frame number
        :param bootstrap: for this one dont calculate bootstrap
        :param bootstrap_steps:
        :return:
        '''

        # TODO CALCULATION OF BINDING ENERGY
        outfr = self.simulation_name + '_full.log'
        try:
            frame_wise = open(outfr, 'w')
        except:
            raise IOError('Could not open file {0} for writing. \n'.format(outfr))

        frame_wise.write(
            '#Time E_VdW_mm(Protein)\tE_Elec_mm(Protein)\tE_Pol(Protein)\tE_Apol(Protein)\tE_VdW_mm(Ligand)\tE_Elec_mm(Ligand)\tE_Pol(Ligand)\tE_Apol(Ligand)\tE_VdW_mm(Complex)\tE_Elec_mm(Complex)\tE_Pol(Complex)\tE_Apol(Complex)\tDelta_E_mm\tDelta_E_Pol\tDelta_E_Apol\tDelta_E_binding\n')

        self.frame_wise_full = frame_wise

        self.c_full = []
        self.full_copy_original.CalcEnergy(self.frame_wise_full, idx, jump_data=jump_data, bootstrap=bootstrap, bootstrap_steps=bootstrap_steps)
        self.c_full.append(self.full_copy_original)

        summary_output_filename = self.simulation_name + '_binding_summary_full.log'
        Summary_Output_File(self.c_full, summary_output_filename, self.meta_file)
        self.autocorr_analysis(self.c_full, 'full')


    def calculate_binding_energy_bootstrap(self, idx=0, bootstrap=True, bootstrap_steps=5000, bootstrap_jump=4):
        '''
        Calculate bootstrap binding energy then analyze autocorrelation and partial correlation

        :param idx: from frame number
        :param bootstrap: for this one dont calculate bootstrap
        :param bootstrap_steps:
        :return:
        '''

        # TODO CALCULATION OF BINDING ENERGY
        outfr = self.simulation_name + '_bootstrap.log'
        try:
            frame_wise = open(outfr, 'w')
        except:
            raise IOError('Could not open file {0} for writing. \n'.format(outfr))

        frame_wise.write(
            '#Time E_VdW_mm(Protein)\tE_Elec_mm(Protein)\tE_Pol(Protein)\tE_Apol(Protein)\tE_VdW_mm(Ligand)\tE_Elec_mm(Ligand)\tE_Pol(Ligand)\tE_Apol(Ligand)\tE_VdW_mm(Complex)\tE_Elec_mm(Complex)\tE_Pol(Complex)\tE_Apol(Complex)\tDelta_E_mm\tDelta_E_Pol\tDelta_E_Apol\tDelta_E_binding\n')

        self.frame_wise_bootstrap = frame_wise

        self.c_bootstrap = []
        self.full_copy_bootstrap.CalcEnergy(self.frame_wise_bootstrap, idx,
                                            bootstrap=bootstrap,
                                            bootstrap_steps=bootstrap_steps,
                                            bootstrap_jump=bootstrap_jump)

        self.c_bootstrap.append(self.full_copy_bootstrap)

        summary_output_filename = self.simulation_name + '_binding_summary_bootstrap.log'
        Summary_Output_File(self.c_bootstrap, summary_output_filename, self.meta_file)
        self.autocorr_analysis(self.c_bootstrap, 'bootstrap')






    def autocorr_analysis(self, energy_val, naming='full'):




        if naming =='full':
            total_en = energy_val[0].TotalEn
            time = energy_val[0].time
        else:
            total_en = energy_val[0].TotalEn_bootstrap
            time = energy_val[0].time_bootstrap


        # Old version :)
        # print('Mean autocorrelation ', np.mean(autocorr(total_en)))
        # plt.semilogx(time, autocorr(total_en))
        # plt.xlabel('Time [ps]', size=16)
        # plt.ylabel('Binding Energy autocorrelation', size=16)
        # plt.show()

        from pandas import Series
        from matplotlib import pyplot
        from statsmodels.graphics.tsaplots import plot_acf
        series = Series.from_array(total_en, index=time)

        # https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
        plot_acf(series, alpha=0.05)
        # pyplot.show()
        plt.savefig(self.simulation_name + '_autocorrelation_bindingEnergy_{0}.png'.format(naming), dpi=600)

        from statsmodels.graphics.tsaplots import plot_pacf

        # plot_pacf(series, lags=50)
        plot_pacf(series)
        plt.savefig(self.simulation_name +'_partial_autocorrelation_bindingEnergy_{0}.png'.format(naming), dpi=600)
        #pyplot.show()
        test = 1



    def plot_binding_energy_full(self):
        bind_energy = self.full_copy_original.TotalEn
        time = self.full_copy_original.time

        dataframe = converters.convert_data_to_pandas(time, bind_energy,
                                                      x_axis_name='time',
                                                      y_axis_name='binding')

        import seaborn as sns
        sns.set(style="ticks")

        plt.clf()


        plt.plot(time, bind_energy)
        plt.savefig('test.png', dpi=600)
        # sns.lmplot(x="time", y="binding",data=dataframe,
        #         ci=None, palette="muted", size=4,
        #            scatter_kws={"s": 50, "alpha": 1})

        # sns.tsplot(data=dataframe)



        test = 1





    def add_centroid_pdb_file(self, filename, simplified_state=True):
        self.centroid_pdb_file = filename

        self.dssp_traj = md.load(self.centroid_pdb_file)

        self.dssp_data = md.compute_dssp(self.dssp_traj, simplified=simplified_state)

        # self.helixes = protein_analysis.find_helixes(self.dssp_data)


        self.helixes = protein_analysis.find_dssp_domain(self.dssp_data, type='H')
        self.strands = protein_analysis.find_dssp_domain(self.dssp_data, type='E')

        self.data_to_save = {self.sim_num: {'residueEnergyData': self.pandas_residue_energy_data[:-1],
                                            'mostResidueContrib': self.most_contributions_plot,
                                            'mostAllContrib': self.most_contributions_plot_all,
                                            'centroidFile': self.centroid_pdb_file,
                                            'dsspObject': self.dssp_data,
                                            'dsspData': self.dssp_data,
                                            'dsspStructures': {'helix': self.helixes,
                                                               'strands': self.strands}}
                             }

        test = 1


    def save_mmpbsa_analysis_pickle(self, filename):
        import pickle
        if filename is None:
            filename = self.simulation_name + '_pickleFile.pickle'
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        pickle.dump(self.data_to_save, open(filename, "wb"))


    def plot_bar_energy_residues(self,
                                 custom_dpi=600,
                                 trasparent_alpha=False):
        # sns.set(style="white", context="talk")
        sns.set(style="ticks", context="paper")

        # Set up the matplotlib figure
        f, ax1 = plt.subplots(1, 1, figsize=(plot_tools.cm2inch(17, 10)), sharex=True)

        # Generate some sequential data

        to_plot_data = self.pandas_residue_energy_data[:-1]

        sns.barplot(to_plot_data['ResidueNum'], to_plot_data['TotalEnergy'],
                    palette="BuGn_d", ax=ax1)
        ax1.set_ylabel("Contribution Energy (kJ/mol)")
        ax1.set_xlabel("Residue Number")

        last_index = to_plot_data['ResidueNum'].iloc[-1]

        # this is buggy
        x_label_key = []

        # ax1.set_xticklabels(to_plot_data['ResidueNum'])  # set new labels
        # # ax1.set_x
        #
        # for ind, label in enumerate(ax1.get_xticklabels()):
        #     if ind+1 == last_index:
        #         label.set_visible(True)
        #     elif (ind+1) % 100 == 0:  # every 100th label is kept
        #         label.set_visible(True)
        #         # label =  round(sim_time[ind])
        #         # x_label_key.append(ind)
        #     else:
        #         label.set_visible(False)
        #     x_label_key.append(ind)

        ax1.set_xlim(1, last_index)

        ax1.xaxis.set_major_locator(ticker.LinearLocator(3))
        ax1.xaxis.set_minor_locator(ticker.LinearLocator(31))

        labels = [item.get_text() for item in ax1.get_xticklabels()]
        test = 1
        labels[0] = '1'
        labels[1] = str(last_index // 2)
        labels[2] = str(last_index)
        ax1.set_xticklabels(labels)

        # ax1.text(0.0, 0.1, "LinearLocator(numticks=3)",
        #         fontsize=14, transform=ax1.transAxes)


        tick_labels = []

        # for ind, tick in enumerate(ax1.get_xticklines()):
        #     # tick part doesn't work
        #     test = ind
        #     # if ind+1 == last_index:
        #     #     tick.set_visible(True)
        #     if (ind+1) % 10 == 0:  # every 100th label is kept
        #         tick.set_visible(True)
        #     else:
        #         tick.set_visible(False)
        #     tick_labels.append(tick)
        #
        # ax1.set_xticklabels

        # for ind, label in enumerate(ax.get_yticklabels()):
        #     if ind % 50 == 0:  # every 100th label is kept
        #         label.set_visible(True)
        #     else:
        #         label.set_visible(False)
        #
        # for ind, tick in enumerate(ax.get_yticklines()):
        #     if ind % 50 == 0:  # every 100th label is kept
        #         tick.set_visible(True)
        #     else:
        #         tick.set_visible(False)

        # Finalize the plot
        sns.despine()
        # plt.setp(f.axes, yticks=[])
        plt.tight_layout()
        # plt.tight_layout(h_pad=3)
        # sns.plt.show()

        f.savefig(self.simulation_name + '_residue_contribution_all.png',
                  dpi=custom_dpi,
                  transparent=trasparent_alpha)


    def plot_most_contributions(self,
                                custom_dpi=600,
                                trasparent_alpha=False):
        sns.set(style="white", context="talk")

        # Set up the matplotlib figure

        # f, ax1 = plt.subplots(1, 1, figsize=(plot_tools.cm2inch(17, 10)), sharex=True)

        # Generate some sequential data





        self.most_contributions_plot = self.most_contributions[self.most_contributions['TotalEnergy'] < -1.0]

        self.most_contributions_plot = self.most_contributions_plot[
            np.isfinite(self.most_contributions_plot['TotalEnergy'])]

        # self.most_contributions_plot = self.most_contributions_plot.dropna(axis=1)
        test = 1

        # sns.barplot(self.most_contributions_plot['Residue'], self.most_contributions_plot['TotalEnergy'],
        #             palette="BuGn_d", ax=ax1)
        # cmap = sns.cubehelix_palette(n_colors=len(self.most_contributions_plot['TotalEnergy']), as_cmap=True)
        cmap = sns.dark_palette("palegreen", as_cmap=True)

        ax1 = self.most_contributions_plot.plot(x='Residue', y='TotalEnergy', yerr='TotalEnergySD', kind='bar',
                                                colormap='Blues',
                                                legend=False)

        # ax1 = self.most_contributions_plot['TotalEnergy'].plot(kind='bar')
        # ax1.bar(self.most_contributions_plot['ResidueNum'], self.most_contributions_plot['TotalEnergy'],
        #         width=40,
        #              yerr=self.most_contributions_plot['TotalEnergySD'])

        ax1.set_ylabel("Contribution Energy (kJ/mol)")
        #
        # # # Center the data to make it diverging
        # # y2 = y1 - 5
        # # sns.barplot(x, y2, palette="RdBu_r", ax=ax2)
        # # ax2.set_ylabel("Diverging")
        # #
        # # # Randomly reorder the data to make it qualitative
        # # y3 = rs.choice(y1, 9, replace=False)
        # # sns.barplot(x, y3, palette="Set3", ax=ax3)
        # # ax3.set_ylabel("Qualitative")
        #
        # # Finalize the plot
        #
        labels = ax1.get_xticklabels()  # get x labels
        # for i, l in enumerate(labels):
        #     if (i % 2 == 0): labels[i] = ''  # skip even labels
        ax1.set_xticklabels(self.most_contributions_plot['Residue'], rotation=50)  # set new labels

        # plt.show()

        #
        #
        # sns.despine(bottom=True)
        # # plt.setp(f.axes, yticks=[])
        plt.tight_layout()
        # # plt.tight_layout(h_pad=3)
        # # sns.plt.show()
        #
        plt.savefig(self.simulation_name + '_most_residue_contribution.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha)


    def plot_sorted_contributions(self,
                                  custom_dpi=600,
                                  trasparent_alpha=False,
                                  lower_criteria=-0.5,
                                  upper_criteria=0.5
                                  ):
        my_cmap = sns.light_palette("Navy", as_cmap=True)

        self.cmap_residue_energy = sns.cubehelix_palette(as_cmap=True)

        self.most_contributions_plot_all = self.most_contributions[
            (self.most_contributions['TotalEnergy'] < lower_criteria) |
            (self.most_contributions['TotalEnergy'] > upper_criteria)]

        colors_sns = sns.cubehelix_palette(n_colors=len(self.most_contributions_plot_all), dark=0.5, light=0.92,
                                           reverse=True)

        # residue_color_data = converters.convert_seaborn_color_to_rgb(colors)

        self.all_residue_colors_to_rgb = converters.convert_values_to_rgba(self.most_contributions_plot_all['TotalEnergy'],
                                                                           cmap=self.cmap_residue_energy, type='seaborn')

        # colors = sns.cubehelix_palette(n_colors=len(self.most_contributions_plot_all), dark=0.5, light=0.92, reverse=True)
        #
        # residue_color_data = converters.convert_seaborn_color_to_rgb(colors)

        # sns.palplot(colors)
        # plot_tools.custom_palplot_vertical(colors)
        # sns.plt.show()

        test = 1

        # self.most_contributions_plot_all.plot(x='Residue', y='TotalEnergy', yerr='TotalEnergySD', kind='bar',
        #                                   colormap=self.cmap_residue_energy,
        #                                   legend=False)

        # f, ax1 = plt.subplots(1, 1, figsize=(plot_tools.cm2inch(17 , 10)), sharex=True)

        sns.set(style="white", context="talk")

        self.most_contributions_plot_all.plot(x='Residue', y='TotalEnergy', yerr='TotalEnergySD', kind='bar',
                                              colors=colors_sns,
                                              legend=False)

        plt.ylabel("Contribution Energy (kJ/mol)")
        plt.xlabel("Residues")

        plt.tight_layout()
        # # plt.tight_layout(h_pad=3)
        # # sns.plt.show()
        #
        plt.savefig(self.simulation_name + '_sorted_residue_contribution.png',
                    dpi=custom_dpi,
                    transparent=trasparent_alpha)


    @hlp.timeit
    def visualize_interactions_pymol(self, show_energy=False):
        # self.clusters_centroids_mmpbsa_dict
        # self.filtered_neighbours
        test = 1

        print('Start of Pymol MD MMPBSA residue show smethod --->  ')

        print('Visualising MMPBSA  residue energy contribution')

        # To pass Values
        # self.cmap_residue_energy
        # self.most_contributions_plot_all
        #
        # self.all_residue_colors_to_rgba


        save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + \
                          'centroid:{0}_mdEnergyAnalyzer_pymolViz.pse'.format(self.sim_num)

        pymol_tools.generate_pymol_residue_energy_viz(self.centroid_pdb_file,
                                                      self.dssp_data,
                                                      self.most_contributions_plot_all,
                                                      save_state_name,

                                                      show_residue_energy=show_energy
                                                      )

        time.sleep(5)

        print('Finished Pymol method ---> verify yolo')




    # try:
    #     fout = open(args.output, 'w')
    # except:
    #     raise IOError('Could not open file {0} for writing. \n'.format(args.output))
    # try:
    #     fmap = open(args.outmap, 'w')
    # except:
    #     raise IOError('Could not open file {0} for writing. \n'.format(args.outmap))
    # fout.write(
    #     '#Residues  MM Energy(+/-)dev/error  Polar Energy(+/-)dev/error APolar Energy(+/-)dev/error Total Energy(+/-)dev/error\n')
    # for i in range(len(resname)):
    #     if (args.cutoff == 999):
    #         fout.write("%-8s  %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f \n" % (
    #             resname[i], Residues[i].FinalMM[0], Residues[i].FinalMM[1], Residues[i].FinalPol[0],
    #             Residues[i].FinalPol[1], Residues[i].FinalAPol[0], Residues[i].FinalAPol[1], Residues[i].TotalEn[0],
    #             Residues[i].TotalEn[1]))
    #     elif (args.cutoff <= Residues[i].TotalEn[0]) or ((-1 * args.cutoff) >= Residues[i].TotalEn[0]):
    #         fout.write("%-8s  %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f \n" % (
    #             resname[i], Residues[i].FinalMM[0], Residues[i].FinalMM[1], Residues[i].FinalPol[0],
    #             Residues[i].FinalPol[1], Residues[i].FinalAPol[0], Residues[i].FinalAPol[1], Residues[i].TotalEn[0],
    #             Residues[i].TotalEn[1]))
    #
    #     fmap.write("%-8d     %4.4f \n" % ((i + 1), Residues[i].TotalEn[0]))  # TODO Binding energy calculation



def autocorr(x):
    "Compute an autocorrelation with numpy"
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    return result / result[0]




def PlotEnrgy(c, FitCoef_all, args, fname):
    CompEn, CompEnErr, ExpEn, CI = [], [], [], []
    for i in range(len(c)):
        CompEn.append(c[i].FinalAvgEnergy)
        ExpEn.append(c[i].freeEn)
        CompEnErr.append(c[i].StdErr)
        CI.append(c[i].CI)
    fig = plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax = fig.add_subplot(111)
    CI = np.array(CI).T

    # To plot data
    ax.errorbar(ExpEn, CompEn, yerr=CI, fmt='o', ecolor='k', color='k', zorder=20000)

    # To plot straight line having median correlation coefficiant
    fit = np.polyfit(ExpEn, CompEn, 1)
    fitCompEn = np.polyval(fit, ExpEn)
    ax.plot(ExpEn, fitCompEn, color='k', lw=3, zorder=20000)

    # To plot straight line having minimum correlation coefficiant
    # fitCompEn = np.polyval(FitCoef[1], ExpEn)
    # ax.plot(ExpEn,fitCompEn,color='g',lw=2)

    # To plot straight line having maximum correlation coefficiant
    # fitCompEn = np.polyval(FitCoef[2], ExpEn)
    # ax.plot(ExpEn,fitCompEn,color='r',lw=2)

    for i in range(len(FitCoef_all[0])):
        fitCompEn = np.polyval([FitCoef_all[0][i], FitCoef_all[1][i]], ExpEn)
        ax.plot(ExpEn, fitCompEn, color='#BDBDBD', lw=0.5, zorder=1)

    ax.set_xlabel('Experimental Free Energy (kJ/mol)', fontsize=24, fontname='Times new Roman')
    ax.set_ylabel('Computational Binding Energy (kJ/mol)', fontsize=24, fontname='Times new Roman')
    xtics = ax.get_xticks()
    plt.xticks(xtics, fontsize=24, fontname='Times new Roman')
    ytics = ax.get_yticks()
    plt.yticks(ytics, fontsize=24, fontname='Times new Roman')
    plt.savefig(fname, dpi=300, orientation='landscape')


def PlotCorr(c, corr_outname, fname, bootstrap_nsteps):
    CompEn, ExpEn = [], []
    for i in range(len(c)):
        CompEn.append(c[i].FinalAvgEnergy)
        ExpEn.append(c[i].freeEn)
        AvgEn = np.sort(c[i].AvgEnBS, kind='mergesort')
        n = len(AvgEn)
        div = int(n / 21)
        AvgEn = AvgEn[:n:div]
        c[i].AvgEnBS = AvgEn

    main_r = np.corrcoef([CompEn, ExpEn])[0][1]
    r, FitCoef = [], []
    Id_0_FitCoef, Id_1_FitCoef = [], []

    f_corrdist = open(corr_outname, 'w')
    # Bootstrap analysis for correlation coefficiant
    nbstep = bootstrap_nsteps
    for i in range(nbstep):
        temp_x, temp_y = [], []
        energy_idx = np.random.randint(0, 22, size=len(c))
        complex_idx = np.random.randint(0, len(c), size=len(c))
        for j in range(len(complex_idx)):
            temp_y.append(c[complex_idx[j]].AvgEnBS[energy_idx[j]])
            temp_x.append(c[complex_idx[j]].freeEn)
        rtmp = np.corrcoef([temp_x, temp_y])[0][1]
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        r.append(rtmp)
        fit = np.polyfit(temp_x, temp_y, 1)
        FitCoef.append(fit)

        f_corrdist.write('{0}\n'.format(rtmp))

        # Seprating Slope and intercept
    Id_0_FitCoef = np.transpose(FitCoef)[0]
    Id_1_FitCoef = np.transpose(FitCoef)[1]

    # Calculating mode of coorelation coefficiant
    density, r_hist = np.histogram(r, 25, normed=True)
    mode = (r_hist[np.argmax(density) + 1] + r_hist[np.argmax(density)]) / 2

    # Calculating Confidence Interval
    r = np.sort(r)
    CI_min_idx = int(0.005 * nbstep)
    CI_max_idx = int(0.995 * nbstep)
    CI_min = mode - r[CI_min_idx]
    CI_max = r[CI_max_idx] - mode
    print("%5.3f %5.3f  %5.3f  %5.3f" % (main_r, mode, CI_min, CI_max))

    # Plotting Correlation Coefficiant Distribution
    fig = plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(r, 40, normed=1, facecolor='#B2B2B2', alpha=0.75, lw=0.1)
    plt.title('Mode = {0:.3f}\nConf. Int. = -{1:.3f}/+{2:.3f}'.format(mode, CI_min, CI_max), fontsize=18,
              fontname='Times new Roman')
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    # y = mlab.normpdf( bincenters, mode, np.std(r))
    # l = ax.plot(bincenters, y, 'k--', lw=1)
    ax.set_xlabel('Correlation Coefficient', fontsize=24, fontname='Times new Roman')
    ax.set_ylabel('Density', fontsize=24, fontname='Times new Roman')
    xtics = ax.get_xticks()
    plt.xticks(xtics, fontsize=24, fontname='Times new Roman')
    ytics = ax.get_yticks()
    plt.yticks(ytics, fontsize=24, fontname='Times new Roman')
    plt.savefig(fname, dpi=300, orientation='landscape')
    return [Id_0_FitCoef, Id_1_FitCoef]


class Complex(object):
    def __init__(self, MmFile, PolFile, APolFile):
        self.frames = []
        self.TotalEn = []
        self.Vdw, self.Elec, self.Pol, self.Sas, self.Sav, self.Wca = [], [], [], [], [], []
        self.MmFile = MmFile
        self.PolFile = PolFile
        self.APolFile = APolFile
        self.AvgEnBS = []
        self.CI = []
        self.FinalAvgEnergy = 0
        self.StdErr = 0



    def jump_data_conv(self, data, jump_data):
        temp_data = []

        for tempus in data:
            new_temp = tempus[::jump_data]
            temp_data.append(new_temp)

        return temp_data

    def CalcEnergy(self, frame_wise, idx, jump_data=1, bootstrap=False, bootstrap_jump=4,  bootstrap_steps=None):

        mmEn = ReadData(self.MmFile, n=7)

        mmEn = ReadData(self.MmFile, n=7)
        polEn = ReadData(self.PolFile, n=4)
        apolEn = ReadData(self.APolFile, n=10)



        if jump_data>1:
            mmEn = self.jump_data_conv( mmEn, jump_data)
            polEn = self.jump_data_conv(polEn, jump_data)
            apolEn = self.jump_data_conv(apolEn, jump_data)



        CheckEnData(mmEn, polEn, apolEn)



        time, MM, Vdw, Elec, Pol, Apol, Sas, Sav, Wca = [], [], [], [], [], [], [], [], []
        for i in range(len(mmEn[0])):
            # Vacuum MM
            Energy = mmEn[5][i] + mmEn[6][i] - (mmEn[1][i] + mmEn[2][i] + mmEn[3][i] + mmEn[4][i])
            MM.append(Energy)
            Energy = mmEn[5][i] - (mmEn[1][i] + mmEn[3][i])
            Vdw.append(Energy)
            Energy = mmEn[6][i] - (mmEn[2][i] + mmEn[4][i])
            Elec.append(Energy)
            # Polar
            Energy = polEn[3][i] - (polEn[1][i] + polEn[2][i])
            Pol.append(Energy)
            # Non-polar
            Energy = apolEn[3][i] + apolEn[6][i] + apolEn[9][i] - (
            apolEn[1][i] + apolEn[2][i] + apolEn[4][i] + apolEn[5][i] + apolEn[7][i] + apolEn[8][i])
            Apol.append(Energy)
            Energy = apolEn[3][i] - (apolEn[1][i] + apolEn[2][i])
            Sas.append(Energy)
            Energy = apolEn[6][i] - (apolEn[4][i] + apolEn[5][i])
            Sav.append(Energy)
            Energy = apolEn[9][i] - (apolEn[7][i] + apolEn[8][i])
            Wca.append(Energy)
            # Final Energy
            time.append(mmEn[0][i])
            Energy = MM[i] + Pol[i] + Apol[i]
            self.TotalEn.append(Energy)


        # TODO HISTOGRAM NEED TO DO SOMETHING
        # TAKE A VERY CAREFUL LOOK
        # https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
        plt.clf()
        plt.hist(self.TotalEn)
        plt.show()
        plt.clf()


        self.time = time
        self.time_bootstrap = time[::bootstrap_jump]
        self.TotalEn_bootstrap = self.TotalEn[::bootstrap_jump]

        # Writing frame wise component energy to file
        frame_wise.write('\n#Complex %d\n' % ((idx + 1)))
        for i in range(len(time)):
            frame_wise.write('%15.3lf %15.3lf %15.3lf %15.3lf %15.3lf' % (
            time[i], mmEn[1][i], mmEn[2][i], polEn[1][i], (apolEn[1][i] + apolEn[4][i] + apolEn[7][i])))
            frame_wise.write('%15.3lf %15.3lf %15.3lf %15.3lf' % (
            mmEn[3][i], mmEn[4][i], polEn[2][i], (apolEn[2][i] + apolEn[5][i] + apolEn[8][i])))
            frame_wise.write('%15.3lf %15.3lf %15.3lf %15.3lf' % (
            mmEn[5][i], mmEn[6][i], polEn[3][i], (apolEn[3][i] + apolEn[6][i] + apolEn[9][i])))
            frame_wise.write('%15.3lf %15.3lf %15.3lf %15.3lf\n' % (MM[i], Pol[i], Apol[i], self.TotalEn[i]))

            # Bootstrap analysis energy components
        if bootstrap is True:
            bsteps = bootstrap_steps
            curr_Vdw = Vdw[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Vdw, bsteps)
            self.Vdw.append(avg_energy)
            self.Vdw.append(error)

            curr_Elec = Elec[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Elec, bsteps)
            self.Elec.append(avg_energy)
            self.Elec.append(error)

            curr_Pol = Pol[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Pol, bsteps)
            self.Pol.append(avg_energy)
            self.Pol.append(error)

            curr_Sas = Sas[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Sas, bsteps)
            self.Sas.append(avg_energy)
            self.Sas.append(error)

            curr_Sav = Sav[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Sav, bsteps)
            self.Sav.append(avg_energy)
            self.Sav.append(error)

            curr_Wca = Wca[::bootstrap_jump]
            avg_energy, error = BootStrap(curr_Wca, bsteps)
            self.Wca.append(avg_energy)
            self.Wca.append(error)

            # Bootstrap => Final Average Energy
            curr_TotalEn = self.TotalEn_bootstrap

            #from matplotlib import pyplot



            self.AvgEnBS, AvgEn, EnErr, CI = ComplexBootStrap(curr_TotalEn, bsteps)
            self.FinalAvgEnergy = AvgEn
            self.StdErr = EnErr
            self.CI = CI
            # If not bootstrap then average and standard deviation
        else:
            self.Vdw.append(np.mean(Vdw))
            self.Vdw.append(np.std(Vdw))
            self.Elec.append(np.mean(Elec))
            self.Elec.append(np.std(Elec))
            self.Pol.append(np.mean(Pol))
            self.Pol.append(np.std(Pol))
            self.Sas.append(np.mean(Sas))
            self.Sas.append(np.std(Sas))
            self.Sav.append(np.mean(Sav))
            self.Sav.append(np.std(Sav))
            self.Wca.append(np.mean(Wca))
            self.Wca.append(np.std(Wca))
            self.FinalAvgEnergy = np.mean(self.TotalEn)
            self.StdErr = np.std(self.TotalEn)


def Summary_Output_File(AllComplex, output_name, meta_file=None):
    try:
        fs = open(output_name, 'w')
    except:
        raise IOError('Could not open file {0} for writing. \n'.format(args.outsum))

    if meta_file:
        try:
            fm = open(output_name + '_meta_verify yolo.txt', 'w')
        except:
            raise IOError('Could not open file {0} for writing. \n'.format(args.outmeta))
        fm.write('# Complex_Number\t\tTotal_Binding_Energy\t\tError\n')

    for n in range(len(AllComplex)):
        fs.write('\n\n#Complex Number: %4d\n' % (n + 1))
        fs.write('===============\n   SUMMARY   \n===============\n\n')
        fs.write('\n van der Waal energy      = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Vdw[0], AllComplex[n].Vdw[1]))
        fs.write('\n Electrostattic energy    = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Elec[0], AllComplex[n].Elec[1]))
        fs.write('\n Polar solvation energy   = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Pol[0], AllComplex[n].Pol[1]))
        fs.write('\n SASA energy              = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Sas[0], AllComplex[n].Sas[1]))
        fs.write('\n SAV energy               = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Sav[0], AllComplex[n].Sav[1]))
        fs.write('\n WCA energy               = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].Wca[0], AllComplex[n].Wca[1]))
        fs.write('\n Binding energy           = %15.3lf   +/-  %7.3lf kJ/mol\n' % (
        AllComplex[n].FinalAvgEnergy, AllComplex[n].StdErr))
        fs.write('\n===============\n    END     \n===============\n\n')

        if meta_file:
            fm.write('%5d %15.3lf %7.3lf\n' % (n + 1, AllComplex[n].FinalAvgEnergy, AllComplex[n].StdErr))


def CheckEnData(mmEn, polEn, apolEn):
    frame = len(mmEn[0])
    for i in range(len(mmEn)):
        if (len(mmEn[i]) != frame):
            raise ValueError("In MM file, size of columns are not equal.")

    for i in range(len(polEn)):
        if (len(polEn[i]) != frame):
            raise ValueError("In Polar file, size of columns are not equal.")

    for i in range(len(apolEn)):
        if (len(apolEn[i]) != frame):
            raise ValueError("In APolar file, size of columns are not equal.")


def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--multiple",
                        help='If given, calculate for multiple complexes. Need Metafile containing path of energy files',
                        action="store_true")
    parser.add_argument("-mf", "--metafile", help='Metafile containing path to energy files of each complex in a row obtained from g_mmpbsa in following order: \
                                                       [MM file] [Polar file] [ Non-polar file] [Ki]  \
						       Ki Should be in NanoMolar (nM)', action="store", default='metafile.dat',
                        metavar='metafile.dat')
    parser.add_argument("-m", "--molmech", help='Vacuum Molecular Mechanics energy file obtained from g_mmpbsa',
                        action="store", default='energy_MM.xvg', metavar='energy_MM.xvg')
    parser.add_argument("-p", "--polar", help='Polar solvation energy file obtained from g_mmpbsa', action="store",
                        default='polar.xvg', metavar='polar.xvg')
    parser.add_argument("-a", "--apolar", help='Non-Polar solvation energy file obtained from g_mmpbsa', action="store",
                        default='apolar.xvg', metavar='apolar.xvg')
    parser.add_argument("-bs", "--bootstrap", help='If given, Enable Boot Strap analysis', action="store_true")
    parser.add_argument("-nbs", "--nbstep", help='Number of boot strap steps for average energy calculation',
                        action="store", type=int, default=1000)
    parser.add_argument("-of", "--outfr", help='Energy File: Energy components frame wise', action="store",
                        default='full_energy.dat', metavar='full_energy.dat')
    parser.add_argument("-os", "--outsum", help='Final Energy File: Full Summary of energy components', action="store",
                        default='summary_energy.dat', metavar='summary_energy.dat')
    parser.add_argument("-om", "--outmeta",
                        help='Final Energy File for Multiple Complexes: Complex wise final binding nergy',
                        action="store", default='meta_energy.dat', metavar='meta_energy.dat')
    parser.add_argument("-ep", "--enplot", help='Experimental Energy vs Calculated Energy Correlation Plot',
                        action="store", default='enplot.png', metavar='enplot.png')
    parser.add_argument("-cd", "--corrdist", help='Correlation distribution data from bootstrapping', action="store",
                        default='corrdist.dat', metavar='corrdist.dat')
    parser.add_argument("-cp", "--corrplot", help='Plot of correlation distribution', action="store",
                        default='corrdist.png', metavar='corrdist.png')

    if len(sys.argv) < 2:
        print('ERROR: No input files. Need help!!!')
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.multiple:
        if not os.path.exists(args.metafile):
            print('\nERROR: {0} not found....\n'.format(args.metafile))
            parser.print_help()
            sys.exit(1)
    else:
        if not os.path.exists(args.molmech):
            print('\nERROR: {0} not found....\n'.format(args.molmech))
            parser.print_help()
            sys.exit(1)
        if not os.path.exists(args.polar):
            print('\nERROR: {0} not found....\n'.format(args.polar))
            parser.print_help()
            sys.exit(1)
        if not os.path.exists(args.apolar):
            print('\nERROR: {0} not found....\n'.format(args.apolar))
            parser.print_help()
            sys.exit(1)

    return args


def ReadData(FileName, n=2):
    infile = open(FileName, 'r')
    x, data = [], []
    for line in infile:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        if (re.match('#|@', line) == None):
            temp = line.split()
            data.append(np.array(temp))
    for j in range(0, n):
        x_temp = []
        for i in range(len(data)):
            try:
                value = float(data[i][j])
            except:
                raise FloatingPointError(
                    '\nCould not convert {0} to floating point number.. Something is wrong in {1}..\n'.format(
                        data[i][j], FileName))

            x_temp.append(value)

        x.append(x_temp)
    return x


def ComplexBootStrap(x, step=1000):
    avg = []
    x = np.array(x)
    n = len(x)
    idx = np.random.randint(0, n, (step, n))
    sample_x = x[idx]
    avg = np.sort(np.mean(sample_x, 1))
    CI_min = avg[int(0.005 * step)]
    CI_max = avg[int(0.995 * step)]

    import scipy
    import scikits.bootstrap as bootstrap
    CIs = bootstrap.ci(data=x, statfunction=scipy.mean, n_samples=20000, alpha=0.005)

    print("Bootstrapped 99.5% confidence intervals\nLow:", CIs[0], "\nHigh:", CIs[1])
    print('------------------------------------------------------------------------------')

    print("Bootstrapped 99.5% confidence intervals ORIGINAL\nLow:", CI_min, "\nHigh:", CI_max)

    # print('Energy = %13.3f; Confidance Interval = (-%-5.3f / +%-5.3f)\n' % (np.mean(avg), (np.mean(avg)-CI_min), (CI_max-np.mean(avg))))
    return avg, np.mean(avg), np.std(avg), [(np.mean(avg) - CI_min), (CI_max - np.mean(avg))]


def BootStrap(x, step=1000):
    if (np.mean(x)) == 0:
        return 0.000, 0.000
    else:
        avg = []
        x = np.array(x)
        n = len(x)
        idx = np.random.randint(0, n, (step, n))
        sample_x = x[idx]
        avg = np.sort(np.mean(sample_x, 1))
        return np.mean(avg), np.std(avg)


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def ReadMetafile(metafile):
    MmFile, PolFile, APolFile, Ki = [], [], [], []
    FileList = open(metafile, 'r')
    for line in FileList:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        temp = line.split()
        MmFile.append(temp[0])
        PolFile.append(temp[1])
        APolFile.append(temp[2])
        Ki.append(float(temp[3]))

        if not os.path.exists(temp[0]):
            raise IOError('Could not open file {0} for reading. \n'.format(temp[0]))

        if not os.path.exists(temp[1]):
            raise IOError('Could not open file {0} for reading. \n'.format(temp[1]))

        if not os.path.exists(temp[2]):
            raise IOError('Could not open file {0} for reading. \n'.format(temp[2]))

    return MmFile, PolFile, APolFile, Ki


# TODO RESIDUE CALUCALTION
def main():
    args = ParseOptions()
    MMEnData, resnameA = ReadData_Residue_Parse(args.molmech)
    polEnData, resnameB = ReadData_Residue_Parse(args.polar)
    apolEnData, resnameC = ReadData_Residue_Parse(args.apolar)
    resname = CheckResname(resnameA, resnameB, resnameC)
    print('Total number of Residue: {0}\n'.format(len(resname) + 1))
    Residues = []
    for i in range(len(resname)):
        CheckEnData(MMEnData[i], polEnData[i], apolEnData[i])
        r = Residue()
        r.CalcEnergy(MMEnData[i], polEnData[i], apolEnData[i], args)
        Residues.append(r)
        print(' %8s %8.4f %8.4f' % (resname[i], r.TotalEn[0], r.TotalEn[1]))
    try:
        fout = open(args.output, 'w')
    except:
        raise IOError('Could not open file {0} for writing. \n'.format(args.output))
    try:
        fmap = open(args.outmap, 'w')
    except:
        raise IOError('Could not open file {0} for writing. \n'.format(args.outmap))
    fout.write(
        '#Residues  MM Energy(+/-)dev/error  Polar Energy(+/-)dev/error APolar Energy(+/-)dev/error Total Energy(+/-)dev/error\n')
    for i in range(len(resname)):
        if (args.cutoff == 999):
            fout.write("%-8s  %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f \n" % (
                resname[i], Residues[i].FinalMM[0], Residues[i].FinalMM[1], Residues[i].FinalPol[0],
                Residues[i].FinalPol[1], Residues[i].FinalAPol[0], Residues[i].FinalAPol[1], Residues[i].TotalEn[0],
                Residues[i].TotalEn[1]))
        elif (args.cutoff <= Residues[i].TotalEn[0]) or ((-1 * args.cutoff) >= Residues[i].TotalEn[0]):
            fout.write("%-8s  %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f    %4.4f  %4.4f \n" % (
                resname[i], Residues[i].FinalMM[0], Residues[i].FinalMM[1], Residues[i].FinalPol[0],
                Residues[i].FinalPol[1], Residues[i].FinalAPol[0], Residues[i].FinalAPol[1], Residues[i].TotalEn[0],
                Residues[i].TotalEn[1]))

        fmap.write("%-8d     %4.4f \n" % ((i + 1), Residues[i].TotalEn[0]))


class Residue(object):
    def __init__(self):
        self.FinalMM, self.FinalPol, self.FinalAPol, self.TotalEn = [], [], [], []

    def BootStrap(self, x, step):
        avg = []
        x = np.array(x)
        n = len(x)
        idx = np.random.randint(0, n, (step, n))
        sample_x = x[idx]
        avg = np.sort(np.mean(sample_x, 1))
        return np.mean(avg), np.std(avg)

    def CalcEnergy(self, MM, Pol, APol, bootstrap, nbstep):
        TotalEn = np.sum([MM, Pol, APol], axis=0)
        if (bootstrap):
            self.FinalMM = self.BootStrap(MM, nbstep)
            self.FinalPol = self.BootStrap(Pol, nbstep)
            if (np.mean(APol) == 0):
                self.FinalAPol = [0.0, 0.0]
            else:
                self.FinalAPol = self.BootStrap(APol, nbstep)
            self.TotalEn = self.BootStrap(TotalEn, nbstep)
        else:
            self.FinalMM = [np.mean(MM), np.std(MM)]
            self.FinalPol = [np.mean(Pol), np.std(Pol)]
            if (np.mean(APol) == 0):
                self.FinalAPol = [0.0, 0.0]
            else:
                self.FinalAPol = [np.mean(APol), np.std(APol)]
            self.TotalEn = [np.mean(TotalEn), np.std(TotalEn)]
        self.FinalMM = np.round(self.FinalMM, 4)
        self.FinalPol = np.round(self.FinalPol, 4)
        self.FinalAPol = np.round(self.FinalAPol, 4)
        self.TotalEn = np.round(self.TotalEn, 4)


def CheckEnData_residue(MM, Pol, APol):
    if (len(Pol) != len(MM)):
        raise ValueError("Times or Frame numbers Mismatch between polar and MM input files")
    if (len(APol) != len(Pol)):
        raise ValueError("Times or Frame numbers Mismatch between apolar and polar input files")
    if (len(APol) != len(MM)):
        raise ValueError("Times or Frame numbers Mismatch between apolar and MM input files")


def ParseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--molmech", help='Molecular Mechanics energy file', action="store",
                        default='contrib_MM.dat', metavar='contrib_MM.dat')
    parser.add_argument("-p", "--polar", help='Polar solvation energy file', action="store", default='contrib_pol.dat',
                        metavar='contrib_pol.dat')
    parser.add_argument("-a", "--apolar", help='Non-Polar solvation energy file', action="store",
                        default='contrib_apol.dat', metavar='contrib_apol.dat')
    parser.add_argument("-bs", "--bootstrap", help='Switch for Error by Boot Strap analysis', action="store_true")
    parser.add_argument("-nbs", "--nbstep", help='Number of boot strap steps', action="store", type=int, default=500,
                        metavar=500)
    parser.add_argument("-ct", "--cutoff", help='Absolute Cutoff: energy output above and below this value',
                        action="store", type=float, default=999, metavar=999)
    parser.add_argument("-o", "--output", help='Final Decomposed Energy File', action="store",
                        default='final_contrib_energy.dat', metavar='final_contrib_energy.dat')
    parser.add_argument("-om", "--outmap", help='energy2bfac input file: to map energy on structure for visualization',
                        action="store", default='energyMapIn.dat', metavar='energyMapIn.dat')

    if len(sys.argv) < 2:
        print('\nERROR: No input files. Need help!!!\n')
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if not os.path.exists(args.molmech):
        print('\n{0} not found....\n'.format(args.molmech))
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.polar):
        print('\n{0} not found....\n'.format(args.polar))
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.apolar):
        print('\n{0} not found....\n'.format(args.apolar))
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def CheckResname(resA, resB, resC):
    if (len(resA) != len(resB)):
        raise ValueError("Total number of residues mismatch between MM and Polar input files")

    if (len(resB) != len(resC)):
        raise ValueError("Total number of residues mismatch between Polar and Apolar input files")

    if (len(resC) != len(resA)):
        raise ValueError("Total number of residues mismatch between MM and apolar input files")

    for i in range(len(resA)):
        if (resA[i] != resB[i]):
            raise ValueError("Residue mismatch at index {0}: MM -> {1} =/= {2} <- Polar .".format(i, resA[i], resB[i]))

    for i in range(len(resB)):
        if (resB[i] != resC[i]):
            raise ValueError(
                "Residue mismatch at index {0}: Polar -> {1} =/= {2} <- APolar .".format(i, resB[i], resC[i]))

    for i in range(len(resA)):
        if (resA[i] != resC[i]):
            raise ValueError("Residue mismatch at index {0}: MM -> {1} =/= {2} <- APolar .".format(i, resA[i], resC[i]))

    return resA


def ReadData_Residue_Parse(FileName):
    try:
        infile = open(FileName, 'r')
    except:
        raise IOError('\n{0} could not open....\n'.format(FileName))

    x, data, resname = [], [], []
    for line in infile:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        if (re.match('#|@', line) == None):
            temp = line.split()
            data.append(np.array(temp))
        if (re.match('#', line)):
            resname = line.split()

    n = len(resname[1:])
    for j in range(1, n):
        x_temp = []
        for i in range(len(data)):
            try:
                value = float(data[i][j])
            except:
                raise FloatingPointError(
                    '\nCould not convert {0} to floating point number.. Something is wrong in {1}..\n'.format(
                        data[i][j], FileName))

            x_temp.append(value)
        x.append(x_temp)
    return x, resname[2:]

# if __name__ == "__main__":
#     main()
