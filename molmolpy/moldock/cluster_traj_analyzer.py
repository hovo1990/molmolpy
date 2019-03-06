#!/usr/bin/env python
#
# @file    cluster_traj_analyzer.py
# @brief   centroid analyzer from molecular docking results
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

import os
import sys
import pickle

from multiprocessing import Pool
import multiprocessing

import csv

import mdtraj as md  # Problem with it
import numpy as np

from molmolpy.utils import folder_utils
from molmolpy.utils import pickle_tools
from molmolpy.utils import converters
from molmolpy.utils import pdb_tools
from molmolpy.utils import plot_tools
from molmolpy.moldock import topology_reconstruction
from sklearn.utils.extmath import row_norms


class ClusterTrajAnalyzerObject(object):
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

    def __init__(self, docking_analysis_object, run_type='standard', folder_path='cluster_traj'):

        self.run_type = 'standard'

        self.docking_analysis_object = docking_analysis_object

        self.docking_analysis_object_type = self.docking_analysis_object.analysis_type

        self.analysis_structure = docking_analysis_object.analysis_structure

        self.molecule_name = self.docking_analysis_object.molecule_name
        self.receptor_name = self.docking_analysis_object.receptor_name

        # folder_info = folder_utils.find_folders(folder_path)
        # self.folder_path = self.molecule_name + '_' +  folder_path

        self.folder_path = docking_analysis_object.save_cluster_models_dir

        self.curr_index = 0

        self.correct_topology = None

        if self.docking_analysis_object_type == 'ultra':
            self.extracted_files_list = self.docking_analysis_object.save_extract_files_list
            #print(self.extracted_files_list)

            print('----------------------------------------------------------------------------------')
            # self.pdb_files = folder_utils.find_files_in_folder('.' + os.sep + self.folder_path, data_format='pdb',
            #                                                    exclude=['centroid.pdb', 'simplified.pdb'],
            #                                                    include='ligBindTraj')
            # print(self.pdb_files)
            # test = 1
        else:
            self.cluster_colors_pre_rgb = self.docking_analysis_object.colors_
            self.cluster_colors_rgb = converters.convert_seaborn_color_to_rgb(self.cluster_colors_pre_rgb)
            self.cluster_colors = self.docking_analysis_object.cluster_colors

            try:
                self.pdb_files = folder_utils.find_files_in_folder('.' + os.sep + self.folder_path, data_format='pdb',
                                                                   exclude=['centroid.pdb', 'simplified.pdb'],
                                                                   include='ligBindTraj')
                print(self.pdb_files)

                # self.pickle_files = folder_utils.find_files_in_folder('.' +os.sep + self.folder_path, data_format='pickle')

                # self.cluster_data_object = pickle_tools.load_pickle_data('.' +os.sep + self.folder_path + os.sep + self.pickle_files[0])

            except Exception as e:
                print('Problem with finding files')
                print('Error is ', e)
                sys.exit(1)


                # self.extract_centroids_auto()

    def extract_centroids_auto_custom(self):
        print('\n')
        print('----------------------------------------------------')
        print('Extracting Ultra analysis Centroids Now')

        keys = list(self.extracted_files_list.keys())

        self.full_data_mdtraj_analysis = {}
        for key in keys:
            self.centroid_data = {}
            print('----->Type of data based on {0}  ---<\n'.format(key))
            indexes = list(range(len(self.extracted_files_list[key])))

            for index in self.extracted_files_list[key]:
                # self.curr_index = self.extracted_files_list[key][index]['key']
                self.curr_index = index
                print('Curr cluster traj index ', self.curr_index)

                # TODO pdb_file wrong
                self.centroid_data.update({self.curr_index: {}})
                data_to_work_with = self.extracted_files_list[key][index]
                # ligand_conf = '.' + os.sep + self.folder_path + os.sep + data_to_work_with
                self.find_centroid_custom(data_to_work_with)
                print('-------------------')
                test = 1

            self.full_data_mdtraj_analysis.update({key: self.centroid_data})
            test = 1

        test = 1

        # self.centroid_data = {}
        # for pdb_file, index in zip(self.pdb_files, indexes):
        #     print('Curr cluster traj index ', index)
        #     self.curr_index = index
        #
        #     self.centroid_data.update({index: {}})
        #     ligand_conf = '.' + os.sep + self.folder_path + os.sep + pdb_file
        #     self.find_centroid(ligand_conf, pdb_file)
        #     print('---------------------------------------------------------\n')

    def find_centroid_custom(self, data_to_work_with, overwrite_file=True):
        print('\n')
        pre_process = data_to_work_with['filename'].split('.')[0]

        ligand_conf = '.' + os.sep + self.folder_path + os.sep + data_to_work_with['filename']

        ligand_save_name = pre_process

        traj = md.load(ligand_conf)
        real_originalData = traj[:]

        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((traj.n_frames, traj.n_frames))
        for i in range(traj.n_frames):
            distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

        print('len(distances) ', len(distances))
        print('Max pairwise rmsd: %f nm' % np.max(distances))
        print('Min pairwise rmsd: %f nm' % np.min(distances))

        # This is the part for centroid
        beta = 1
        index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        print('Index for representative structure ', index)

        centroid = traj[index]
        print(centroid)

        bindEner_curr = data_to_work_with['currModels'][index]['molDetail']['vina_info'][0]
        print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))

        print('Color of represenative structure is {0}'.format(data_to_work_with['colors']))
        print('RGB Color of represenative structure is {0}'.format(data_to_work_with['rgbColors']))

        actual_name, closest_name = plot_tools.get_cluster_color_name(data_to_work_with['rgbColors'])

        print('----------------------------------------------------\n')

        toWrite = real_originalData[index]
        print('toWrite checking is ', toWrite)

        centroid_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_centroid.pdb'
        print('centroid file is ', centroid_file)
        toWrite.save(centroid_file, force_overwrite=overwrite_file)

        self.centroid_data[self.curr_index].update({'mdtraj_cluster': real_originalData})
        self.centroid_data[self.curr_index].update({'ligand_save_name': ligand_save_name})

        self.centroid_data[self.curr_index].update({'ligand_index': self.curr_index})
        self.centroid_data[self.curr_index].update({'centroid_index': index})

        self.centroid_data[self.curr_index].update({'centroid_mdtraj_pre': centroid})
        self.centroid_data[self.curr_index].update({'centroid_mdtraj': toWrite})
        self.centroid_data[self.curr_index].update({'centroid_file': centroid_file})
        self.centroid_data[self.curr_index].update({'distances': distances})

        print('verify yolo ', self.curr_index)
        # self.centroid_data[self.curr_index].update({'color': self.cluster_colors_rgb[self.curr_index]})
        self.centroid_data[self.curr_index].update({'color': data_to_work_with['rgbColors']})
        self.centroid_data[self.curr_index].update({'colorOriginal': data_to_work_with['colors']})

        test = 1



    def extract_centroids_auto_ultra(self):
        '''
        Find centroids from clustered docking objects
        :return:
        '''
        print('\n')
        print('----------------------------------------------------')
        print('Extracting Ultra Centroids Now')



        print(self.extracted_files_list.keys())

        to_extract = self.extracted_files_list['centroid']
        print(to_extract.keys())
        test = 1

        self.centroid_data = {}
        for index in to_extract:
            data = to_extract[index]
            print(data.keys())


            relativePath = data['relativePath']
            filename = data['filename']
            colors = data['colors']
            rgbColors = data['rgbColors']
            currModels = data['currModels']
            key = data['key']
            percentage = data['percentage']

            self.find_centroid_ultra(relativePath, filename, colors, rgbColors, currModels, key, percentage)

            test = 1
        # indexes = list(range(len(self.pdb_files)))
        #
        # self.centroid_data = {}
        # for pdb_file, index in zip(self.pdb_files, indexes):
        #     print('Curr cluster traj index ', index)
        #     self.curr_index = index
        #
        #     self.centroid_data.update({index: {}})
        #     ligand_conf = '.' + os.sep + self.folder_path + os.sep + pdb_file
        #     self.find_centroid(ligand_conf, pdb_file)
        #     print('---------------------------------------------------------\n')



    def find_centroid_ultra(self, relativePath, filename, colors, rgbColors, currModels, key, percentage, overwrite_file=True):
        ''''''
        print('\n')
        print('Current key is ', key)

        self.centroid_data.update({key:{}})


        pre_process = filename.split('.')[0]
        ligand_save_name = pre_process

        traj = md.load(relativePath)
        real_originalData = traj[:]

        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((traj.n_frames, traj.n_frames))
        for i in range(traj.n_frames):
            distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

        print('len(distances) ', len(distances))
        print('Max pairwise rmsd: %f nm' % np.max(distances))
        print('Min pairwise rmsd: %f nm' % np.min(distances))

        # This is the part for centroid
        beta = 1
        index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        print('Index for representative structure ', index)

        centroid = traj[index]
        print(centroid)



        # print('currModels is ', currModels)
        # print(type(currModels))

        print(currModels[index].keys())
        print(currModels[index]['molDetail'].keys())
        bindEner_curr = currModels[index]['molDetail']['vina_info'][0]
        print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))



        print('Color of represenative structure is {0}'.format(rgbColors))
        print('----------------------------------------------------\n')

        toWrite = real_originalData[index]
        print('toWrite checking is ', toWrite)

        centroid_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_centroid.pdb'
        print('centroid file is ', centroid_file)
        toWrite.save(centroid_file, force_overwrite=overwrite_file)

        self.centroid_data[key].update({'mdtraj_cluster': real_originalData})
        self.centroid_data[key].update({'ligand_save_name': ligand_save_name})

        self.centroid_data[key].update({'ligand_index': key})
        self.centroid_data[key].update({'centroid_index': index})

        self.centroid_data[key].update({'centroid_mdtraj_pre': centroid})
        self.centroid_data[key].update({'centroid_mdtraj': toWrite})
        self.centroid_data[key].update({'centroid_file': centroid_file})
        self.centroid_data[key].update({'distances': distances})

        print('verify yolo ', self.curr_index)

        self.centroid_data[self.curr_index].update({'color': rgbColors})

        test = 1





    def extract_centroids_auto(self):
        '''
        Find centroids from clustered docking objects
        :return:
        '''
        print('\n')
        print('----------------------------------------------------')
        print('Extracting Centroids Now')
        indexes = list(range(len(self.pdb_files)))

        self.centroid_data = {}
        for pdb_file, index in zip(self.pdb_files, indexes):
            print('Curr cluster traj index ', index)
            self.curr_index = index

            self.centroid_data.update({index: {}})
            ligand_conf = '.' + os.sep + self.folder_path + os.sep + pdb_file
            self.find_centroid(ligand_conf, pdb_file)
            print('---------------------------------------------------------\n')

    def find_centroid(self, ligand_conf, pdb_file=None, overwrite_file=True, show_color=False):
        print('\n')
        pre_process = pdb_file.split('.')[0]
        ligand_save_name = pre_process

        traj = md.load(ligand_conf)
        real_originalData = traj[:]

        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
        distances = np.empty((traj.n_frames, traj.n_frames))
        for i in range(traj.n_frames):
            distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

        print('len(distances) ', len(distances))
        print('Max pairwise rmsd: %f nm' % np.max(distances))
        print('Min pairwise rmsd: %f nm' % np.min(distances))

        # This is the part for centroid
        beta = 1
        index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
        print('Index for representative structure ', index)

        centroid = traj[index]
        print(centroid)

        bindEner_curr = self.analysis_structure.iloc[index]['BindingEnergy']
        print('Energy for index {0} centroid is {1} kcal/mol'.format(index, bindEner_curr))


        if show_color is True:
            print('Color of represenative structure is {0}'.format(self.cluster_colors_rgb[self.curr_index]))
            print('----------------------------------------------------\n')

        toWrite = real_originalData[index]
        print('toWrite checking is ', toWrite)

        centroid_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_centroid.pdb'
        print('centroid file is ', centroid_file)
        toWrite.save(centroid_file, force_overwrite=overwrite_file)

        self.centroid_data[self.curr_index].update({'mdtraj_cluster': real_originalData})
        self.centroid_data[self.curr_index].update({'ligand_save_name': ligand_save_name})

        self.centroid_data[self.curr_index].update({'ligand_index': self.curr_index})
        self.centroid_data[self.curr_index].update({'centroid_index': index})

        self.centroid_data[self.curr_index].update({'centroid_mdtraj_pre': centroid})
        self.centroid_data[self.curr_index].update({'centroid_mdtraj': toWrite})
        self.centroid_data[self.curr_index].update({'centroid_file': centroid_file})
        self.centroid_data[self.curr_index].update({'distances': distances})

        print('verify yolo ', self.curr_index)

        if show_color is True:
            self.centroid_data[self.curr_index].update({'color': self.cluster_colors_rgb[self.curr_index]})

        test = 1

    def write_simplified_centroids(self, ligand_centers, ligand_index, ligand_save_name):
        simplified_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_simplified.pdb'
        file_to_write = open(simplified_file, 'w')
        for row_num in range(len(ligand_centers)):
            # print(row)

            col1 = 'ATOM'

            col2 = row_num + 1

            col3 = 'C'
            col4 = 'DUM'

            col5 = 'Z'

            col6 = ligand_index

            col7 = self.conformation_centers[row_num][0]
            col8 = self.conformation_centers[row_num][1]
            col9 = self.conformation_centers[row_num][2]

            col10 = 1.00
            col11 = 0.00
            col12 = 'C'

            row = [col1, col2, col3, col4, col5,
                   col6, col7, col8, col9, col10, col11, col12]

            writeRow = pdb_tools.write_row_pdb(row)
            # print('tada ',writeRow)
            file_to_write.write(writeRow)
        # pdbParseWrite.write_Lig(molData,file_to_write)
        file_to_write.close()

    # TODO don't use this function use VMD
    # TODO save only part of the atoms, too many

    def simplified_cluster_extraction(self):

        for centoid_data_index in self.centroid_data:
            md_traj_cluster = self.centroid_data[centoid_data_index]['mdtraj_cluster']
            ligand_save_name = self.centroid_data[centoid_data_index]['ligand_save_name']
            ligand_index = self.centroid_data[centoid_data_index]['ligand_index']

            coordinates = md_traj_cluster.xyz

            topology = md_traj_cluster.topology

            table, bonds = topology.to_dataframe()

            t2 = md.Topology.from_dataframe(table, bonds)

            conformation_centers = np.mean(md_traj_cluster.xyz, axis=1)

            # self.conformation_centers = conformation_centers

            self.write_simplified_centroids(ligand_centers=conformation_centers, ligand_index=ligand_index,
                                            ligand_save_name=ligand_save_name)

    def simplified_cluster_extraction_mdtraj_custom(self):

        for key in self.full_data_mdtraj_analysis:
            centroid_data = self.full_data_mdtraj_analysis[key]
            for centoid_data_index in centroid_data:
                md_traj_cluster = centroid_data[centoid_data_index]['mdtraj_cluster']
                ligand_save_name = centroid_data[centoid_data_index]['ligand_save_name']
                ligand_index = centroid_data[centoid_data_index]['ligand_index']

                cluster_color = centroid_data[centoid_data_index]['color']

                print('Ligand Save name is {0}'.format(ligand_save_name))
                print('RGB values for cluster is ', cluster_color)

                actual_name, closest_name = plot_tools.get_cluster_color_name(cluster_color)

                print('<------------------------------->\n')

                coordinates = md_traj_cluster.xyz

                topology = md_traj_cluster.topology

                traj = md_traj_cluster[:]

                atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'C']
                new_traj = traj.atom_slice(atom_indices=atom_indices)

                simplified_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_simplified.pdb'

                centroid_data[centoid_data_index].update({'simplified_file': simplified_file})

                new_traj.save(simplified_file)

        test = 1

    def simplified_cluster_extraction_mdtraj(self):

        for centoid_data_index in self.centroid_data:
            md_traj_cluster = self.centroid_data[centoid_data_index]['mdtraj_cluster']
            ligand_save_name = self.centroid_data[centoid_data_index]['ligand_save_name']
            ligand_index = self.centroid_data[centoid_data_index]['ligand_index']

            cluster_color = self.centroid_data[centoid_data_index]['color']

            print('RGB values for cluster is ', cluster_color)

            actual_name, closest_name = plot_tools.get_cluster_color_name(cluster_color)

            coordinates = md_traj_cluster.xyz

            topology = md_traj_cluster.topology

            traj = md_traj_cluster[:]

            atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'C']
            new_traj = traj.atom_slice(atom_indices=atom_indices)

            simplified_file = '.' + os.sep + self.folder_path + os.sep + ligand_save_name + '_simplified.pdb'

            new_traj.save(simplified_file)

            #
            # distances = np.empty((traj.n_frames, traj.n_frames))
            # for i in range(traj.n_frames):
            #     distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

    def add_correct_topology(self, file_path):
        self.correct_topology = file_path

    def parallel_run(self, working_folder, centroid_data, centoid_data_index, correct_topology, parallel, num_threads):

        topology_to_fix = centroid_data[centoid_data_index]['centroid_file']

        topology_reconstruction_object = topology_reconstruction.TopologyReconstructionObject(
            working_folder=working_folder,
            topology_to_fix=topology_to_fix,
            correct_topology=correct_topology,
            parallel=parallel,
            num_threads=num_threads)

        # Write correct pdb file
        print('Writing correct topology ')
        topology_reconstruction_object.write_correct_topology()

        saved_topology_file = topology_reconstruction_object.get_correct_topology_filename()

        print('------->>>>>\n')

        # to_return = {topology_to_fix: {'reconstructionObject':topology_reconstruction,
        #                                'savedTopologyFile':saved_topology_file,
        #                                'topologyToFixName':topology_to_fix}}

        return [centoid_data_index, centroid_data, topology_to_fix, saved_topology_file, topology_reconstruction_object]
        # return to_return

    def generate_pymol_viz_example(self):
        import pymol

        pymol.finish_launching()

        receptor_file = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology/dock_100ns.pdb'
        pymol.cmd.load(receptor_file, 'receptorFile')
        pymol.cmd.publication('receptorFile')

        ligand_file = '/media/Work/MEGA/Programming/LasR_QRC/correct_topology/ligBindTraj_type_region_1_centroid_nx_fixed.pdb'
        lig = 'lig'
        pymol.cmd.load(ligand_file, lig)
        pymol.cmd.publication(lig)

        region_file = '/media/Work/MEGA/Programming/LasR_QRC/LasR_Quercetin_cluster_traj/ligBindTraj_type_region_1_simplified.pdb'
        reg = 'reg'
        pymol.cmd.load(region_file, reg)
        pymol.cmd.show_as(representation='dots', selection=reg)

        # in the future
        # pymol.cmd.cealign()


        # This works
        pymol.cmd.save('verify yolo.pse')
        #
        # pymol.cmd.quit()

    def add_receptor_for_viz(self, receptor_file):
        self.receptor_file_viz = receptor_file

    def generate_pymol_viz(self, key='centroid'):
        import pymol
        from time import sleep

        pymol.finish_launching()

        pymol.cmd.reinitialize()
        # Set background color to white
        pymol.cmd.bg_color("white")

        receptor_file = self.receptor_file_viz
        pymol.cmd.load(receptor_file, 'receptorFile')
        pymol.cmd.publication('receptorFile')

        centroid_data = self.full_data_mdtraj_analysis[key]

        self.pymol_objects = {}

        for centroid_data_index in centroid_data:
            curr_index = centroid_data_index
            correct_file_pymol_name = key + '_{0}'.format(curr_index)

            correct_topol_filepath = centroid_data[curr_index]['correct_topol_filepath']
            simplified_object = centroid_data[curr_index]['simplified_file']
            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            curr_color = 'cluster_color_{0}'.format(centroid_data_index)
            pymol.cmd.set_color(curr_color, centroid_data[curr_index]['colorOriginal'])
            pymol.cmd.color(curr_color, correct_file_pymol_name)

            correct_file_pymol_simple_name = key + '_simple_{0}'.format(curr_index)
            pymol.cmd.load(simplified_object, correct_file_pymol_simple_name)
            pymol.cmd.show_as(representation='dots', selection=correct_file_pymol_simple_name)

            pymol.cmd.color(curr_color, correct_file_pymol_simple_name)

            self.pymol_objects.update({centroid_data_index: {'topol': correct_file_pymol_name,
                                                             'simple': correct_file_pymol_simple_name}})
            sleep(0.5)

        test = 1

        # in the future
        # pymol.cmd.cealign()
        # This works
        print('Finished Pymol for {0}  ---- > \n'.format(key))
        save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + key + '_pymolViz.pse'
        pymol.cmd.save(save_state_name)
        sleep(0.5)


        #
        # pymol.cmd.quit()

    def generate_pymol_viz_for_thread(self, key, receptor, full_data_mdtraj_analysis, save_name):
        import pymol
        from time import sleep

        pymol.finish_launching()

        pymol.cmd.reinitialize()

        # Set background color to white
        pymol.cmd.bg_color("white")

        receptor_file = receptor
        pymol.cmd.load(receptor_file, 'receptorFile')
        pymol.cmd.publication('receptorFile')

        centroid_data = full_data_mdtraj_analysis[key]

        pymol_objects = {}

        for centroid_data_index in centroid_data:
            curr_index = centroid_data_index
            correct_file_pymol_name = key + '_{0}'.format(curr_index)

            correct_topol_filepath = centroid_data[curr_index]['correct_topol_filepath']
            simplified_object = centroid_data[curr_index]['simplified_file']
            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            curr_color = '{0}_cluster_color_{1}'.format(key, centroid_data_index)
            pymol.cmd.set_color(curr_color, centroid_data[curr_index]['colorOriginal'])
            pymol.cmd.color(curr_color, correct_file_pymol_name)

            correct_file_pymol_simple_name = key + '_simple_{0}'.format(curr_index)
            pymol.cmd.load(simplified_object, correct_file_pymol_simple_name)
            pymol.cmd.show_as(representation='dots', selection=correct_file_pymol_simple_name)

            pymol.cmd.color(curr_color, correct_file_pymol_simple_name)

            pymol_objects.update({centroid_data_index: {'topol': correct_file_pymol_name,
                                                        'simple': correct_file_pymol_simple_name}})
            sleep(0.5)

        test = 1

        # in the future
        # pymol.cmd.cealign()
        # This works
        print('Finished Pymol for {0}  ---- > \n'.format(key))
        save_state_name = save_name
        pymol.cmd.save(save_state_name)

        # pymol.cmd.quit()

        #sleep(2)

        # return pymol_objects

    def generate_pymol_viz_thread(self, key='centroid'):
        '''This is to make sure that pymol methods run separately'''
        import threading, time
        print('Start of Pymol method --->  {0} \n'.format(key))
        save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + key + '_pymolViz.pse'

        self.generate_pymol_viz_for_thread(key, self.receptor_file_viz, self.full_data_mdtraj_analysis, save_state_name)

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

        print('Finished Pymol method --->  {0} \n'.format(key))

    def generate_hybrid_pymol_viz(self, key='centroid'):
        import pymol
        from time import sleep

        pymol.finish_launching()
        pymol.cmd.reinitialize()

        # Set background color to white
        pymol.cmd.bg_color("white")

        receptor_file = self.receptor_file_viz
        pymol.cmd.load(receptor_file, 'receptorFile')
        pymol.cmd.publication('receptorFile')

        centroid_data = self.full_data_mdtraj_analysis['reshape']

        self.pymol_objects = {}

        for centroid_data_index in centroid_data:
            curr_index = centroid_data_index
            correct_file_pymol_name = 'reshape'+ '_{0}'.format(curr_index)

            correct_topol_filepath = centroid_data[curr_index]['correct_topol_filepath']

            pymol.cmd.load(correct_topol_filepath, correct_file_pymol_name)
            pymol.cmd.publication(correct_file_pymol_name)

            curr_color = 'reshape_cluster_color_{0}'.format(centroid_data_index)
            pymol.cmd.set_color(curr_color, centroid_data[curr_index]['colorOriginal'])
            pymol.cmd.color(curr_color, correct_file_pymol_name)

            self.pymol_objects.update({centroid_data_index: {'topol': correct_file_pymol_name}})

        centroid_data = self.full_data_mdtraj_analysis['centroid']

        for centroid_data_index in centroid_data:
            curr_index = centroid_data_index
            simplified_object = centroid_data[curr_index]['simplified_file']
            correct_file_pymol_simple_name = 'centroid' + '_simple_{0}'.format(curr_index)
            pymol.cmd.load(simplified_object, correct_file_pymol_simple_name)
            pymol.cmd.show_as(representation='dots', selection=correct_file_pymol_simple_name)

            curr_color = 'centroid_cluster_color_{0}'.format(centroid_data_index)
            pymol.cmd.set_color(curr_color, centroid_data[curr_index]['colorOriginal'])
            pymol.cmd.color(curr_color, correct_file_pymol_simple_name)

            self.pymol_objects.update({centroid_data_index: {'simple': correct_file_pymol_simple_name}})

            sleep(0.5)

        test = 1

        # in the future
        # pymol.cmd.cealign()
        # This works
        save_state_name = self.receptor_name + '_' + self.molecule_name + '_' + 'hybrid' + '_pymolViz.pse'
        pymol.cmd.save(save_state_name)


        sleep(0.5)
        pymol.cmd.quit()
        #


    def reconstruct_centroids_topology_custom(self, working_folder_path,
                                              correct_topology_info,
                                              parallel=False,
                                              real_parallelism=True,
                                              num_of_threads=7):

        self.add_correct_topology(correct_topology_info)

        for key in self.full_data_mdtraj_analysis:
            centroid_data = self.full_data_mdtraj_analysis[key]
            if real_parallelism is True:
                pool = multiprocessing.Pool(num_of_threads)

                # range_n_clusters = list(range(1, 11))

                function_arguments_to_call = [[working_folder_path, centroid_data, centoid_data_index,
                                               self.correct_topology, False, 7] for
                                              centoid_data_index in centroid_data]

                test = 1

                # results = pool.starmap(parallel_md_silhouette_analysis_pca,self.range_n_clusters,self.reduced_cartesian,
                #                                                 self.pca_traj.time, algorithm )
                results = pool.starmap(self.parallel_run, function_arguments_to_call)

                test = 1
                # d = [x for x in results]
                # d = [list(x.keys())[0] for x in results]

                # d = {key: value for (key, value) in iterable}
                # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
                indexes = [x[0] for x in results]

                for i in indexes:
                    centroid_data[i].update({'correct_topol_object': results[i][4]})
                    centroid_data[i].update({'correct_topol_filepath': results[i][3]})

                # self.clusters_info = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}

                # This will free memory
                pool.close()


            elif real_parallelism is False:
                for centoid_data_index in centroid_data:
                    centroid_file_path = centroid_data[centoid_data_index]['centroid_file']
                    topology_reconstruction_object = topology_reconstruction.TopologyReconstructionObject(
                        working_folder=working_folder_path,
                        topology_to_fix=centroid_file_path,
                        correct_topology=self.correct_topology,
                        parallel=parallel)

                    # Write correct pdb file
                    print('Writing correct topology ')
                    topology_reconstruction_object.write_correct_topology()

                    saved_topology_file = topology_reconstruction_object.get_correct_topology_filename()

                    centroid_data[centoid_data_index].update({'correct_topol_object': topology_reconstruction_object})
                    centroid_data[centoid_data_index].update({'correct_topol_filepath': saved_topology_file})

                    test = 1

    def reconstruct_centroids_topology(self, working_folder_path,
                                       correct_topology_info,
                                       parallel=False):

        self.add_correct_topology(correct_topology_info)
        for centoid_data_index in self.centroid_data:
            centroid_file_path = self.centroid_data[centoid_data_index]['centroid_file']
            topology_reconstruction_object = topology_reconstruction.TopologyReconstructionObject(
                working_folder=working_folder_path,
                topology_to_fix=centroid_file_path,
                correct_topology=self.correct_topology,
                parallel=parallel)

            # Write correct pdb file
            print('Writing correct topology ')
            topology_reconstruction_object.write_correct_topology()

            saved_topology_file = topology_reconstruction_object.get_correct_topology_filename()

            self.centroid_data[centoid_data_index].update({'correct_topol_object': topology_reconstruction_object})
            self.centroid_data[centoid_data_index].update({'correct_topol_filepath': saved_topology_file})

            test = 1










            # def find_centroid(ligand_conf, ligand_energy):
            #     dtype = [('index', np.int64), ('bindEn', np.float64)]
            #     energy_Data = np.empty((0,), dtype=dtype)  # This is for clusters
            #     csvReaderData = csv.reader(open(ligand_energy, "r"), delimiter=',')
            #     for row in csvReaderData:
            #         # print('row is ',row)
            #         temp_data = np.array([(row[0], row[3])], dtype=dtype)
            #         energy_Data = np.append(energy_Data, temp_data, axis=0)  # Here lies a problem
            #         # print(','.join(row))
            #
            #     print('energy data', energy_Data[0])
            #
            ### ---> Conformation Analysis
            # traj = md.load(ligand_conf)
            # real_originalData = traj[:]

# traj[0].save('uberTest.pdb')
#
#     print('traj len is ', len(traj))
#
#     print('traj frames ,', traj.n_frames)
#
#     atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
#     distances = np.empty((traj.n_frames, traj.n_frames))
#     for i in range(traj.n_frames):
#         distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)
#
#     print('len(distances ', len(distances))
#     print('Max pairwise rmsd: %f nm' % np.max(distances))
#     print('Min pairwise rmsd: %f nm' % np.min(distances))
#
#     beta = 1
#     index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
#     print(index)
#
#     centroid = traj[index]
#     print(centroid)
#
#     toWrite = real_originalData[index]
#     print('toWrite checking is ', toWrite)
#     toWrite.save('BestOFTheBest.pdb')
