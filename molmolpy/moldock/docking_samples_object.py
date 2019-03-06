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

import os
import sys

import numpy as np
import scipy as sp
import pandas as pd
from molmolpy.parser import molecule_object
from molmolpy.utils import folder_utils
from molmolpy.utils import helper as hlp


class DockSamplesObject(object):
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

    def __init__(self, folder_path, load_way='molmolpy',
                 molname='Unknown',
                 receptor_name='Unknown',
                 info_type='docking',
                 color='b',
                 z_order=10,
                 marker='o',
                 size=10):

        print('Dock samples object has been created')
        self.folder_path = folder_path

        print(folder_path)

        self.molecule_name = molname
        self.receptor_name = receptor_name
        self.color = color
        self.size = size
        self.marker = marker
        self.z_order = z_order

        self.info_type = info_type

        self.simulation_name = 'docking_' + self.receptor_name + '_' + self.molecule_name

        if self.info_type == 'docking':

            # original data before transformation
            self.directories = self.find_sample_folders(self.folder_path)
            print(self.directories)

            self.sample_files = self.obtain_samples()
            print(self.sample_files)

            # Obtained  samples
            self.samples_data = self.load_samples()

            # VIP
            self.equivalent_models__ = {}
            self.analysis_structure__ = self.transform_for_analysis()

            # TODO this is a new structure for data analysis
            # self.analysis_reshape_structure__ = self.transform_by_reshape()

            self.analysis_reshape_structure__, self.analysis_centroid_structure__ = self.transform_by_ultra()

            self.concatenated_analysis__ = self.concatenate_analysis(self.analysis_reshape_structure__,
                                                                     self.analysis_centroid_structure__)

            test = 1

        elif self.info_type == 'docking_new':
            self.directories = [self.folder_path]
            self.sample_files = self.obtain_samples()
            self.samples_data = self.load_samples()

            self.equivalent_models__ = {}
            self.analysis_structure__ = self.transform_for_analysis()

            # TODO this is a new structure for data analysis for exhaustiveness
            # self.analysis_reshape_structure__ = self.transform_by_reshape()
            self.analysis_reshape_structure__, self.analysis_centroid_structure__ = self.transform_by_ultra()
            
            self.concatenated_analysis__ = self.concatenate_analysis(self.analysis_reshape_structure__,
                                                                     self.analysis_centroid_structure__)

            
            test = 1
        elif self.info_type == 'exhaust':
            self.directories = [self.folder_path]
            self.sample_files = self.obtain_samples()
            self.samples_data = self.load_samples()

            self.equivalent_models__ = {}
            self.analysis_structure__ = self.transform_for_analysis()

            # TODO this is a new structure for data analysis for exhaustiveness
            # self.analysis_reshape_structure__ = self.transform_by_reshape()

            self.analysis_reshape_structure__, self.analysis_centroid_structure__ = self.transform_by_ultra()
            
            self.concatenated_analysis__ = self.concatenate_analysis(self.analysis_reshape_structure__,
                                                                     self.analysis_centroid_structure__)


            test = 1

        test = 1

    def concatenate_analysis(self, X, Y):
        concat = pd.concat([X, Y], axis=0)

        return concat


    def get_molecule_name(self):
        return self.molecule_name

    def get_receptor_name(self):
        return self.receptor_name

    def set_molecule_name(self, mol_name):
        self.molecule_name = mol_name

    def set_receptor_name(self, receptor_name):
        self.receptor_name = receptor_name

    @hlp.timeit
    def transform_for_analysis(self):
        model = 1

        columns_dock_center = ['SampleInfoNum', 'ModelNum', 'X', 'Y', 'Z', 'BindingEnergy', 'MolName']

        dock_df = pd.DataFrame(columns=columns_dock_center)

        for i in sorted(self.samples_data.keys()):
            models = self.samples_data[i]
            # print(model)
            for y in models.mol_data__:
                # This should be the structure for equivalency of models
                # print(model, i, y)
                self.equivalent_models__.update({model: {'molName': self.molecule_name, 'file': i, 'modelNum': y,
                                                         'molDetail': models.mol_data__[y]}})

                curr_model = models.mol_data__[y]
                curr_frame = curr_model['dataframe']
                curr_x = curr_frame['X'].mean()
                curr_y = curr_frame['Y'].mean()
                curr_z = curr_frame['Z'].mean()
                curr_bind = curr_model['vina_info'][0]
                curr_mol_name = self.molecule_name

                sample_info_num = int(models.sample_info_num)
                dock_df.loc[model] = [sample_info_num, int(model), curr_x, curr_y, curr_z, curr_bind, curr_mol_name]
                # print(y, models.mol_data__[y]['dataframe'])
                model += 1
        # print(self.equivalent_models)

        dock_df['ModelNum'] = dock_df['ModelNum'].astype(int)
        # dock_df['ModelNum'] = dock_df['ModelNum'].astype('category')
        return dock_df

    @hlp.timeit
    def transform_by_ultra(self):
        model = 1

        columns_dock_center = ['SampleInfoNum', 'ModelNum', 'X', 'Y', 'Z', 'BindingEnergy', 'MolName']

        dock_df_reshape = pd.DataFrame()  # (columns=columns_dock_center)

        dock_df_centroid = pd.DataFrame()  # (columns=columns_dock_center)

        for i in sorted(self.samples_data.keys()):
            models = self.samples_data[i]
            # print(model)
            for y in models.mol_data__:
                # This should be the structure for equivalency of models
                # print(model, i, y)
                # self.equivalent_models__.update({model: {'molName': self.molecule_name, 'file': i, 'modelNum': y,
                #                                          'molDetail': models.mol_data__[y]}})

                curr_model = models.mol_data__[y]
                curr_frame = curr_model['dataframe']

                curr_model_data = curr_frame[['X', 'Y', 'Z']]

                a = curr_model_data.values
                b = np.reshape(a, -1)  # Convert to 1D row

                reshaped_frame = pd.DataFrame(b)

                curr_x = curr_frame['X'].mean()
                curr_y = curr_frame['Y'].mean()
                curr_z = curr_frame['Z'].mean()

                curr_bind = curr_model['vina_info'][0]
                curr_mol_name = self.molecule_name

                # very important step
                if model == 1:
                    self.data_cols = [x for x in range(1, len(b) + 1)]
                    self.cols = ['SampleInfoNum', 'ModelNum'] + [x for x in range(1, len(b) + 1)] + ['BindingEnergy',
                                                                                                     'MolName', 'Type']
                    dock_df_reshape = pd.DataFrame(columns=self.cols)
                    dock_df_centroid = pd.DataFrame(columns=self.cols)
                    print('shape is ', dock_df_reshape.shape)

                    # start = 1
                    # end = len(reshaped_frame)+1

                type_reshape = 'reshape'
                type_centroid = 'centroid'

                sample_info_num = int(models.sample_info_num)
                dock_df_reshape.loc[model] = [sample_info_num, int(model)] + b.tolist() + [curr_bind, curr_mol_name,
                                                                                           type_reshape]

                data_part1 = [curr_x, curr_y, curr_z]

                # fill zeros
                # TODO this is not effective
                data_part2 = [0 for x in range(4, len(b) + 1)]

                # final_data = data_part1 + data_part2
                times = int(len(b)/3)

                # TODO try 2
                final_data = data_part1 * times

                dock_df_centroid.loc[model] = [sample_info_num, int(model)] + final_data + [curr_bind, curr_mol_name,
                                                                                            type_centroid]
                # dock_df.loc[model]['ModelNum'] = int(model)
                # dock_df.loc[model]['BindingEnergy'] = curr_bind
                # dock_df.loc[model]['MolName'] = curr_mol_name
                # dock_df.loc[model][start:end] = b

                # print(y, models.mol_data__[y]['dataframe'])
                model += 1
        # print(self.equivalent_models)

        # dock_df['ModelNum'] = dock_df['ModelNum'].astype(int)


        # dock_df['ModelNum'] = dock_df['ModelNum'].astype('category')
        return dock_df_reshape, dock_df_centroid

    @hlp.timeit
    def transform_by_reshape(self):
        model = 1

        columns_dock_center = ['SampleInfoNum', 'ModelNum', 'X', 'Y', 'Z', 'BindingEnergy', 'MolName']

        dock_df = pd.DataFrame()  # (columns=columns_dock_center)

        for i in sorted(self.samples_data.keys()):
            models = self.samples_data[i]
            # print(model)
            for y in models.mol_data__:
                # This should be the structure for equivalency of models
                # print(model, i, y)
                # self.equivalent_models__.update({model: {'molName': self.molecule_name, 'file': i, 'modelNum': y,
                #                                          'molDetail': models.mol_data__[y]}})

                curr_model = models.mol_data__[y]
                curr_frame = curr_model['dataframe']

                curr_model_data = curr_frame[['X', 'Y', 'Z']]

                a = curr_model_data.values
                b = np.reshape(a, -1)  # Convert to 1D row

                reshaped_frame = pd.DataFrame(b)

                curr_bind = curr_model['vina_info'][0]
                curr_mol_name = self.molecule_name

                # very important step
                if model == 1:
                    self.data_cols = [x for x in range(1, len(b) + 1)]
                    self.cols = ['SampleInfoNum', 'ModelNum'] + [x for x in range(1, len(b) + 1)] + ['BindingEnergy',
                                                                                                     'MolName']
                    dock_df = pd.DataFrame(columns=self.cols)
                    print('shape is ', dock_df.shape)

                    # start = 1
                    # end = len(reshaped_frame)+1

                sample_info_num = int(models.sample_info_num)
                dock_df.loc[model] = [sample_info_num, int(model)] + b.tolist() + [curr_bind, curr_mol_name]
                # dock_df.loc[model]['ModelNum'] = int(model)
                # dock_df.loc[model]['BindingEnergy'] = curr_bind
                # dock_df.loc[model]['MolName'] = curr_mol_name
                # dock_df.loc[model][start:end] = b

                # print(y, models.mol_data__[y]['dataframe'])
                model += 1
        # print(self.equivalent_models)

        # dock_df['ModelNum'] = dock_df['ModelNum'].astype(int)


        # dock_df['ModelNum'] = dock_df['ModelNum'].astype('category')
        return dock_df

    @hlp.timeit
    def load_samples(self):
        sample_data = {}
        for sample in self.sample_files:
            load_mol = molecule_object.MoleculeObject(sample, mol_name=self.molecule_name)
            sample_data.update({sample: load_mol})
        return sample_data

    @hlp.timeit
    # TODO there is bug here
    def obtain_samples(self):
        sample_files = []
        for folder in self.directories:
            samples = self.find_sample_files(folder)
            # print(samples)
            if 'exhaust' in self.info_type:
                sample_files = [folder + os.sep + sample for sample in samples]
            elif 'docking_new' in self.info_type:
                sample_files = [folder + os.sep + sample for sample in samples]
            else:
                sample_files.append(folder + os.sep + samples[0])
        return sample_files

    @hlp.timeit
    # This might need to get modified
    def find_sample_files(self, folder):
        try:
            VIP = []
            for dirname, dirnames, filenames in os.walk(folder):
                for i in filenames:
                    # print i
                    if 'out' in i:
                        VIP.append(i)
                        # This is not necessary since info is inside pdbqt file
                        # elif 'vina_sample_' in i:
                        #     VIP.append(i)
            return VIP
        except Exception as e:
            print("error in find_files: ", e)
            sys.exit(0)

    @hlp.timeit
    def find_sample_folders(self, folder_path='.', dir_name='vina_sample'):
        try:
            dir_names = []
            for dirname, dirnames, filenames in os.walk(folder_path):
                # print(dirname, '-')
                if dir_name in dirname:  #
                    dir_names.append(dirname)
            # print sorted(dir_names)
            return sorted(dir_names)
        except Exception as e:
            print("Problem with finding folders : ", e)
            sys.exit(0)

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
