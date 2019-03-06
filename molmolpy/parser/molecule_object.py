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

import os
import pandas as pd


class MoleculeObject(object):
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

    def __init__(self, file_path, load_way='molmolpy',
                 mol_name='Unknown'):

        # print('Molecule object has been created')
        self.file_path = file_path

        self.molecule_name = mol_name
        # original data before transformation
        self.info, self.object = self.PDBQTparse(self.file_path)
        # print(self.info)
        # print(self.object)
        self.get_info_from_filename()

        self.mol_data__ = self.transform_data()

    def get_info_from_filename(self):
        temp_data = self.file_path.split(os.sep)
        self.file_name = temp_data[-1]
        new_temp_data =self.file_name.split('_')
        self.sample_info_type = new_temp_data[1]
        self.sample_info_num = new_temp_data[-2]

    # For printing class objects
    def __repr__(self):
        text = 'This is single molecule object with \n'
        return text + str(len(self.info)) + ' Models\n '

    def get_molecule_name(self):
        return self.molecule_name

    def set_molecule_name(self, mol_name):
        self.molecule_name = mol_name

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

    # This part is for helper functions
    def cleanVal(self, val):
        tempus = val.split(' ')
        # print tempus
        for i in tempus:
            if i == '':
                pass
            else:
                return i

    def extractInfo(self, line):
        lineSplit = line.split(' ')
        # print('lineSplit ',lineSplit)
        cleanedVal = []
        for i in lineSplit:
            if i == '':
                pass
            else:
                cleanedVal.append(i)
        return cleanedVal

    # For parsing Autodock pdbqt output file
    def PDBQTparse(self, filename):
        inFile = open(filename, 'r')
        try:
            final_list = []
            final_info = []  # Model num energy info
            count = 1
            info = []
            model_data = []
            for line in inFile:
                # print('Line is ',line)
                temp = []
                if 'MODEL' in line or 'REMARK VINA' in line:
                    tempInfo = self.extractInfo(line)
                    # print('tempInfo ',tempInfo)
                    if 'MODEL' in tempInfo:
                        modelNum = int(tempInfo[-1])
                        # print('ModelNum is ', modelNum)
                        if modelNum > count:  # Becareful
                            count = modelNum
                            final_info.append(info)
                            final_list.append(model_data)
                            info = []
                            model_data = []
                        info.append(modelNum)
                    else:
                        info.append(float(tempInfo[-3]))
                        info.append(float(tempInfo[-2]))
                        info.append(float(tempInfo[-1]))
                if self.cleanVal(line[0:7]) == 'ATOM' or self.cleanVal(line[0:7]) == 'HETATM':
                    temp.append(self.cleanVal(line[0:7]))  # Record name [0]
                    temp.append(self.cleanVal(line[7:12]))  # Atom serial number [1]
                    temp.append(self.cleanVal(line[12:17]))  # Atom name [2]
                    temp.append(self.cleanVal(line[17:21]))  # Residue name [3]
                    temp.append(self.cleanVal(line[21:23]))  # Chain identifier [4]
                    temp.append(self.cleanVal(line[23:27]))  # Residue sequence number [5]
                    temp.append(self.cleanVal(line[31:39]))  # Orthogonal coordinates for X in Angstroms [6]
                    temp.append(self.cleanVal(line[39:47]))  # Orthogonal coordinates for Y in Angstroms [7]
                    temp.append(self.cleanVal(line[47:55]))  # Orthogonal coordinates for Z in Angstroms [8]
                    temp.append(self.cleanVal(line[55:61]))  # Occupancy [9]
                    temp.append(self.cleanVal(line[61:67]))  # Temperature factor [10]
                    temp.append(
                        self.cleanVal(line[67:76]))  # Real(10.4)    partialChrg  Gasteiger PEOE partial charge *q*.
                    temp.append(self.cleanVal(line[77:79]))  # Element symbol, right-justified. [11]
                    temp.append(self.cleanVal(line[79:81]))  # Element symbol, right-justified. [12]
                    # print('temp is ',temp)
                    if temp[-1] == '\n':
                        model_data.append(temp[:-1])
                    else:
                        model_data.append(temp)
            # -->>>>> VERY IMPORTANT HERE
            final_info.append(info)
            final_list.append(model_data)
            return final_info, final_list
        
        except Exception as e:
            print('Oh come on  PDB parse problem: ', e)
        finally:
            inFile.close()
