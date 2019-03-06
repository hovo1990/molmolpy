#!/usr/bin/env python
#
# @file    cluster_traj_analyzer.py
# @brief   centroid analyzer from molecular docking results
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
import pickle

import csv

import mdtraj as md  # Problem with it
import numpy as np

import math
import os
import random
import re
from operator import eq

import itertools

import networkx as nx
import numpy as np
# import pdbRealParseWrite  # My module for parsing and writing verify yolo
from networkx.algorithms import isomorphism
from pybel import *  # OpenBabel python 2
from sklearn.neighbors import NearestNeighbors

from molmolpy.utils import folder_utils
from molmolpy.utils import pickle_tools
from molmolpy.utils import converters
from molmolpy.utils import pdb_tools

from molmolpy.utils import helper as hlp

import multiprocessing
from joblib import Parallel, delayed

# Picometers
atomicBondLength = {
    'H-H': 74,
    'C-C': 154,
    'N-N': 145,
    'O-O': 148,
    'F-F': 142,
    'Cl-Cl': 199,
    'Br-Br': 228,
    'I-I': 268,
    'C-N': 147,
    'N-C': 147,
    'C-O': 143,
    'O-C': 143,
    'C-S': 182,
    'C-F': 135,
    'C-Cl': 177,
    'C-Br': 194,
    'C-I': 214,
    'C-H': 109,
    'N-H': 101,
    'H-N': 101,
    'O-H': 96,
    'H-O': 96,
    'F-H': 92,
    'Cl-H': 127,
    'Br-H': 141,
    'I-H': 161,
    'C=C': 134,
    'C=-C': 120,
    'N=-N': 110
}

inv_bondLength = {v: k for k, v in atomicBondLength.items()}

print('badoom', inv_bondLength)


# #For checking labels
# ATOMS = ['H','C','O','N','S']



class TopologyReconstructionObject(object):
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

    def __init__(self, working_folder=None, topology_to_fix=None, correct_topology=None, parallel=False, num_threads=7):

        # self.docking_analysis_object = docking_analysis_object
        #
        # self.cluster_colors_pre_rgb = self.docking_analysis_object.colors_
        # self.cluster_colors_rgb = converters.convert_seaborn_color_to_rgb(self.cluster_colors_pre_rgb)
        # self.cluster_colors = self.docking_analysis_object.cluster_colors
        #
        # self.analysis_structure = docking_analysis_object.analysis_structure
        #
        # self.molecule_name = self.docking_analysis_object.molecule_name
        # self.receptor_name = self.docking_analysis_object.receptor_name
        #
        # # folder_info = folder_utils.find_folders(folder_path)
        # # self.folder_path = self.molecule_name + '_' +  folder_path
        #
        # self.folder_path = docking_analysis_object.save_cluster_models_dir
        #
        # self.curr_index = 0

        self.parallel_compute = parallel
        self.num_threads = 7

        self.topology_to_fix = topology_to_fix
        self.correct_topology = correct_topology

        self.working_folder = working_folder

        self.fix_topology(self.topology_to_fix, self.correct_topology, self.working_folder)

    def fix_topology(self, ligand_centroid, original_ligand, working_folder):
        filename = ligand_centroid

        outputName_part1 = ligand_centroid.split(os.sep)
        outputName_part2 = outputName_part1[-1].split('.')
        outputName_part3 = outputName_part2[0]

        # This part works
        output_name = self.working_folder + os.sep + outputName_part3 + '_pybel_prep.pdb'

        self.corr_topology_output_name = self.working_folder + os.sep + outputName_part3 + '_nx_fixed.pdb'

        self.output_name = output_name

        # self.saveBabel(filename, output_name)
        self.addHydrogen(filename, output_name)

        filename = ligand_centroid
        self.vinaPreFix_ligand, self.vina_ligand_dict, self.vina_ligand_neighbours = self.molStuff(filename)

        # print('->->-'*20)

        self.vina_fixed_ligand, self.vina_fixed_ligand_dict, self.vina_fixed_ligand_neighbours = self.molStuff(
            output_name)

        print('->->-' * 20)
        # print(vinaFixedHSL)
        # (vinaFixedHslNeighbours)


        # # print('---------------------------\n\n')
        # #
        # filename = original_ligand


        HSL, hslDict, hslNeighbours = self.molStuff(original_ligand)
        # print(HSL)
        # print(hslNeighbours,vinaFixedHslNeighbours)
        self.vinaFixGraph = self.createGraph(self.vinaPreFix_ligand, self.vina_fixed_ligand,
                                             self.vina_fixed_ligand_neighbours)

        self.original_graph = self.createGraph(self.vinaPreFix_ligand, HSL, hslNeighbours)

        # TODO is it possible to parallelize
        self.compareGraphs(self.vinaFixGraph, self.original_graph, self.vina_fixed_ligand, HSL)
        # TODO need a faster implementation verify yolo

        # self.compare_graphs_parallel(self.vinaFixGraph, self.original_graph, self.vina_fixed_ligand, HSL)



        # self.extract_centroids_auto()

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

    def convertToNumpy(self, molecule):
        final_Data = np.empty((0, 3), float)  # verify yoloLSDOSDS
        for atom in molecule:
            # print(atom)
            temp_data = []
            # print('model is ',model)
            X = float(atom[6])
            Y = float(atom[7])
            Z = float(atom[8])
            temp_data.append(X)
            temp_data.append(Y)
            temp_data.append(Z)

            # print(temp_data)
            temp_data = np.array([temp_data])
            # print(temp_data, temp_data.shape, temp_data.dtype)

            # final_Data.append(temp_data)
            final_Data = np.append(final_Data, temp_data, axis=0)
        # print('verify yolo ',final_Data)
        return final_Data

    def molStuff(self, filename):
        temp_info, real_PDB_file = self.PDBQTparse(filename)
        # print(temp_info,  real_PDB_file )
        print('----' * 20)

        real_data = real_PDB_file[0]

        real_dict = {}
        for i in real_data:
            real_dict.update({i[2]: i})

        molNumpy = self.convertToNumpy(real_data)
        # print(molNumpy)

        # machineAnalysis(molNumpy)

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(molNumpy)  # Allright
        # nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(molNumpy) #Allright
        distances, neighbourIndices = nbrs.kneighbors(molNumpy)
        # print('indices ',indices)


        # # print('distances ',distances)

        # for indice in neighbourIndices:
        #     for i in indice:
        #         print('    ', real_data[i])
        #     print('----'*20)
        return real_data, real_dict, neighbourIndices

    def printNeighbours(self, indices, Mol):
        print("Printing Neighbours")
        for i in indices:
            print('    ', Mol[i])
        print('----' * 20)

    def saveBabel(self, filename, outputName="HSL_conf.pdb", ):
        # mol = readfile("pdb", filename).next()

        # Updated in new version
        mol = readfile("pdb", filename).__next__()
        print('mol is ', mol)

        # mol.addh()
        # file_to_save = working_folder + os.sep + outputName
        file_to_save = outputName

        output = Outputfile("pdb", file_to_save, overwrite=True)
        output.write(mol)
        output.close()
        print("added Hydrogens :D")
        print(
            "--------------------------------------------------------------------------------------------------------")

    # TODO check Openbabel installation
    def addHydrogen(self, filename, outputName="vinaConf.pdb"):
        # mol = readfile("pdb", filename).next()


        mol = readfile("pdb", filename).__next__()
        print('mol is ', mol)
        print('mol is ', mol.data)

        mol.addh()

        # file_to_save = working_folder + os.sep + outputName
        file_to_save = outputName

        output = Outputfile("pdb", file_to_save, overwrite=True)
        output.write(mol)
        output.close()
        print("added Hydrogens :D")

    def moleculeTransformation(self, realMol, realDict, realNeighbours, vinaPreFix, dockMol, dockNeighbours):
        real_backup = realMol[:]
        print('len Prefix ', len(vinaPreFix))
        print('len fixed ', len(dockMol))
        # Here we have a serious problem
        index = 1
        # vinaPreFix = vinaPreFix[:index] #Test purposes
        for i in vinaPreFix:  # dockMol:
            # print('i is ',i)
            keyDock = i[2]
            dockX = float(i[6])
            dockY = float(i[7])
            dockZ = float(i[8])
            # print('realdict key ',realDict[keyDock])

            # realDict[keyDock][6] = dockX
            # realDict[keyDock][7] = dockY
            # realDict[keyDock][8] = dockZ

            # print('new H atom  ',new_centre)

            # print('realdict modif key ',realDict[keyDock])
            # print('----'*5)
            # print('\n')

            lookNkey = keyDock  # realDict[keyDock][2]
            # print('lookNkey is ', lookNkey)
            for indiceOriginal, indiceH in zip(realNeighbours, dockNeighbours):  # [:index] Huge bug here but whye=
                # print('neighborsOrign ', indiceOriginal)
                # print('neighboursH ',indiceH)
                # print('----------')
                tempOrigData = realMol[indiceOriginal[0]]
                tempOrigKey = tempOrigData[2]
                tempVinaData = realMol[indiceH[0]]
                tempVinaKey = tempVinaData[2]
                # print(lookNkey,tempOrigKey,tempVinaKey)
                if lookNkey == tempVinaKey and lookNkey == tempOrigKey and 'H' not in lookNkey:
                    print('tada ', lookNkey, tempOrigKey, tempVinaKey)
                    indOriginal = indiceOriginal[1:]
                    indH = indiceH[1:]
                    print(indOriginal, indH)
                    self.printNeighbours(indOriginal, realMol)
                    self.printNeighbours(indH, dockMol)
                    #     # print()
                    #     for x,y in zip(indOriginal, indH): #Zip not a  good idea
                    #         keyOriginal = realMol[x][2]
                    #         keyVina = dockMol[y][2]
                    #         if 'H' in keyOriginal and 'H' in keyVina:
                    #             print(lookNkey,keyOriginal, keyVina)
                    #             pass

        print('\n')
        # molData = []
        # for i in realMol:
        #     #print(' i is ',i)
        #     key = i[2]
        #     molData.append(realDict[key])
        #     #molData.append(i)

        # file_to_write = open("ligBTestus.pdb", 'w')
        # for row in molData:
        #     #print(row)
        #     writeRow = pdbRealParseWrite.write_rowPDB(row)
        #     #print('tada ',writeRow)
        #     file_to_write.write(writeRow)
        # #pdbParseWrite.write_Lig(molData,file_to_write)
        # file_to_write.close()

    def dot(self, v1, v2):
        return sum(x * y for x, y in zip(v1, v2))

    def length(self, v):
        return self.dot(v, v)

    def sim(self, v1, v2):
        return self.dot(v1, v2) / (self.length(v1) * self.length(v2))

    def getVectors(self, neighbours, molecule):
        keyDataIndex = neighbours[0]
        vectorDict = {keyDataIndex: []}
        keyData = molecule[keyDataIndex]
        keyVector = keyData[2]
        keyDataVector = np.array([float(keyData[6]), float(keyData[7]), float(keyData[8])])
        # print('keyDataVector ,',keyDataVector)
        # print('get vectors ',keyData)
        # print('->-'*10)
        connectionsId = neighbours[1:]
        for connectId in connectionsId:
            connIndex = connectId
            connData = molecule[connIndex]
            connDataId = connData[2]
            # print('coomDataOD ',connDataId)
            connDataVector = np.array([float(connData[6]), float(connData[7]), float(connData[8])])
            # print('connData Vector ',connDataVector)
            # print('connData ',connData)
            # print('--------------')
            currVector = connDataVector - keyDataVector
            # print('Curr vector ',currVector)
            ##print('====='*10)
            vectorDict[keyDataIndex].append([keyVector, connIndex, connDataId, currVector])
        # print(' \n\n')
        # print('vectorDict ',vectorDict)
        # print(' \n\n')
        return vectorDict

    def dot(self, vector1, vector2):
        return np.dot(vector1, vector2)

    def length(self, vector):
        return np.dot(vector, vector)

    def similarity(self, vector1, vector2):
        # print('tadoo ',length(vector1))
        return self.dot(vector1, vector2) / (self.length(vector1) * self.length(vector2))

    def extractLetter(self, object):  # Very important function
        letter = re.split('(\d+)', object)
        # print(letter[0])
        return letter[0]

    def compareVectors(self, vectorDock, vectorOriginal):  # Needs a lot of checking
        keyDock = list(vectorDock.keys())[0]
        keyOrig = list(vectorOriginal.keys())[0]
        # print('compVect keyDock ', keyDock,keyOrig)
        totalScore = 0
        fixedOrder = []
        for currVectDock in vectorDock[keyDock]:  # This needs a fixing, o boy motherchecker
            highScore = 0
            vectorId = None
            for currVecOrig in vectorOriginal[keyOrig]:
                similarityVal = self.similarity(currVectDock[-1], currVecOrig[-1])
                if similarityVal > 0:
                    print('tada currVecDock ', keyDock, currVectDock)
                    print('tada currVecOrig ', keyOrig, currVecOrig)
                    print('similarity is ', similarityVal)
                    print('-------------------------------')
                if similarityVal > highScore:
                    highScore = similarityVal
                    vectorId = currVecOrig[:]
            totalScore += highScore
            # print('tada currVec ',keyDock, currVectDock)
            # print('High score is ', highScore)
            # print('vectorId ',vectorId)
            # print('-------------------\n')
            fixedOrder.append(vectorId)
        # print('total score in compare ',totalScore)
        dockVector = vectorDock[keyDock]

        fixedDict = {keyOrig: fixedOrder}
        # print('dockVector ',dockVector)
        # print('fixed Dict in compVect ',fixedDict)

        for x, y in zip(dockVector,
                        fixedOrder):  # Scoring part look alright but there is a problem with Cosine similarity
            # print('x y',x,y)
            try:
                objectDock1 = x[0]
                currLetterDock1 = self.extractLetter(objectDock1)
                objectDock2 = x[2]
                currLetterDock2 = self.extractLetter(objectDock2)
                bondDock = currLetterDock1 + '-' + currLetterDock2
                objectOrig1 = y[0]
                currLetterOrig1 = self.extractLetter(objectOrig1)
                objectOrig2 = y[2]
                currLetterOrig2 = self.extractLetter(objectOrig2)
                bondOrig = currLetterOrig1 + '-' + currLetterOrig2
                if bondDock == bondOrig:
                    totalScore += 5
                    print('bonds motherchecka ', bondDock, bondOrig)
                    print('totalScore ', totalScore)
                else:
                    totalScore -= 20
            except:
                totalScore -= 60
        print('FIXED ORDER ', fixedOrder)
        print('FIXED DICT ', fixedDict)
        print('totalScore Final ', totalScore)
        print('------------\n')

        return totalScore, fixedDict

    def caclEclideanDistance(self, atom1Mol, atom2Mol):
        atom1X = float(atom1Mol[6])
        atom1Y = float(atom1Mol[7])
        atom1Z = float(atom1Mol[8])

        atom2X = float(atom2Mol[6])
        atom2Y = float(atom2Mol[7])
        atom2Z = float(atom2Mol[8])
        distPart1 = (atom2X - atom1X) ** 2 + (atom2Y - atom1Y) ** 2 + (atom2Z - atom1Z) ** 2
        return math.sqrt(distPart1)

    def addEdge(self, atomVIP_index, atom2_index, dockMol):
        atom1Mol = dockMol[atomVIP_index]
        atom2Mol = dockMol[atom2_index]

        distance = round(self.caclEclideanDistance(atom1Mol, atom2Mol), 3) * 100

        try:

            atom1Mol_name = self.extractLetter(atom1Mol[2])
            atom2Mol_name = self.extractLetter(atom2Mol[2])
            bond = atom1Mol_name + '-' + atom2Mol_name

            bond_dist = atomicBondLength[bond]

            diff_dist = abs(bond_dist - distance)
            print(atom1Mol, atom2Mol)
            print('calc dist ', distance)
            print('bond is ', bond)
            print('teoretical bond dist ', bond_dist)
            print('diff is ', diff_dist)

            print('-----' * 10)
            if diff_dist < 1.0:
                print('YAY ohohohoh')
                return True
            if diff_dist < 10.0:
                print('check yeah')
                return True
        except:
            return False

    # Use vina with H or not?
    def createGraph(self, VinaPreFix, dockMol, dockNeighbours):
        molGraph = nx.Graph()
        toIndex = 5
        labels = {}
        for neighbour in dockNeighbours:  # Testing part
            print('neighbour ', neighbour)
            nodeId = neighbour[0]
            atomTypeName = self.extractLetter(dockMol[nodeId][2])
            atomTopo = dockMol[nodeId][2]
            molGraph.add_node(nodeId, atomType=atomTypeName, atomTopoName=atomTopo)
            # print(molGraph.node[nodeId])
            labels[nodeId] = atomTopo
            toLookInd = neighbour[1:]
            for index in toLookInd:
                nodeId_2 = index
                atomTypeName = self.extractLetter(dockMol[nodeId_2][2])
                atomTopo = dockMol[nodeId_2][2]
                molGraph.add_node(nodeId_2, atomType=atomTypeName, atomTopoName=atomTopo)
                labels[nodeId_2] = atomTopo
                boolResult = self.addEdge(nodeId, nodeId_2, dockMol)
                if boolResult == True:
                    molGraph.add_edge(nodeId, nodeId_2)  # , {'weight':0.15}) Need to add weight
        print('len Nodes ', len(molGraph.nodes()))
        print('nodes ', molGraph.nodes())
        pos = nx.circular_layout(molGraph)
        nx.draw_networkx_labels(molGraph, pos, labels, font_size=10)
        nx.draw_circular(molGraph, node_size=15)
        # plt.show()
        return molGraph

    def getAttrubutesFromNodes(self, origAtomTypes, origAtomTopoName):
        atomTypes = []
        atomTopoNames = []
        keys = list(origAtomTypes.keys())
        print('keys are', keys)
        for i in keys:
            atomType = origAtomTypes[i]
            atomTopoName = origAtomTopoName[i]
            if 'C' in atomType and 'C' in atomTopoName:
                atomTypes.append(atomType)
                atomTopoNames.append(atomTopoName)
            if 'O' in atomType and 'O' in atomTopoName:
                atomTypes.append(atomType)
                atomTopoNames.append(atomTopoName)
        print(atomTypes)
        print(atomTopoNames)
        print('------------------->>>>___>>>>>>')
        return atomTypes, atomTopoNames

    def score(self, similar_nodes, dockMol, origMol):
        score = 0
        dockNode = list(similar_nodes.keys())
        for node in dockNode:
            temp1 = dockMol[node][2]
            temp2 = origMol[similar_nodes[node]][2]
            if temp1 == temp2:
                score += 1
                # else:
                #     score -= 10
        return score

    #
    # def calc_similarity(self, iso_num, iterator, dockMol, origMol):
    #
    #     similar_nodes = iterator.__next__()
    #
    #     currScore = self.score(similar_nodes, dockMol, origMol)
    #
    #     results = {str(iso_num):{'score':currScore,
    #                              'simNodes':similar_nodes}}
    #
    #     return  results


    def calc_similarity(self, similar_nodes):

        dockMol = self.dockMol
        origMol = self.origMol
        currScore = self.score(similar_nodes, dockMol, origMol)

        # results = [{'score':currScore,'simNodes':similar_nodes}]

        return (currScore, similar_nodes)

    @hlp.timeit
    def compareGraphs(self, dockGraph,
                      origGraph,
                      dockMol,
                      origMol,
                      todo_count=100000):

        modifMol = origMol[:]

        print('origraph nodes are ', origGraph.nodes())
        origAtomTypes = nx.get_node_attributes(origGraph, 'atomType')
        origAtomTopoName = nx.get_node_attributes(origGraph, 'atomTopoName')
        # print('baboom ',origAtomTypes)
        # print('baboomUltra ',origAtomTopoName )
        atomTypes, atomTopoNames = self.getAttrubutesFromNodes(origAtomTypes, origAtomTopoName)
        # origNodes = origGraph.nodes()
        # for node in origNodes:
        #     print(node.atomType)
        #     print(node.atomTopoName)

        nm = isomorphism.generic_node_match('atomTopoName', 'C14', eq)  # Test Not Correct way to use
        is_Iso = nx.is_isomorphic(dockGraph, origGraph)  # no weights considered
        print('is it is_isomorphic ', is_Iso)
        GM = isomorphism.GraphMatcher(dockGraph, origGraph)  # ,  node_match=nm) # Didn't work
        print('tada ', GM.is_isomorphic())
        print('GM mapping moherfcuker', GM.mapping)

        # tadus = GM.match()
        # print('tada sdasjdajskdas ',tadus)
        # for i in tadus:
        #     print(i)

        # TODO here use multiprocessing
        isoIter = GM.isomorphisms_iter()
        test = 1

        self.dockMol = dockMol
        self.origMol = origMol

        if self.parallel_compute is True:
            self.parallel_cluster_proc = []

            # test = list(range(todo_count))
            # self.iso_numbers= iso_numbers= list(range(1, todo_count+1))

            # for testing
            # self.iso_numbers= iso_numbers= list(range(1, 100+1))


            # isoIter.__next__()
            # isomorphs = [isoIter.__next__() for x in iso_numbers]

            # self.iso_numbers = iso_numbers = list(range(1, todo_count + 1))

            # TODO this is the slow part what to do, Old and slow
            num_cores = multiprocessing.cpu_count()

            test = 1

            # This aint good either
            # results = Parallel(n_jobs=self.num_threads)(delayed(self.calc_similarity)(i) for i in isoIter)

            test = 1

            # self.isomorphs = {str(x):isoIter.__next__() for x in iso_numbers}
            # function_arguments_to_call = [[x, self.isomorphs[str(x)], dockMol, origMol] for x in
            #                               self.iso_numbers]
            #
            # pool = multiprocessing.Pool(self.num_threads)
            #
            #
            # # this does not work
            # # function_arguments_to_call = [[x, isoIter, dockMol, origMol] for x in
            # #                               self.iso_numbers]
            #
            # test = 1
            #
            #
            # results = pool.starmap(self.calc_similarity, function_arguments_to_call)
            #
            # test = 1
            #
            # self.all_scores = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
            # globalScore = 0
            # bestNode = None
            #
            # for iso_num in self.iso_numbers:
            #
            #     currScore  = self.all_scores[str(iso_num)]
            #     if currScore > globalScore:
            #         globalScore = currScore
            #         bestNode = self.isomorphs[str(iso_num)]
            #
            #
            # self.bestNodus = bestNode
            # self.bestScorus = globalScore
            #
            # print('best score = ', globalScore)
            # print('best nodes ', bestNode)
            L = todo_count  # limit for performance timing
            p = multiprocessing.Pool(self.num_threads)
            n_count = 0
            bestNode = None
            globalScore = 0
            for (score, similar_nodus) in p.imap_unordered(self.calc_similarity, isoIter, 1000):
                if n_count >= L:
                    break
                if score > globalScore:
                    globalScore = score
                    bestNode = similar_nodus
                n_count += 1

            test = 1
        elif self.parallel_compute is False:
            print('tada is ', isoIter)
            count = 0
            globalScore = 0
            bestNode = None
            for iso in isoIter:
                # if count > 100000:
                if count > todo_count:
                    break
                similar_nodes = iso
                # print(similar_nodes)
                currScore = self.score(similar_nodes, dockMol, origMol)
                if currScore > globalScore:
                    globalScore = currScore
                    bestNode = similar_nodes
                count += 1
                # print('count ',count)

            print('best score = ', globalScore)
            print('best nodes ', bestNode)

        similar_nodes = bestNode
        similarKeys = similar_nodes.keys()
        final_Data = []
        for dockIndex in similarKeys:
            print('dockIndex ', dockIndex)
            print('modifKey ', similar_nodes[dockIndex])
            atomDock = dockMol[dockIndex]
            atomModif = modifMol[similar_nodes[dockIndex]]
            print('atomDock ', atomDock)
            print('atomModif ', atomModif)
            atom1X = float(atomDock[6])
            atom1Y = float(atomDock[7])
            atom1Z = float(atomDock[8])
            atomModif[6] = atom1X
            atomModif[7] = atom1Y
            atomModif[8] = atom1Z
            # final_Data.append(atomModif) #BUG BUG ERROR YIKES
            print('---------------------------------------')

        self.modifMol = modifMol

    def score_parallel(self, similar_nodes):
        score = 0
        dockNode = list(similar_nodes.keys())
        for node in dockNode:
            temp1 = self.dockMol[node][2]
            temp2 = self.origMol[similar_nodes[node]][2]
            if temp1 == temp2:
                score += 1
                # else:
                #     score -= 10
        return score

    # TODO not working parallel implementation
    def compare_graphs_parallel(self, dockGraph, origGraph, dockMol, origMol,
                                todo_count=100000):
        modifMol = origMol[:]

        print('origraph nodes are ', origGraph.nodes())
        origAtomTypes = nx.get_node_attributes(origGraph, 'atomType')
        origAtomTopoName = nx.get_node_attributes(origGraph, 'atomTopoName')
        # print('baboom ',origAtomTypes)
        # print('baboomUltra ',origAtomTopoName )
        atomTypes, atomTopoNames = self.getAttrubutesFromNodes(origAtomTypes, origAtomTopoName)
        # origNodes = origGraph.nodes()
        # for node in origNodes:
        #     print(node.atomType)
        #     print(node.atomTopoName)

        nm = isomorphism.generic_node_match('atomTopoName', 'C14', eq)  # Test Not Correct way to use
        is_Iso = nx.is_isomorphic(dockGraph, origGraph)  # no weights considered
        print('is it is_isomorphic ', is_Iso)
        GM = isomorphism.GraphMatcher(dockGraph, origGraph)  # ,  node_match=nm) # Didn't work
        print('tada ', GM.is_isomorphic())
        print('GM mapping moherfcuker', GM.mapping)

        # tadus = GM.match()
        # print('tada sdasjdajskdas ',tadus)
        # for i in tadus:
        #     print(i)


        isoIter = GM.isomorphisms_iter()

        print('tada is ', isoIter)
        count = 0
        globalScore = 0
        bestNodes = None

        num_proc = multiprocessing.cpu_count()
        print('Your number of CPUs is ', num_proc)

        self.dockMol = dockMol
        self.origMol = origMol

        # for iso in isoIter:
        #     # if count > 100000:
        #     if count > todo_count:
        #         break
        #     similar_nodes = iso
        #     # print(similar_nodes)

        pool = multiprocessing.Pool(num_proc)

        results = pool.map(self.score_parallel, isoIter)

        # if currScore > globalScore:
        #     globalScore = currScore
        #     bestNodes = similar_nodes
        # count += 1
        # # print('count ',count)

        print('best score = ', globalScore)
        print('best nodes ', bestNodes)

        similar_nodes = bestNodes
        similarKeys = similar_nodes.keys()
        final_Data = []
        for dockIndex in similarKeys:
            print('dockIndex ', dockIndex)
            print('modifKey ', similar_nodes[dockIndex])
            atomDock = dockMol[dockIndex]
            atomModif = modifMol[similar_nodes[dockIndex]]
            print('atomDock ', atomDock)
            print('atomModif ', atomModif)
            atom1X = float(atomDock[6])
            atom1Y = float(atomDock[7])
            atom1Z = float(atomDock[8])
            atomModif[6] = atom1X
            atomModif[7] = atom1Y
            atomModif[8] = atom1Z
            # final_Data.append(atomModif) #BUG BUG ERROR YIKES
            print('---------------------------------------')

        self.modifMol = modifMol

    def get_correct_topology_filename(self):
        return self.corr_topology_output_name

    def write_correct_topology(self):
        file_to_write = open(self.corr_topology_output_name, 'w')
        for row in self.modifMol:
            # print(row)
            writeRow = pdb_tools.write_row_pdb(row)
            # print('tada ',writeRow)
            file_to_write.write(writeRow)
        # pdbParseWrite.write_Lig(molData,file_to_write)
        file_to_write.close()
