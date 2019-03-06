
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


import os
import math
import multiprocessing

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

from rdkit.Chem import MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate


from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign

import pybel

from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import numpy as np
import pandas as pd

from molmolpy.utils import folder_utils



def get_circular_similarity(correct_ligand, mol_to_fix,
                            type_fp='bv', use_features=False):
    type_fp = type_fp
    correct_ligand_fingerprint = SimilarityMaps.GetMorganFingerprint(correct_ligand, radius=1, nBits=4096, fpType=type_fp,
                                                                     useFeatures=use_features)
    mol_to_fix_fingerprint = SimilarityMaps.GetMorganFingerprint(mol_to_fix, radius=1, nBits=4096, fpType=type_fp,
                                                                     useFeatures=use_features)





    sim_func = DataStructs.FingerprintSimilarity

    temp_autoinducer_TanimotoSimilarty = sim_func(correct_ligand_fingerprint, mol_to_fix_fingerprint,
                                                  metric=DataStructs.TanimotoSimilarity)

    test = 1

    curr_metric = DataStructs.TanimotoSimilarity
    fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(correct_ligand,
                                                                   mol_to_fix,
                                                                   lambda m, idx: SimilarityMaps.GetMorganFingerprint(m,
                                                                                                                      atomId=idx,
                                                                                                                      radius=1,
                                                                                                                      fpType='bv'),
                                                                   metric=curr_metric)
    # print(maxweight)
    # fig.suptitle('test title', fontsize=20)
    ax = fig.gca()
    # ax.title
    plt.title('test', fontsize=30)
    fig.set_size_inches(7, 7)
    fig.set_dpi(600)
    fig.savefig(
        'test.png',
        bbox_inches='tight')


    test = 1

    # temp_autoinducer_DiceSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                            metric=DataStructs.DiceSimilarity)
    # temp_autoinducer_CosineSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                              metric=DataStructs.CosineSimilarity)
    # temp_autoinducer_SokalSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                             metric=DataStructs.SokalSimilarity)
    # temp_autoinducer_RusselSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                              metric=DataStructs.RusselSimilarity)
    # temp_autoinducer_RogotGoldbergSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                                     metric=DataStructs.RogotGoldbergSimilarity)
    # temp_autoinducer_AllBitSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                              metric=DataStructs.AllBitSimilarity)
    #
    # temp_autoinducer_KulczynskiSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                                  metric=DataStructs.KulczynskiSimilarity)
    #
    # temp_autoinducer_McConnaugheySimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                                    metric=DataStructs.McConnaugheySimilarity)
    #
    # temp_autoinducer_AsymmetricSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                                  metric=DataStructs.AsymmetricSimilarity)
    #
    # temp_autoinducer_BraunBlanquetSimilarity = sim_func(correct_ligand_fingerprint, correct_ligand_fingerprint,
    #                                                     metric=DataStructs.BraunBlanquetSimilarity)

    # all_temps = [temp_autoinducer_TanimotoSimilarty,
    #              temp_autoinducer_DiceSimilarity,
    #              temp_autoinducer_CosineSimilarity,
    #              temp_autoinducer_SokalSimilarity,
    #              temp_autoinducer_RusselSimilarity,
    #              temp_autoinducer_RogotGoldbergSimilarity,
    #              temp_autoinducer_AllBitSimilarity,
    #              temp_autoinducer_KulczynskiSimilarity,
    #              temp_autoinducer_McConnaugheySimilarity,
    #              temp_autoinducer_AsymmetricSimilarity,
    #              temp_autoinducer_BraunBlanquetSimilarity]
    #
    # all_temps_np = np.array(all_temps)
    # all_temp_average = np.mean(all_temps_np)
    #
    # pandas_columns = ['Hormone', 'PubChem ID', 'ID', 'Tanimoto', 'Dice', 'Cosine', 'Sokal', 'Russel', 'RogotGoldberg',
    #                   'AllBit', 'Kulczynski', 'McConnaughey', 'Asymmetric', 'BraunBlanquet', 'Average']
    #
    # analysis_frame = pd.DataFrame(columns=pandas_columns)
    #
    # analysis_frame.loc[0] = ['3-O-C12 HSL', 127864, 0] + all_temps + [all_temp_average]
    #
    # index = 1
    # for id in accession_numbers:
    #     curr_hormone = hormone_names[index - 1]
    #     curr_pubchem_id = pubchem_id[index - 1]
    #
    #     # if type(curr_pubchem_id) is int:
    #     #     curr_pubchem_id = int(pubchem_id)
    #     # else:
    #     #     curr_pubchem_id = -1
    #
    #     temp_sim_TanimotoSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                            metric=DataStructs.TanimotoSimilarity)
    #     temp_sim_DiceSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                        metric=DataStructs.DiceSimilarity)
    #     temp_sim_CosineSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                          metric=DataStructs.CosineSimilarity)
    #     temp_sim_SokalSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                         metric=DataStructs.SokalSimilarity)
    #     temp_sim_RusselSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                          metric=DataStructs.RusselSimilarity)
    #     temp_sim_RogotGoldbergSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                                 metric=DataStructs.RogotGoldbergSimilarity)
    #     temp_sim_AllBitSimilarity = DataStructs.FingerprintSimilarity(correct_ligand_fingerprint,
    #                                                                   molecule_fingerprints[int(id)],
    #                                                                   metric=DataStructs.AllBitSimilarity)
    #
    #     temp_sim_KulczynskiSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                              metric=DataStructs.KulczynskiSimilarity)
    #
    #     temp_sim_McConnaugheySimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                                metric=DataStructs.McConnaugheySimilarity)
    #
    #     temp_sim_AsymmetricSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                              metric=DataStructs.AsymmetricSimilarity)
    #
    #     temp_sim_BraunBlanquetSimilarity = sim_func(correct_ligand_fingerprint, molecule_fingerprints[int(id)],
    #                                                 metric=DataStructs.BraunBlanquetSimilarity)
    #
    #     all_temps = [temp_sim_TanimotoSimilarity,
    #                  temp_sim_DiceSimilarity,
    #                  temp_sim_CosineSimilarity,
    #                  temp_sim_SokalSimilarity,
    #                  temp_sim_RusselSimilarity,
    #                  temp_sim_RogotGoldbergSimilarity,
    #                  temp_sim_AllBitSimilarity,
    #                  temp_sim_KulczynskiSimilarity,
    #                  temp_sim_McConnaugheySimilarity,
    #                  temp_sim_AsymmetricSimilarity,
    #                  temp_sim_BraunBlanquetSimilarity]
    #
    #     all_temps_np = np.array(all_temps)
    #     all_temp_average = np.mean(all_temps_np)
    #
    #     analysis_frame.loc[index] = [curr_hormone, curr_pubchem_id, int(id)] + all_temps + [all_temp_average]
    #     index += 1
    #
    # return analysis_frame


def get_atom_pair_similarity(correct_ligand, mol_to_fix,
                            type_fp='normal', use_features=False):
    type_fp = type_fp


    correct_ligand_fingerprint = SimilarityMaps.GetAPFingerprint(correct_ligand, nBits=4096, fpType=type_fp,
                                                                     )
    mol_to_fix_fingerprint = SimilarityMaps.GetAPFingerprint(mol_to_fix, nBits=4096,  fpType=type_fp,
                                                                     )


    test = 1


    sim_func = DataStructs.FingerprintSimilarity

    temp_autoinducer_TanimotoSimilarty = sim_func(correct_ligand_fingerprint, mol_to_fix_fingerprint,
                                                  metric=DataStructs.TanimotoSimilarity)

    test = 1

    curr_metric = DataStructs.TanimotoSimilarity
    fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(correct_ligand,
                                                                   mol_to_fix,
                                                                   lambda m, idx: SimilarityMaps.GetAPFingerprint(m,
                                                                                                                      atomId=idx,
                                                                                                                      fpType='normal'),
                                                                   metric=curr_metric)
    # print(maxweight)
    # fig.suptitle('test title', fontsize=20)
    ax = fig.gca()
    # ax.title
    plt.title('test', fontsize=30)
    fig.set_size_inches(7, 7)
    fig.set_dpi(600)
    fig.savefig(
        'test_ap.png',
        bbox_inches='tight')


    test = 1


def get_gobbi_similarity(correct_ligand, mol_to_fix,
                            type_fp='normal', use_features=False):
    # ref = Chem.MolFromSmiles('NC(=[NH2+])c1ccc(C[C@@H](NC(=O)CNS(=O)(=O)c2ccc3ccccc3c2)C(=O)N2CCCCC2)cc1')
    ref = Chem.MolFromSmiles('C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O')
    # mol1 = Chem.MolFromPDBFile(RDConfig.RDBaseDir + '/rdkit/Chem/test_data/1DWD_ligand.pdb')
    mol1 = AllChem.AssignBondOrdersFromTemplate(ref, correct_ligand)
    # mol2 = Chem.MolFromPDBFile(RDConfig.RDBaseDir + '/rdkit/Chem/test_data/1PPC_ligand.pdb')
    mol2 = AllChem.AssignBondOrdersFromTemplate(ref, mol_to_fix)

    factory = Gobbi_Pharm2D.factory
    fp1 = Generate.Gen2DFingerprint(mol1, factory, dMat=Chem.Get3DDistanceMatrix(mol1))
    fp2 = Generate.Gen2DFingerprint(mol2, factory, dMat=Chem.Get3DDistanceMatrix(mol2))
    # Tanimoto similarity
    tani = DataStructs.TanimotoSimilarity(fp1, fp2)
    print('GOBBI similarity is ------> ', tani)