# -*- coding: utf-8 -*-
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


import csv
import os
import sys

import hdbscan
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pdbParseWrite  # My module for parsing and writing verify yolo
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from sklearn.cluster import AffinityPropagation


def find_folders(folderInit):
    try:
        dir_names = []
        path = '.'
        subdirectories = os.listdir(path)
        # print(subdirectories)
        for dirname in subdirectories:
            if folderInit in dirname:  # 'vina_sample'
                dir_names.append(dirname)
        # print sorted(dir_names)
        return sorted(dir_names)
    except Exception as e:
        print("Problem with finding folders : ", e)
        sys.exit(0)


def find_sample_folders():
    try:
        dir_names = []
        for dirname, dirnames, filenames in os.walk('.'):
            # print(dirname, '-')
            if 'vina_sample' in dirname:  #
                dir_names.append(dirname)
        # print sorted(dir_names)
        return sorted(dir_names)
    except Exception as e:
        print("Problem with finding folders : ", e)
        sys.exit(0)


def find_VIT_file(folder):
    try:
        VIP = []
        for dirname, dirnames, filenames in os.walk(folder):
            # print dirname, '-'
            # print filenames
            for i in filenames:
                # print i
                if 'out' in i:
                    VIP.append(i)
                elif 'vina_sample_' in i:
                    VIP.append(i)
        return VIP
    except Exception as e:
        print("error in find_files: ", e)
        sys.exit(0)


def find_median(Mol):
    Xm = 0
    Ym = 0
    Zm = 0
    for line in Mol:
        Xm += float(line[6])
        Ym += float(line[7])
        Zm += float(line[8])
    Xm = Xm / len(Mol)
    Ym = Ym / len(Mol)
    Zm = Zm / len(Mol)
    return Xm, Ym, Zm


def cleanVal(val):
    tempus = val.split(' ')
    # print tempus
    for i in tempus:
        if i == '':
            pass
        else:
            return i


def extractInfo(line):
    lineSplit = line.split(' ')
    # print('lineSplit ',lineSplit)
    cleanedVal = []
    for i in lineSplit:
        if i == '':
            pass
        else:
            cleanedVal.append(i)
    return cleanedVal


def PDBQTparse(filename):
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
                tempInfo = extractInfo(line)
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
            if cleanVal(line[0:7]) == 'ATOM' or cleanVal(line[0:7]) == 'HETATM':
                temp.append(cleanVal(line[0:7]))  # Record name [0]
                temp.append(cleanVal(line[7:12]))  # Atom serial number [1]
                temp.append(cleanVal(line[12:17]))  # Atom name [2]
                temp.append(cleanVal(line[17:21]))  # Residue name [3]
                temp.append(cleanVal(line[21:23]))  # Chain identifier [4]
                temp.append(cleanVal(line[23:27]))  # Residue sequence number [5]
                temp.append(cleanVal(line[31:39]))  # Orthogonal coordinates for X in Angstroms [6]
                temp.append(cleanVal(line[39:47]))  # Orthogonal coordinates for Y in Angstroms [7]
                temp.append(cleanVal(line[47:55]))  # Orthogonal coordinates for Z in Angstroms [8]
                temp.append(cleanVal(line[55:61]))  # Occupancy [9]
                temp.append(cleanVal(line[61:67]))  # Temperature factor [10]
                temp.append(cleanVal(line[67:76]))  # Real(10.4)    partialChrg  Gasteiger PEOE partial charge *q*.
                temp.append(cleanVal(line[77:79]))  # Element symbol, right-justified. [11]
                temp.append(cleanVal(line[79:81]))  # Element symbol, right-justified. [12]
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


def write_preProcData_log(sampleDataDict=None):
    try:

        authorInfo = '''#################################################################
    # If you used Axmax  Vina in your work, please cite:            #
    #                                                               #
    # H. Grabski                                                    #
    # Axmax    Vina: improving axmaxutyun in Scientific Research    #
    # with Machine Learning                                         #
    #                                                               #
    # Please see https://github.com/hovo1990/GROM for more info     #
    #################################################################\n
    '''

        dataInfo = '''
     Sample    | mode |   affinity | dist from best mode | X center   | Y center   | Z center   |
               |      | (kcal/mol) | rmsd l.b.| rmsd u.b.| (Angstrom) | (Angstrom) | (Angstrom) |
---------------+------+------------+----------+----------+------------+------------+------------+\n'''

        # string = '->'*20 + '\n'
        # string += 'Simulation PDB_move Settings Log\n'
        # string +=  'times: %s\nreceptor Name: %s\n'%(times,recName)
        # string += 'ligand Name: %s\n'%(ligName)
        # string += 'distance R from Pivot: %s (in Angstroms)\n' %(R)
        # string += 'Theta Rotation: %s\n' %(theta_n)
        # string += '--'*20 + '\n'
        # print(string)

        data_string = ''
        # keys = sorted()
        for sample in sorted(sampleDataDict):
            # print('sample is ',sample)
            data_string += sample + '\n'
            for model in sampleDataDict[sample]:
                temp_string = ''
                # print('model is ',model)
                temp_string += ' ' * 16
                temp_string += '{:^8}'.format(str(model[0][0]))
                temp_string += '{:^13}'.format(str(model[0][1]))
                temp_string += '{:^12}'.format(str(model[0][2]))
                temp_string += '{:^10}'.format(str(model[0][3]))
                X = round(model[0][4], 4)
                Y = round(model[0][5], 4)
                Z = round(model[0][6], 4)
                temp_string += '{:^13}'.format(str(X))
                temp_string += '{:^13}'.format(str(Y))
                temp_string += '{:^13}'.format(str(Z))
                temp_string += '\n'
                # print('tada ',temp_string)
                data_string += temp_string

                # print('sample is ',sample,sampleDataDict[sample])

        settings_log = open('preProcessedData.log', 'w')
        settings_log.write(authorInfo)
        settings_log.write(dataInfo)
        settings_log.write(data_string)
        settings_log.close()
    except Exception as e:
        print("Could not write processed data verify yolo")


def runVinaSamplePreAnalysis(folders):  # Need to modify it verify yolo
    try:
        sampleData = {}
        molDatus = []
        infoDatus = []
        count = 0
        for folder in folders:  # VIP
            working_folder = folder[2:]
            # print('working folder is ',working_folder)
            VIP_files = sorted(find_VIT_file(folder))
            # print('VIP files is ',VIP_files)
            os.chdir(working_folder)

            temp_info, temp_PDBQT_file = PDBQTparse(VIP_files[-1])
            print(len(temp_info))
            print(len(temp_PDBQT_file))

            sampleData.update({working_folder: []})  # What a noobie mistake i made before :D
            for info, Mol in zip(temp_info, temp_PDBQT_file):
                # print('info ',info)
                # print('Mol ',Mol)
                infoMol = [count, folder[2:]] + info[:]
                infoDatus.append(infoMol)
                molDatus.append(Mol)
                Xm, Ym, Zm = find_median(Mol)
                center = [Xm, Ym, Zm]
                # print('center ',center)
                data_to_add = [infoMol + center]
                print('infoMol ', infoMol)
                print('data_to_add ', data_to_add)
                sampleData[working_folder].append(data_to_add)  # Very serious modification
                count += 1

            print('-*-' * 50)
            # print('Lets go back')
            os.chdir('..')
        # Cycle ends Here
        # Time to write preProcessed Data
        # write_preProcData_log(sampleData) #Already wrote verify yolo
        # keys = list(sampleData.keys())
        # print(keys[0].split('sample'))
        # print(sorted(keys))
        return sampleData, molDatus, infoDatus
    except Exception as e:
        print("Error with Main Function ", e)
        sys.exit(0)
        # finally:
        #     shutdown()


def runVinaLocalPreAnalysis(folders):
    try:
        folderData = {}
        for folder in folders:  # VIP
            print('folder is ', folder)
            working_folder = folder
            os.chdir(working_folder)
            sampleFolders = find_sample_folders()
            data_To_Analyze, molData, infoData = runVinaSamplePreAnalysis(sampleFolders)
            folderData.update({working_folder: data_To_Analyze})

            print('-*-' * 50)
            # print('Lets go back')
            os.chdir('..')
        # Cycle ends Here
        # Time to write preProcessed Data
        # write_preProcData_log(sampleData) #Already wrote verify yolo
        # keys = list(sampleData.keys())
        # print(keys[0].split('sample'))
        # print(sorted(keys))
        return folderData, molData, infoData
    except Exception as e:
        print("Error with  runVinaLocalPreAnalysis Function ", e)
        sys.exit(0)
        # finally:
        #     shutdown()


def machinePreAnalysis(data_To_Analyze):
    # print(data_To_Analyze)
    final_Data = np.empty((0, 5), float)  # verify yoloLSDOSDS
    labels_Data = np.empty((0, 1), float)
    # print(final_Data, final_Data.shape, final_Data.dtype) #verify yolo important
    for sample in sorted(data_To_Analyze):
        # print('sample ', sample)
        for model in data_To_Analyze[sample]:
            temp_data = []
            print('model checking  is ', model)

            X = float(model[0][6])
            Y = float(model[0][7])
            Z = float(model[0][8])
            temp_data.append(X)
            temp_data.append(Y)
            temp_data.append(Z)
            temp_data.append(float(model[0][3]))  # deltaG
            temp_data.append(int(model[0][0]))  # Model count
            print('temp data is ', temp_data)
            temp_data = np.array([temp_data])
            # print(temp_data, temp_data.shape, temp_data.dtype)
            temp_label = np.array([[float(model[0][3])]])
            # print(temp_label, temp_label.shape, temp_label.dtype)
            # final_Data.append(temp_data)
            final_Data = np.append(final_Data, temp_data, axis=0)
            labels_Data = np.append(labels_Data, temp_label, axis=0)
    # print('verify yolo ',final_Data)
    return final_Data, labels_Data


def plotSimpleData(data_dict):
    fignum = 1
    fig = plt.figure(fignum)
    plt.clf()
    ax = Axes3D(fig)

    # print('xyz is ',xyz)
    plt.cla()

    keys = list(data_dict.keys())
    colors = plt.cm.Spectral(np.linspace(0, 1, len(keys)))
    # print('keys is ',keys)

    for i in range(len(keys)):
        key = keys[i]
        # print('key is ',key)
        ax.scatter(data_dict[key][0], data_dict[key][1], data_dict[key][2], 'o', s=800, c=colors[i],
                   label="key %s deltaG= %s " % (key, data_dict[key][3]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend(loc='lower left', ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    plt.title('Tada')
    plt.show()  # not now


def scatterProtein(ax):
    name = 'LASR'
    filename = 'em2A.pdb'

    structLASR = pdbParseWrite.proParser(filename, name)

    for model in structLASR:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print('verify yolo ', atom.get_coord())
                    if atom.element == 'C':
                        coords = atom.get_coord()
                        ax.scatter(coords[0], coords[1], coords[2], '+', s=10,
                                   c='b')  # label = "deltaG = %s±%s (%s%%) label=%s" %(round(bind_mean,2),round(bind_std,2), percentage,k))


def select_BestRows(data, deltaG_threshold, mu):
    best_deltaG = round(mu, 1)  # data[0][-1] # float(mu)
    print("best deltaG is ", best_deltaG)
    # print('data is ',data)
    deltaG_threshold = deltaG_threshold
    extractedData = []
    for i in range(len(data)):
        temp_Bind = data[i][3]
        # print('temp bind ',temp_Bind)
        if (temp_Bind >= best_deltaG - deltaG_threshold and temp_Bind <= best_deltaG + deltaG_threshold):
            extractedData.append(data[i])
            # print('i is ',i)
            #     pass
            # else:
            #     print('Yahoo')
            #     extractedData = data[0:i]
            #     break
    # extractedData = data[(data[:, 3] == best_deltaG)]
    # print('extracted Data',extractedData)
    print('len ', len(extractedData))
    clean_Data = np.empty((0, 4))  # This is for clusters
    # print('pre_clean ',clean_Data.shape) #This is troublesome
    for i in extractedData:
        # print('i is ',i)
        temp_data = np.array([[i[0], i[1], i[2], i[3]]])  # THis was modified
        # print('temp_data ',temp_data.shape)
        clean_Data = np.append(clean_Data, temp_data, axis=0)  # Here lies a problem
    # print('cleanData ', clean_Data)
    return clean_Data


def select_AllData(data):
    extractedData = data[:]
    print('len ', len(extractedData))
    clean_Data = np.empty((0, 4))  # This is for clusters
    # print('pre_clean ',clean_Data.shape) #This is troublesome
    for i in extractedData:
        # print('i is ',i)
        temp_data = np.array([[i[0], i[1], i[2], i[3]]])  # THis was modified
        # print('temp_data ',temp_data.shape)
        clean_Data = np.append(clean_Data, temp_data, axis=0)  # Here lies a problem
    # print('cleanData ', clean_Data)
    return clean_Data


def plotHistogramm(data, show=False):
    fignum = 1
    fig = plt.figure(fignum)
    plt.clf()
    (mu, sigma) = norm.fit(data)
    # weights = np.ones_like(data)/float(len(data))
    n, bins, patches = plt.hist(data, 25, normed=True, color='green', alpha=0.75)  # VIP
    print('n is ', n, ' len(n) ', len(n))
    print('bins is ', bins, ' len(bins) ', len(bins))
    print('patches is ', patches)
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.title(r'$\mathrm{Histogram\ of\ binding Energy:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    if show == True:
        plt.show()
    return mu, sigma


def analyze_Data_Extreme(X, cluster_Center_Data, labels):
    print('analyze_Data_Extreme(X, cluster_Center_Data) ----->>>>>')
    Center_data = cluster_Center_Data  # np.sort(cluster_Center_Data, order = ['bindMean' ]) #'rmsd','percentage','
    print('--------------------------------------------')
    print("Sorted Center Data is ", Center_data)
    print('--------------------------------------------')

    # most_points = sorted_percent[-1]
    # # print('most_points is ',most_points)


    extractedData = {}
    # SAVE ALL DATA
    # label_key = sorted_Center_data[0][0] #I select key with most points this is the problem, but now what? Most important Part
    for label_key in Center_data:
        xyz = X[labels == label_key[0]]
        # print('xyz is ',xyz)
        print('xyz len ', len(xyz))
        mu, sigma = plotHistogramm(xyz[:, 3], show=False)
        print('mu is ', mu)
        sorted_Data = convert_toSortableData(xyz)  ##Func 1
        print('Sorted data ', sorted_Data)  # This is Ok, but
        threshold = round(float(sigma), 4)  # VIP
        print('threshold ', threshold)

        # extractedData = select_BestRows(sorted_Data,threshold , mu) #Func 2
        tempus = select_AllData(sorted_Data)
        extractedData.update({label_key[0]: tempus})  # This one has a bug
        # #print('len(extr) ',extractedData)
        # #plotExtractedData(extractedData)
    return extractedData


def extractus_datus(dataFolders):
    posData_dict = {}
    keys = list(dataFolders.keys())
    for i in keys:
        temp_data = dataFolders[i]
        final_data, labels_Data = machinePreAnalysis(temp_data)
        X, labels, cluster_Center_Data, label_percentage, molOrder = machineAnalysis(final_data,
                                                                                     labels_Data)  # Modified
        print('label_percentage is ', label_percentage)
        # print(X)
        # TODO need to modify this for
        posData = analyze_Data_Extreme(X, cluster_Center_Data, labels)  # VIP need to change this
        # print('posData is ',posData)
        posData_dict.update({i: posData})  # This is for writing PDB coordinates myabe add energy
        # machineAnalysisStep2(step2Data)
        # print('key is ',i)
    # print('posDataDict ',posData_dict) #So here it is
    return posData_dict, molOrder


def convert_toSortableData(xyz):
    dtype = [('Xc', np.float64), ('Yc', np.float64), ('Zc', np.float64), ('bindEn', np.float64)]
    clean_Data = np.empty((0,), dtype=dtype)  # This is for clusters
    for i in xyz:
        # print('i is ',i)
        temp_data = np.array([(i[0], i[1], i[2], i[3])], dtype=dtype)
        # print('verify yolo temp Data ',temp_data, temp_data.shape)
        # temp_data = np.array(temp_data, dtype =clean_Data.dtype) #Here's also a problem
        clean_Data = np.append(clean_Data, temp_data, axis=0)  # Here lies a problem
    sorted_Data = np.sort(clean_Data, order='bindEn')
    # print(sorted_Data)
    # print(len(sorted_Data))

    return sorted_Data


def rmsd_fromBest_deltaG(sorted_data):
    best_deltaG_pos = sorted_data[0]
    print('best_deltaG :--> ', best_deltaG_pos)

    rmsd = 0

    for pos in sorted_data[1:]:
        rmsd += (best_deltaG_pos[0] - pos[0]) ** 2 + (best_deltaG_pos[1] - pos[1]) ** 2 + (best_deltaG_pos[2] - pos[
            2]) ** 2
    rmsd = np.sqrt(rmsd / len(sorted_data))
    print("RMSD of cluster from best point is ", rmsd)
    return round(rmsd, 3)


def minimize_function(preference_val, X, y):
    db = AffinityPropagation(damping=.9, preference=preference_val).fit(X[:, [0, 2]], y)  # IMPORTA
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('len is ', n_clusters_)

    return n_clusters_


def plot_frequency(data, show=True):
    fignum = 2
    fig = plt.figure(fignum)
    plt.clf()
    (mu, sigma) = norm.fit(data)
    # weights = np.ones_like(data)/float(len(data))
    n, bins, patches = plt.hist(data, normed=False, alpha=0.75)  # VIP
    print('n is ', n, ' len(n) ', len(n))
    print('bins is ', bins, ' len(bins) ', len(bins))
    print('patches is ', patches)
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.title(r'$\mathrm{Histogram\ of\ frequency:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    if show == True:
        plt.show()


def machineAnalysis(X, binding_energy):
    print('len of Data is ', X.shape[0])
    fignum = 1
    fig = plt.figure(fignum)
    plt.clf()
    ax = Axes3D(fig)
    y = binding_energy
    # X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    # db = DBSCAN(eps=2, min_samples=100).fit(X[:,[0,2]],y) #IMPORTANT

    # x0 = np.array([-500, -700, -1000, -1600, -2000])
    # # res = minimize(minimize_function, x0, args = (X,y), method='nelder-mead',
    # #                options = {'xtol': 1e-8, 'disp': True})
    # x_stuff = X
    # y_stuff =y
    # x0 = -3000
    # res = optimize.bisect(minimize_function,-6000,6000, xtol=1e-12, args=(x_stuff, y_stuff,))
    # res = optimize.minimize_scalar(minimize_function,  args=(x_stuff, y_stuff,))
    # # res = optimize.minimize(minimize_function,x0, args=(x_stuff, y_stuff,), method='Nelder-Mead')
    # print(res)
    # sys.exit(0)
    # db = DBSCAN(eps=0.2, min_samples=100).fit(X[:, [0, 2]], y)  # IMPORTANT
    # db = AffinityPropagation(damping=.9).fit(X[:,[0,2]],y) #IMPORTA


    # HDB SCAN MOTHERcheckER
    db = hdbscan.HDBSCAN(min_cluster_size=125)
    labels = db.fit_predict(X[:, [0, 2]], y)
    print('labels ', labels)
    #
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    # print('labels is ',labels)
    print('labels shape is ', labels.shape[0])
    # print('db  are  ',db.components_)
    labelsShape = labels.shape[0]

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # plot_frequency(labels)

    print('Estimated number of clusters: %d' % n_clusters_)
    unique_labels = list(set(labels))
    print('Unique labels ', unique_labels)

    worthy_data = labels[labels != -1]
    notWorthy_data = labels[labels == -1]
    real_labels = set(worthy_data)
    # print("Worthy Data ",worthy_data)
    print("Real Labels man ", real_labels)
    shape_worthy = worthy_data.shape[0]
    print("All Worthy data points ", int(shape_worthy))
    print("Not Worthy data points ", int(notWorthy_data.shape[0]))

    plt.cla()

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # print("Colors is ",colors)

    # Here could be the solution
    dtype = [('label', np.int8), ('CLx', np.float64), ('CLy', np.float64), ('CLz', np.float64),
             ('bindMean', np.float64),
             ('bindStd', np.float64), ('quantity', int), ('percentage', np.float64), ('rmsd', np.float64), ]
    cluster_Center_Data = np.empty((0,), dtype=dtype)  # This is for clusters
    # print("cluster_Center_Data ",clean_Data, clean_Data.shape)
    # print("clean Data dtype ", clean_Data.dtype)
    # print("clean Data [0] dtype" ,dtype[0])

    label_percent = {}
    # Need to return X, clean_data, and another dict for best position


    molOrder = {}
    for k in unique_labels:  # Need to modify WORKS
        # print('k is ',k)
        xyz = X[labels == k]
        if k == -1:
            color = 'b'
            print('what the hell ', xyz[:, 4])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], facecolor=(0, 0, 0, 0), marker='^', s=100, c=color,
                       label='Outlier')
        else:
            # Need to make this function a lot better
            print('xyz is ', xyz)
            sorted_data = convert_toSortableData(xyz)  # New stuff
            # print('Sorted Data is ',sorted_data)
            curr_rmsd = round(rmsd_fromBest_deltaG(sorted_data), 3)
            xyzShape = xyz.shape[0]
            Xc = np.mean(xyz[:, 0])
            # print('Xc is ', Xc, np.mean(xyz[:, 0]))
            Yc = np.mean(xyz[:, 1])
            Zc = np.mean(xyz[:, 2])

            # This part need to be modified to save in a dictionary for all cluster data
            molOrder.update(
                {k: xyz[:, 4]})  # This problematic not the most points, doesn't behave right when multiple clusters
            # print('Xc of cluster is ', Xc)
            bindEnergy_column = xyz[:, 3]
            bind_mean = np.mean(bindEnergy_column)
            bind_std = np.std(bindEnergy_column)
            bind_shape = bindEnergy_column.shape[0]
            percentage = round((bind_shape / float(labelsShape)) * 100, 2)  # This part represents a lot better
            label_percent.update({k: percentage})
            # print("curr array size is ",bindEnergy_column.shape )
            temp_data = np.array([(k, Xc, Yc, Zc, bind_mean, bind_std, bind_shape, percentage, curr_rmsd)], dtype=dtype)
            # print('verify yolo temp Data ',temp_data, temp_data.shape)
            # temp_data = np.array(temp_data, dtype =clean_Data.dtype) #Here's also a problem
            cluster_Center_Data = np.append(cluster_Center_Data, temp_data, axis=0)  # Here lies a problem

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', s=80, c=colors[k], edgecolor='g',
                       label="deltaG = %s±%s (%s%%) label=%s   rmsd = %s A" % (
                           round(bind_mean, 2), round(bind_std, 2), percentage, k, curr_rmsd))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend(loc='lower left', ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    plt.title('Estimated number of clusters: %d (%d/%d)' % (n_clusters_, shape_worthy, X.shape[0]))
    plt.show()  # not now
    # print("Clean Data len() ",len(cluster_Center_Data )) #Problem already Here
    # clean_DataSorted = np.sort(cluster_Center_Data , order = 'quantity')
    # print("Sorted verify yolo ",clean_DataSorted[::-1])
    return X, labels, cluster_Center_Data, label_percent, molOrder  # Something is wrong with the data


def postProcessing(posData_dict):
    simple = {}
    for i in posData_dict:
        Xc = np.mean(posData_dict[i][:, 0])
        # print('Xc is ', Xc, np.mean(xyz[:, 0]))
        Yc = np.mean(posData_dict[i][:, 1])
        Zc = np.mean(posData_dict[i][:, 2])
        # print('Xc of cluster is ', Xc)
        bindEnergy_column = posData_dict[i][:, 3]
        bind_mean = np.mean(bindEnergy_column)
        simple.update({i: [Xc, Yc, Zc, bind_mean]})
    # print('simple is ',simple)
    datus = list(simple.items())
    # print('datus ',datus)
    new_datus = []
    for i in datus:
        new_datus.append(i[-1])
    # print('new_datus is ',new_datus)
    pdbParseWrite.runMod(new_datus)
    # plotSimpleData(simple)


def writePDBQTs(pdbMols):
    molData = pdbMols[0]
    print(list(molData))
    file_to_write = open("ligBind.pdb", 'a')
    pdbParseWrite.write_Lig(molData, file_to_write)
    file_to_write.close()


def convertOneStrangeRow(mol, id):
    data = []
    for i in mol:
        # print('i is ',i)
        x = float(i[6])
        y = float(i[7])
        z = float(i[8])
        data.append(np.array([x, y, z, id]))
        # data += [id]
    npData = np.array(data)
    # print(len(npData) )
    return npData


# final_Data = np.empty((0,4), float) #verify yoloLSDOSDS

#                 final_Data = np.append(final_Data, temp_data, axis = 0)
#                 labels_Data = np.append(labels_Data, temp_label, axis = 0)


def extractAwesomeConformations(posData_dict=None):
    try:
        sampleData = {}
        currMol_num = 0
        # sampleData = []
        npRows = np.empty((0, 67))
        print('npRows shape ', npRows.shape)
        csvWriteData = csv.writer(open("frameEnergy.csv", "w"))
        for folder in posData_dict:  # VIP
            # print(' in extract folder is ',folder)
            working_folder = folder
            os.chdir(working_folder)
            sampleFolders = find_sample_folders()
            # print('sampleFolders ',sampleFolders)
            for folder_sample in sampleFolders:  # VIP
                working_SampleFolder = folder_sample[2:]
                # print('working folder is ',working_folder)
                VIP_files = sorted(find_VIT_file(folder_sample))
                # print('VIP files is ',VIP_files)
                os.chdir(working_SampleFolder)

                temp_info, temp_PDBQT_file = PDBQTparse(VIP_files[-1])
                # print(len(temp_info))
                # print(len(temp_PDBQT_file))
                # print(temp_PDBQT_file)

                # sampleData.update({working_folder:[]}) #What a noobie mistake i made before :D
                for info, Mol in zip(temp_info, temp_PDBQT_file):
                    # print('info ',info)
                    # print('Mol ',Mol)
                    Xm, Ym, Zm = find_median(Mol)
                    center = [Xm, Ym, Zm]
                    # print('center ',center)
                    clusterCenter = posData_dict[folder]
                    for clCenter in clusterCenter:
                        calcX = abs(Xm - clCenter[0]) < 0.000001
                        calcY = abs(Ym - clCenter[1]) < 0.000001
                        calcZ = abs(Zm - clCenter[2]) < 0.000001
                        if calcX == True and calcY == True and calcZ == True:
                            # print('Tada ',center, )
                            # print('Prev ', clCenter[0],clCenter[1],clCenter[2])
                            # print('Mol is ',Mol)
                            # sampleData[working_folder].append(Mol)
                            # sampleData.append(Mol)
                            sampleData.update({currMol_num: Mol})
                            csvWriteData.writerow([str(currMol_num), str(clCenter[3])])
                            # temp_npData = convertOneStrangeRow(Mol, currMol_num) #not working
                            # print('npRows shape ',npRows.shape)
                            # print('tempData shape ',temp_npData.shape)
                            # npRows = np.append(npRows, [temp_npData], axis = 0) #not working
                            currMol_num += 1
                            # sampleData[working_folder].append([info+center])

                # print('Lets go back')
                os.chdir('..')
            # Cycle ends Here
            print('-*-' * 20)
            os.chdir('..')
        # print('sampleData ', sampleData) #Interesting output
        print('npRows is ', npRows)
        # csvWriteData.close()
        return sampleData, npRows
    except Exception as e:
        print("Error with dadoo Function ", e)
        sys.exit(0)


def extractClusterMols(posData_dict, MolOrderDict):
    try:
        key_folder = list(posData_dict.keys())
        sampleDataDict = {}
        molDatusDict = {}
        infoDatusDict = {}
        for key in posData_dict[key_folder[0]]:
            currMol_num = 0
            # sampleData = []
            molDatus = []
            infoDatus = []
            sampleData = {}
            MolOrder = [int(x) for x in MolOrderDict[key]]
            print('motherchecker ', MolOrder)
            for folder in posData_dict:  # VIP
                # print(' in extract folder is ',folder)
                working_folder = folder
                os.chdir(working_folder)
                sampleFolders = find_sample_folders()
                # print('sampleFolders ',sampleFolders)
                for folder_sample in sampleFolders:  # VIP
                    working_SampleFolder = folder_sample[2:]
                    # print('working folder is ',working_folder)
                    VIP_files = sorted(find_VIT_file(folder_sample))
                    # print('VIP files is ',VIP_files)
                    os.chdir(working_SampleFolder)

                    temp_info, temp_PDBQT_file = PDBQTparse(VIP_files[-1])
                    # print(len(temp_info))
                    # print(len(temp_PDBQT_file))
                    # print(temp_PDBQT_file)

                    # sampleData.update({working_folder:[]}) #What a noobie mistake i made before :D
                    for info, Mol in zip(temp_info, temp_PDBQT_file):
                        # print('info ',info)
                        # print('Mol ',Mol

                        if int(currMol_num) in MolOrder:
                            # print('currMol_num ',currMol_num)
                            infoMol = [currMol_num, working_SampleFolder] + info[:]
                            infoDatus.append(infoMol)
                            molDatus.append(Mol)
                            sampleData.update({currMol_num: Mol})
                        currMol_num += 1



                        # for i in MolOrder:
                        #     #print('i is ',i)
                        #     if int(currMol_num) - int(i) < 0.001:
                        #         #print('currMol_num ',currMol_num)
                        #         infoMol = [currMol_num, working_SampleFolder] + info[:]
                        #         infoDatus.append(infoMol)
                        #         molDatus.append(Mol)
                        #         sampleData.update({currMol_num:Mol})
                        #         currMol_num += 1
                        #         break



                        # Xm, Ym, Zm = find_median(Mol)
                        # center = [Xm, Ym, Zm]
                        # #print('center ',center)
                        # clusterCenter = posData_dict[folder]
                        # for clCenter in clusterCenter:
                        #     calcX = abs(Xm - clCenter[0])  < 0.000001
                        #     calcY = abs(Ym - clCenter[1])  < 0.000001
                        #     calcZ = abs(Zm - clCenter[2])  < 0.000001
                        #     if calcX == True and calcY == True and calcZ == True and currMol_num in MolOrder:
                        #         print('Tada ',center, )
                        #         print('currMol_num ',currMol_num)
                        #         # print('Prev ', clCenter[0],clCenter[1],clCenter[2])
                        #         #print('Mol is ',Mol)
                        #         #sampleData[working_folder].append(Mol)
                        # sampleData.append(Mol)

                        # infoMol = [currMol_num, folder[2:]] + info[:]
                        # infoDatus.append(infoMol)
                        # molDatus.append(Mol)
                        # sampleData.update({currMol_num:Mol})

                        # currMol_num += 1
                        # sampleData[working_folder].append([info+center])

                    # print('Lets go back')
                    os.chdir('..')
                # Cycle ends Here
                print('-*-' * 20)
                os.chdir('..')
            # print('sampleData ', sampleData) #Interesting output
            # csvWriteData.close()
            sampleDataDict.update({key: sampleData})
            molDatusDict.update({key: molDatus})
            infoDatusDict.update({key: infoDatus})
        return sampleDataDict, molDatusDict, infoDatusDict
    except Exception as e:
        print("Error with dadoo Function ", e)
        sys.exit(0)


def writeTraj(pdbMols):
    # molData = pdbMols[0]
    # print(list(pdbMols))
    ids = sorted(list(pdbMols))
    print('ids is ', ids)
    file_to_write = open("ligBindTraj.pdb", 'w')

    for idI in ids:
        pdbParseWrite.write_Lig(pdbMols[idI], file_to_write)
        file_to_write.write('ENDMDL\n')
    file_to_write.close()


def extractTrajectory(molData, infoMol):
    keys = list(molData.keys())
    for key in keys:
        file_to_write = open("ligBindTraj_{0}.pdb".format(key), 'w')
        csvWriteData = csv.writer(open("frameEnergy_{0}.csv".format(key), "w"))
        for idI, infMol in zip(molData[key], infoMol[key]):
            csvWriteData.writerow([str(infMol[0]), str(infMol[1]), str(infMol[2]),
                                   str(infMol[3]), str(infMol[4]), str(infMol[5])])
            pdbParseWrite.write_Lig(idI, file_to_write)
            file_to_write.write('ENDMDL\n')
        file_to_write.close()


def auto_run(folders):
    try:
        dataFolders, molData, infoMol = runVinaLocalPreAnalysis(folders)  # alright this part work well
        # extractTrajectory(molData, infoMol)
        posData_dict, molOrder = extractus_datus(dataFolders)  # Prefinal
        print('posData_dict is ', posData_dict)
        print('molOrder ', molOrder)
        # postProcessing(posData_dict) #This is last step
        # print('lalooo ',dataFolders) #Contains only centers so it's not useful

        #  print('yahoo ', posData_dict)
        #  ##extract original data


        # TODO excract all clusters
        pdbMols, molDatus, infoDatus = extractClusterMols(posData_dict, molOrder)
        extractTrajectory(molDatus, infoDatus)

        # #  print('len of single pdbMols', len(pdbMols[1]))
        # #  #machineConformerAnalysis(mlRows) not the right time
        # #  #Important function
        # # # writePDBQTs(pdbMols)
        #  writeTraj(pdbMols) #This is buugy because of coordinate duplication

        # print(dataFolders['MonA'])
        # final_data, labels_Data = machinePreAnalysis(data_To_Analyze)
        # step2Data = machineAnalysis(final_data,labels_Data)
        # machineAnalysisStep2(step2Data)
        print('Tuto motherchecking bene :D')
    except (KeyboardInterrupt, SystemExit):
        print('Exiting automated script')
        sys.exit(0)


try:
    localFolders = find_folders('Mon')
    print(sorted(localFolders))
    print(len(localFolders))
    auto_run(localFolders)  # important


except Exception as e:
    print("error is ", e)
