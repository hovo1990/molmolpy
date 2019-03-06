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



from msmbuilder import featurizer
from msmbuilder.featurizer import DihedralFeaturizer
from numba import jit

from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd

#@jit
def chunks_mdtraj(ary, indices_or_sections, axis=0):
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle scalar case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.')
        Neach_section, extras = divmod(Ntotal, Nsections)


        test = 1
        section_sizes = ([0] +
                         extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])

        div_points = np.array(section_sizes).cumsum()

    sub_arys = []
    # sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]

        temp_array = ary[st:end]

        # sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
        sub_arys.append(temp_array)

    return sub_arys


def chunks_mdtraj_dict(ary, indices_or_sections, axis=0):
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle scalar case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.')
        Neach_section, extras = divmod(Ntotal, Nsections)


        test = 1
        section_sizes = ([0] +
                         extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])

        div_points = np.array(section_sizes).cumsum()

    sub_arys = []
    # sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]

        temp_array = ary[st:end]

        # sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
        sub_arys.append([i, temp_array])
        #sub_arys.update({i:temp_array})

    return sub_arys




@jit
def individual_traj_featurize(data_to_process):
    #print('Running individual traj featurize\n')
    test = 1
    #print("Data process to do is :", data_to_process)
    featurizer_type = data_to_process[0]

    if featurizer_type == 'Dihedral':
        featurizer_data = DihedralFeaturizer(types=['phi', 'psi'])
        # print('Featurizer created:\n')

    featurized_data = featurizer_data.fit_transform(data_to_process[2])

    #print('Finished individual traj featurize\n')
    return [data_to_process[1], featurized_data]

@jit
def prep_pd_frame(joined_data):
    # df = pd.DataFrame(columns=range(joined_data[0].shape[1]))
    df = pd.DataFrame(columns=range(len(joined_data[0][0])))
    for i in range(len(joined_data)):
    # for i in range(10):
        df.loc[i] = joined_data[i][0]

    return df


#@jit
def featurize_data(trajs,featurizer_type, client=None, step=40, num_of_threads=7, times_divide=10):
    print('Running Featurization on dask')

    # STEP 10 is a test
    #traj_chunks = chunks_mdtraj(trajs[::step], num_of_threads)

    to_divide = num_of_threads*times_divide


    print('Prep chunks')
    traj_chunks = chunks_mdtraj_dict(trajs[::step], to_divide )

    # pool = multiprocessing.Pool(num_of_threads)

    # range_n_clusters = list(range(1, 11))

    function_arguments_to_call = [[featurizer_type, traj_chunk[0], traj_chunk[1]] for
                                  traj_chunk in traj_chunks]

    # function_arguments_to_call = [[working_folder_path, centroid_data, centoid_data_index,
    #                                self.correct_topology, False, 7] for
    #                               centoid_data_index in centroid_data]
    #
    # test = 1
    #
    #results = pool.starmap(featurizer_data.fit_transform, function_arguments_to_call)

    # One core test works well
    # test = individual_traj_featurize(function_arguments_to_call[0])
    # testus=1

    # Dask Run
    print('Dask Run start ')
    #function_arguments_to_call = [[i,i+10, i*200] for i in range(100)]
    import pickle

    # virtual sites were causing issues so update at least mdtraj 1.9.2

    #pickle.dump( function_arguments_to_call, open( "save.p", "wb" ) )

    big_future = client.scatter(function_arguments_to_call)
    #
    # #test = client.recreate_error_locally(individual_traj_featurize, big_future)
    #
    futures_to_run = client.map(individual_traj_featurize, big_future, pure=False,retries=2)


    #final_result = result.result()
    print('Start of gathering data-->>>')

    # data = []
    # for curr_result in result:
    #     temp = curr_result.result()
    #     data.append(temp)
    # print(data)
    full_data = client.gather(futures_to_run)

    print('Finished gathering data-->>>')


    print('Joining data')

    #df = pd.DataFrame()
    joined_data = []
    for data in full_data:
        orig_data = data[1]
        for i in orig_data:
            joined_data.append(i.tolist()[0])

    df = pd.DataFrame(joined_data)

    featurized_data = df

        # DASK RUN CYCLE
    # dask_jobs = []
    # for data_to_run, num_run in zip(function_arguments_to_call, range(to_divide)):
    #     #future = client.scatter(data_to_run)
    #     run_job = client.submit(individual_traj_featurize, data_to_run,
    #                             key='key_run_{0}'.format(num_run),
    #                             pure=False, retries=2)
    #     dask_jobs.append(run_job)


    test = 1

    # results = pool.starmap(self.parallel_run, function_arguments_to_call)
    #
    # test = 1
    # # d = [x for x in results]
    # # d = [list(x.keys())[0] for x in results]
    #
    # # d = {key: value for (key, value) in iterable}
    # # d = {list(x.keys())[0]: x[list(x.keys())[0]] for x in results}
    # indexes = [x[0] for x in results]

    test = 1
    print('Finished Featurization on dask')
    return featurized_data