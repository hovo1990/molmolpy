# -*- coding: utf-8 -*-


# !/usr/bin/env python
#
# @file    __init__.py
# @brief   G_MMPBSA DASK PROJECT
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
import itertools

import time

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
import multiprocessing

import mdtraj as md

from molmolpy.utils.cluster_quality import *
from molmolpy.utils import folder_utils

import json

from molmolpy.utils import helper as hlp

# matplotlib.style.use('ggplot')
sns.set(style="darkgrid")

low_seed = 1
high_seed = 999999999

mgltools_utilities = '/home/john1990/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'


class GMMPBSAObject(object):
    """
    Usage example


        >>> EPI_folder = '/media/Work/MEGA/Programming/StressHormones/dock_EPI'

        >>> EPI_samples = '/media/Work/MEGA/Programming/StressHormones/'
        >>>
        >>>
        >>> receptor_file = EPI_folder + os.sep + 'centroid_model_clust2.pdbqt'
        >>> ligand_file = EPI_folder + os.sep + 'EPI.pdbqt'
        >>> molname = 'EPI'
        >>> receptor_name = 'LasR'
        >>> run_type = 'vina_sample'
        >>>
        >>>
        >>>
        >>> receptor_file = EPI_folder + os.sep + 'centroid.pdb'
        >>> ligand_file = EPI_folder + os.sep + 'EPI.pdb'
        >>> molname = 'EPI'
        >>> receptor_name = 'LasR'
        >>>
        >>>
        >>> EPI_uber_dock = uber_docker.UberDockerObject(receptor_file, ligand_file, '.', molname=molname, receptor_name=receptor_name)
        >>>
        >>>
        >>> EPI_uber_dock.prepare_uber_dock_protocol()
        >>> EPI_uber_dock.run_uber_dock_protocol()

    Use together

        >>> self.prepare_uber_dock_protocol() for preparation
        >>> self.run_uber_dock_protocol()

    or seperately

        >>> EPI_uber_dock.calculate_max_radius_from_com()
        >>> EPI_uber_dock.calculate_cube_edges()
        >>> EPI_uber_dock.calculate_box_edges_from_com()
        >>>
        >>>
        >>> EPI_uber_dock.prepare_uber_docker()
        >>>
        >>>
        >>> #This is for rDock, and it works so comment this part for a while
        >>> EPI_uber_dock.prepare_rdock_settings()
        >>> EPI_uber_dock.generate_rdock_cavity()
        >>> # Prepare and run Dock programs
        >>> EPI_uber_dock.prep_rDock_dock_run_commands()
        >>> EPI_uber_dock.run_rDock_simulation(parallel=True,  waitTime=15)
        >>>
        >>> #This is for FlexAid
        >>> EPI_uber_dock.prepare_flexaid_settings()
        >>> EPI_uber_dock.process_flexaid_ligand()
        >>> EPI_uber_dock.get_flexaid_clefts()
        >>> EPI_uber_dock.flexaid_generate_ga_dat_parameters()
        >>> EPI_uber_dock.flexaid_generate_config_input()
        >>> EPI_uber_dock.prep_FlexAid_dock_run_commands()
        >>> EPI_uber_dock.run_FlexAid_simulation(parallel=True,  waitTime=15)
        >>>
        >>>
        >>> # This is for Autodock vina
        >>> EPI_uber_dock.set_up_Vina_Box()
        >>> EPI_uber_dock.prepare_Vina_run()
        >>> EPI_uber_dock.prepVinaSim_uberDock()
        >>> EPI_uber_dock.runVinaSim_uber()



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


    >>> LasR_MOR_mmpbsa_calc = g_mmpbsa_dask.GMMPBSAObject(traj, topol_file, tpr_file, mdp_file, index_file, first_index, second_index, molname, receptor_name)
    >>>
    >>>
    >>>
    >>> LasR_MOR_mmpbsa_calc.prepare_g_mmpbsa_dask_protocol(client)
    >>>
    >>>
    >>> LasR_MOR_mmpbsa_calc.prepare_for_dask_cluster(parallel=True)
    >>> #
    >>> # LasR_MOR_mmpbsa_calc.run_dask_docking(client)


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

    def __init__(self,
                 traj, topol, tpr_file, mdp_file, index_file, first_index, second_index,

                 molname='Unknown',
                 receptor_name='Unknown',
                 folder_path='.',
                 job_name = 'Unknown',
                 load_state_file=None):

        self.load_state_file = load_state_file

        if load_state_file is not None:
            self.load_state_data_json(self.load_state_file)
        else:
            print('G_MMPBSA Object has been created')

            self.trajectory_file = traj
            self.topology_file = topol

            self.tpr_file = tpr_file
            self.mdp_file = mdp_file
            self.index_file = index_file

            self.first_index = first_index
            self.second_index = second_index

            self.prep_g_mmpbsa_run = False

            self.folder_exists = False

            # Running vina,whether it's for exhaustiveness or traditional run
            self.folder_path = folder_path

            self.command_run_list = []
            self.command_samples_run_list = []

            self.molecule_name = molname
            self.ligand_name = molname
            self.receptor_name = receptor_name

            self.run_type = 'g_mmpbsa'

            self.state_data = {}

            self.state_data_samples = {}



            self.g_mmpbsa_run_finished = False

            self.g_mmpbsa_sim_states = {'simStates': {}}

            self.objects_loaded = False

            self.g_mmpbsa_prepared = False
            # This part needs clarification
            self.prep_mdtraj_object()

            # original data before transformation

            # Add receptor name

    def set_mgltools_path(self, path):
        print('MGLTools path is set to ', path)
        self.mgltools_utilities = path

    def set_flexaid_path(self, path):
        print('FlexAid path is set to ', path)
        self.flexaid_path = path

    def set_ledock_path(self, path):
        print('LeDock path is set to ', path)
        self.ledock_path = path

    def prep_mdtraj_object(self):
        '''
        Prepare receptor mdtraj object

        get mdtraj topology and save as pandas dataframe

        Calculate pdb receptor center of mass


        :return:
        '''
        self.trajectory_mdtraj = md.load_xtc(self.trajectory_file, top=self.topology_file)

        self.trajectory_mdtraj_topology = self.trajectory_mdtraj.topology
        self.trajectory_mdtraj_topology_dataframe = self.trajectory_mdtraj.topology.to_dataframe()

        self.objects_loaded = True

    def get_uber_g_mmpbsa_run_folder_name(self):
        curr_folder = os.getcwd()
        return curr_folder + os.sep + self.run_folder_name



    def prepare_g_mmpbsa_dask_protocol(self, dask_client=None,
                                       prep_g_mmpbsa=True):
        '''
        prepare dask tasks for g_mmpbsa
        :return:
        '''

        self.prepare_g_mmpbsa()

        test = 1


        curr_client = dask_client

        # Testing Phase
        total_free_cores = 16


        # Production
        # worker_status = run_dask_tools.get_dask_worker_status(curr_client)
        #
        # get_worker_free = run_dask_tools.check_free_resources(worker_status)
        #
        #
        # test = 1
        #
        # total_free_cores = 0
        #
        # for worker in get_worker_free:
        #     preped = get_worker_free[worker]['preped']
        #     total_free_cores += preped['freeCores']





        if prep_g_mmpbsa is False:
            print('prep gmmpbsa ', prep_g_mmpbsa)
            return 'Do not prepare run files'

        if self.g_mmpbsa_prepared is True:
            print('Do not prep files')
            return 'Do not prep files'


        traj_len = len(self.trajectory_mdtraj)

        import math
        # Free core approach
        div_traj = math.ceil(traj_len/total_free_cores)
        # select_indexes = list(range(total_free_cores))

        # Maximum parallel
        #div_traj = math.trunc(traj_len/total_free_cores)
        select_frames = list(range(0,traj_len,div_traj))
        select_indexes = list(range(len(select_frames)))

        folder_to_save = self.g_mmpbsa_folder

        temp_mdtraj = []
        temp_mdtraj_indexes = []
        file_save_list = []
        abs_file_save_list = []


        simStates = {'simStates':{}}

        for i,traj in zip(select_indexes,select_frames):
            temp_state = {str(i):{}}

            temp_traj = self.trajectory_mdtraj[traj:traj+div_traj]
            temp_mdtraj.append(temp_traj)
            temp_mdtraj_indexes.append(i)
            file_save = 'traj_part{0}.xtc'.format(i)
            abs_file_save = folder_to_save + os.sep + file_save

            file_save_list.append(file_save)
            abs_file_save_list.append(abs_file_save)


            temp_state[str(i)].update({'runFinished':False,
                                       'index':i,
                                       'absFolder':folder_to_save,
                                       'fileSave':file_save,
                                       'absFileSave':abs_file_save,
                                       'firstIndex':self.first_index,
                                       'secondIndex':self.second_index,
                                       'indexFile':self.index_file,
                                       'mdpFile':self.mdp_file,
                                       'tprFile':self.tpr_file})


            energy_mm = 'energy_MM_{0}.xvg'.format(i)
            polar = 'polar_{0}.xvg'.format(i)
            apolar = 'apolar_{0}.xvg'.format(i)
            contrib_mm = 'contrib_MM_{0}.dat'.format(i)
            contrib_pol = 'contrib_pol_{0}.dat'.format(i)
            contrib_apol = 'contrib_apol_{0}.dat'.format(i)

            temp_state[str(i)].update({'energyMM':energy_mm,
                                       'polar':polar,
                                       'apolar':apolar,
                                       'contrib_MM':contrib_mm,
                                       'contrib_pol':contrib_pol,
                                       'contrib_apol':contrib_apol})


            temp_traj.save(abs_file_save)

            temp_state[str(i)].update({'fileSaved': True
                                       })

            simStates['simStates'].update(temp_state)


        self.mdtraj_frames = select_frames
        self.mdtraj_sliced = temp_mdtraj
        self.mdtraj_parts = temp_mdtraj_indexes

        self.file_save_list = file_save_list
        self.abs_file_save_list = abs_file_save_list

        self.simStates = simStates
        test = 1

        self.g_mmpbsa_prepared = True

        self.state_data['energySoftware']['g_mmpbsa'].update({'frames': self.mdtraj_frames})
        self.state_data['energySoftware']['g_mmpbsa'].update({'prepare': self.g_mmpbsa_prepared})
        self.state_data['energySoftware']['g_mmpbsa'].update({'parts': self.mdtraj_parts})
        self.state_data['energySoftware']['g_mmpbsa'].update({'fileList': self.file_save_list})
        self.state_data['energySoftware']['g_mmpbsa'].update({'absFileList': self.abs_file_save_list})
        self.state_data['energySoftware']['g_mmpbsa'].update(self.simStates)


        self.state_data['energySoftware']['g_mmpbsa'].update({'firstIndex': self.first_index})
        self.state_data['energySoftware']['g_mmpbsa'].update({'secondIndex': self.second_index})
        self.state_data['energySoftware']['g_mmpbsa'].update({'indexFile': self.index_file})
        self.state_data['energySoftware']['g_mmpbsa'].update({'mdpFile': self.mdp_file})
        self.state_data['energySoftware']['g_mmpbsa'].update({'tprFile': self.tpr_file})


        self.save_state_data_json()
        test = 1
        #self.g_mmpbsa_sim_states = self.state_data['energySoftware']['g_mmpbsa']['simStates']
        #self.ledock_samples = self.state_data['energySoftware']['g_mmpbsa']['LeDockSample_list']

        # Divide trajectory to number of free cores

        # TODO article Pagadala Software for molecular docking: a review
        # This will be for leDock




        # if prep_g_mmpbsa is True:
        #     # self.prepare_uber_docker()
        #     self.prepare_ledock_settings()
        #     self.prep_LeDock_dock_run_commands()



    @hlp.timeit
    def prep_LeDock_dock_run_commands(self, num_samples=10):
        '''
        Prepare rdock run commands and save to json
        :param num_samples: test value 6
        :return:
        '''

        try:
            self.g_mmpbsa_sim_states = self.state_data['dockSoftware']['LeDock']['simStates']
            self.ledock_samples = self.state_data['dockSoftware']['LeDock']['LeDockSample_list']
            print('No need to generate  LeDock commands')
        except:
            self.state_data['dockSoftware']['LeDock'].update({'LeDockSample_list': self.ledock_samples})
            self.state_data['dockSoftware']['LeDock'].update(self.LeDock_sim_states)

            for sample_num in self.ledock_samples:
                self.prep_LeDock_dock_command(sample_num)
                print('Now continue for LeDock:D')

            self.save_state_data_json()

            test = 1

        self.prep_LeDock_run = True


    @hlp.timeit
    def prep_LeDock_dock_command(self, sample_num, pose_gen=20):
        '''
        prepare each separate rDock run command
        :param sample_num:
        :param pose_gen: default generate 20 poses

        :return:
        '''
        try:
            if self.setup_ledock_pameters is not False:
                # print("Running Vina")
                # TODO need to think about seed

                #./ ledock_linux_x86 dock. in

                command_receptor = self.ledock_path + os.sep + 'ledock_linux_x86'

                sample_data = self.ledock_input_info[str(sample_num)]

                parm_name = sample_data['ledock_parm_name']


                test = 1

                self.save_run_name = "ledock_{0}_sample_{1}".format(self.run_type, sample_num)

                random_seed = np.random.randint(low_seed, high_seed)
                command_to_run = "{0} {1}".format(command_receptor, parm_name)

                ligand_clear_dok = sample_data['ligand_clear_name'] + '.dok'
                # -spli MOR_flexaid.dok
                command_to_clean = "{0} -spli {1}".format(command_receptor, ligand_clear_dok)



                print(command_to_run)
                self.LeDock_command_run_list.append(command_to_run)
                print("Launching new Sim")

                temp_dict = {str(sample_num): {'save_run_name': self.save_run_name,
                                               'commandRun': command_to_run,
                                               'commandToClean':command_to_clean,
                                               'dokFileName':ligand_clear_dok,
                                               'runFinished': False}}

                self.LeDock_sim_states.update(temp_dict)

                self.state_data['dockSoftware']['LeDock']['simStates'].update(temp_dict)
                # try:
                #     os.system(command_to_run)
                # except KeyboardInterrupt:
                #     # quit
                #     sys.exit()
                print("LeDock command generation finished")
            else:
                print('Please setup LeDock settings')
        except Exception as e:
            print("error in runSim: ", e)
            sys.exit(0)




    @hlp.timeit
    def check_dask_jobs(self, submitted_jobs_dask, finished_jobs, finished_jobs_dict):
        import copy

        # modified_submitted_jobs_dask = copy.deepcopy(submitted_jobs_dask)
        for i, job in enumerate(submitted_jobs_dask):
            status = job.status
            if status == 'finished':
                test = 1

                # pop_item = modified_submitted_jobs_dask.pop(i)
                try:
                    if finished_jobs_dict[i] is True:
                        continue
                except Exception as error:
                    pass

                finished_jobs.append(job)
                finished_jobs_dict.update({i: True})

                results = job.result()
                test = 1

                try:
                    key = list(results.keys())[0]

                    prog = results[key]['Program']  # need [0] key

                    sample_num = results[key]['part_num']


                    if prog == 'g_mmpbsa':
                        sample_num = results[key]['part_num']

                        results_dask = results[key]['dask']

                        original_data = self.state_data['energySoftware'][prog]

                        abs_folder = self.g_mmpbsa_folder # original_data['AbsFolder']

                        out_name = abs_folder + os.sep + results_dask['out_filename']
                        out_mem = results_dask['out_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()


                        out_name = abs_folder + os.sep + results_dask['apolar_filename']
                        out_mem = results_dask['apolar_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()

                        out_name = abs_folder + os.sep + results_dask['polar_filename']
                        out_mem = results_dask['polar_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()

                        out_name = abs_folder + os.sep + results_dask['energyMM_filename']
                        out_mem = results_dask['energyMM_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()

                        out_name = abs_folder + os.sep + results_dask['contribMM_filename']
                        out_mem = results_dask['contribMM_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()

                        out_name = abs_folder + os.sep + results_dask['contrib_apol_filename']
                        out_mem = results_dask['contrib_apol_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()

                        out_name = abs_folder + os.sep + results_dask['contrib_pol_filename']
                        out_mem = results_dask['contrib_pol_mem']

                        out_file = open(out_name, 'w')
                        out_file.write(out_mem)
                        out_file.close()
                        # out_pdbqt_filename = out_pdbqt_name

                        # self.state_data['dockSoftware'][prog]['simStates'][str(sample_num )] = \
                        # results[key]

                        update_results = copy.deepcopy(results)
                        update_results[key].pop('dask', None)
                        # self.state_data['dockSoftware'][prog]['simStates'][str(sample_num )] = results[key]
                        # self.state_data['energySoftware'][prog]['simStates'][str(sample_num)] = update_results[key]
                        self.before_dask['energySoftware'][prog]['simStates'][str(sample_num)] = update_results[key]

                        # results_dask = results[key]['dask']
                    # else:
                    # self.state_data['dockSoftware'][prog]['simStates'][str(sample_num)] = results[key]

                    # if filename is None and filedata is None:
                    #     # filename = self.json_state_file
                    #     filename = self.absolute_json_state_file
                    #     filedata = self.state_data

                    self.save_state_data_json(filedata=self.before_dask, filename=self.absolute_json_state_file)

                    # allow CPU to cool down
                    # self.hold_nSec(5)

                    print('This success ---> ', i)



                except Exception as error:
                    print('error is ', error)
                    # print('i is ', i)

        print('Finished checking dask submissions ---\n')
        print('---' * 10)
        return finished_jobs, finished_jobs_dict

    # @hlp.timeit
    def run_dask_gmmpbsa(self, client=None, max_jobs_to_run=10):
        # from molmolpy.moldock import run_dask_tools
        from molmolpy.tools import run_dask_tools
        test = 1

        curr_client = client

        worker_status = run_dask_tools.get_dask_worker_status(curr_client)

        get_worker_free = run_dask_tools.check_free_resources(worker_status)

        import copy
        original_get_worker_free = copy.deepcopy(get_worker_free)


        # TEST IT WORKS
        # queue_jobs = self.run_mmpbsa_dask
        # job_test = queue_jobs[0]
        #
        # result = run_dask_tools.run_gmmpbsa_using_dask(job_test)

        test = 1

        # Local upload test
        # big_future = self.dask_prep
        # run_dask_tools.upload_g_mmpbsa_files_dask(big_future)


        #TODO
        # Scatter a lot better using scatter for big files for upload G_MMPBSA files
        # test = 1
        # tasks_upload = []
        # big_future = client.scatter(self.dask_prep, broadcast=True)
        # for worker in get_worker_free:
        #     worker_info = get_worker_free[worker]
        #     worker_address = worker_info['preped']['workerAddress']
        #
        #     retries_num = 2
        #
        #     # Upload files to all clients client.upload_file
        #     task = client.submit(run_dask_tools.upload_g_mmpbsa_files_dask,
        #                          big_future,
        #                          workers=[worker_address],
        #                          key='key_scatter_{0}'.format(worker_address),
        #                          retries=retries_num)
        #     tasks_upload.append(task)
        #     print("Starting uploading to ", worker_address)

        test = 1



        # TODO
        # This part runs the main program
        submitted_jobs = []
        submitted_jobs_dask = []
        queue_jobs = self.run_mmpbsa_dask

        job_quantity = len(queue_jobs)
        finished_jobs = []
        finished_jobs_dict = {}
        worker_status_free = None

        test = 1

        # maybe 2 async threads, one checks finished simulations, other submits jobs

        ###############################################################################################
        gmmbpsa_min_mem = 1000
        retries_num = 2


        curr_index = 0

        curr_worker = 0

        # prepare worker ids for easier switch
        worker_ids = {}
        for i, id in enumerate(get_worker_free):
            worker_ids.update({i: id})

        custom_index_curr = 3
        while len(queue_jobs) > 0:
            if curr_index == len(queue_jobs):
                curr_index = 0

            if curr_worker == len(worker_ids):
                curr_worker = 0

            print('-----------------------------------------------------------------')

            worker_status_temp = run_dask_tools.get_dask_worker_status(curr_client, custom_index=custom_index_curr)
            get_worker_free_temp = run_dask_tools.check_free_resources(worker_status_temp)
            custom_index_curr += 2
            print('----------------TEST------------------')

            curr_item = queue_jobs[curr_index]

            test = 1

            curr_worker_id = worker_ids[curr_worker]
            workstation_info_temp = get_worker_free_temp[curr_worker_id]
            workstation_preped_temp = workstation_info_temp['preped']

            workstation_address = workstation_preped_temp['workerAddress']

            # This way folder is buggy
            workstation_dir = original_get_worker_free[curr_worker_id]['preped']['workerDir']


            workstation_freemem = workstation_preped_temp['freeMemory']
            workstation_freecpu = workstation_preped_temp['freeCores']

            curr_item_prog = curr_item['Program']

            ############################################################

            # submitted_jobs_dask len less than 16
            jobs_running = len(submitted_jobs_dask) - len(finished_jobs)
            max_jobus = max_jobs_to_run


            # g_mmpbsa part
            if curr_item_prog == 'g_mmpbsa':
                if workstation_freemem > gmmbpsa_min_mem and jobs_running  <max_jobus:
                    print('Submit MMPBSA job to DASK')
                    pop_item = queue_jobs.pop(curr_index)

                    key_name = pop_item['save_run_name']

                    run_name = 'key_{0}_{1}'.format(key_name, curr_worker_id)
                    print('Cur run ', run_name)

                    if curr_index == 0:
                        curr_index = 0
                    else:
                        curr_index -= 1

                    pop_item.update({'workingDir':workstation_dir})

                    submitted_jobs.append(pop_item)




                    # MAYBE CHECK FOLDER HERE
                    #
                    #big_future = client.scatter(pop_item, workers=[workstation_address], hash=False)
                    big_future = pop_item
                    task_g_mmpbsa = client.submit(run_dask_tools.run_gmmpbsa_using_dask,
                                                big_future,
                                                workers=[workstation_address],
                                                key=run_name,
                                                retries=retries_num)

                    submitted_jobs_dask.append(task_g_mmpbsa)
                else:
                    key_name = curr_item['save_run_name']

                    run_name = 'key_{0}_{1}'.format(key_name, curr_worker_id)
                    print('Passed running ', run_name)



            # submitted_jobs_dask_temp, finished_jobs_temp = self.check_dask_jobs(submitted_jobs_dask,finished_jobs)
            finished_jobs, finished_jobs_dict = self.check_dask_jobs(submitted_jobs_dask, finished_jobs,
                                                                     finished_jobs_dict)

            test = 1

            ###################################################3
            # update index
            # print(curr_item)

            # How to save submitted jobs state

            print('-------')
            if curr_index == 0 and len(submitted_jobs_dask) == 1:
                curr_index = 0
            else:
                curr_index += 1

            curr_worker += 1
            time.sleep(10)
            test = 1

        # ###############################################################################################
        #
        # # work_address = workstation1_preped['workerAddress']
        # #
        # # # This is to run on dask server
        # #
        # # # TODO this works need to create a quiiee
        # # retries_num = 2
        # # task = client.submit(run_dask_tools.run_vina_using_dask,
        # #                      data,
        # #                      workers=[work_address],
        # #                      key='key_test',
        # #                      retries=retries_num)
        #
        # # TODO This part needs further refinement
        #
        # # break
        #
        # test = 1
        #
        print('Last Check of submitted jobs')
        while len(finished_jobs) != job_quantity:
            finished_jobs, finished_jobs_dict = self.check_dask_jobs(submitted_jobs_dask, finished_jobs,
                                                                     finished_jobs_dict)
            time.sleep(60)
            print('->' * 10)

        print('Everything is finished :))))))')
        print('---' * 10)
        print('\n')


    def prepare_for_dask_cluster(self, LeDock=2, rDock=2, FlexAid=2, Vina=2, parallel=False):
        '''
        run uber dock protocol for LeDock, rDock,FlexAid, Vina
        :return:
        '''
        current_pid = multiprocessing.current_process().pid

        print("Main Process with PID:{}".format(current_pid))

        # free_threads_for_Vina = num_threads - LeDock-rDock-FlexAid

        run_g_mmpbsa = []

        run_mmpbsa_queue = []

        # Prepare outputs

        import copy
        self.before_dask = copy.deepcopy(self.state_data)

        ################################################################################
        if self.g_mmpbsa_prepared is True:
            full_g_mmpbsa_data = self.state_data['energySoftware']['g_mmpbsa']
            test = 1

            tpr_abs= full_g_mmpbsa_data['tprFile']
            tpr_file = open(tpr_abs, 'rb')
            tpr_mem = tpr_file.read()
            tpr_filename = tpr_abs.split(os.sep)[-1]
            #
            mdp_abs= full_g_mmpbsa_data['mdpFile']
            mdp_file = open(mdp_abs, 'r')
            mdp_mem = mdp_file.read()
            mdp_filename = mdp_abs.split(os.sep)[-1]

            index_abs= full_g_mmpbsa_data['indexFile']
            index_file = open(index_abs, 'r')
            index_mem = index_file.read()
            index_filename = index_abs.split(os.sep)[-1]

            # data_pre = self.state_data['energySoftware']['g_mmpbsa']
            # data_pre.update({'dask': {}})

            data_pre = {}

            data_pre.update({'tprName':tpr_filename, 'tprMem':tpr_mem})
            data_pre.update({'mdpName':mdp_filename, 'mdpMem':mdp_mem})
            data_pre.update({'indexName':index_filename, 'indexMem':index_mem})

            self.dask_prep = data_pre

            for part_num in full_g_mmpbsa_data['parts']:
                # self.run_FlexAid_sim(FlexAid_sample_num, waitTime=waitTime)
                data = self.state_data['energySoftware']['g_mmpbsa']['simStates'][str(part_num)]

                save_run_name = "g_mmpbsa_part_{0}".format(part_num)


                data.update({'Program': 'g_mmpbsa'})
                data.update({'part_num': part_num})
                data.update({'save_run_name':  save_run_name})

                data.update({'dask': {}})

                traj_abs = data['absFileSave']
                traj_file = open(traj_abs, 'rb')
                traj_mem = traj_file.read()
                traj_filename = data['fileSave']

                data['dask'].update({'tprName': tpr_filename})
                data['dask'].update({'mdpName': mdp_filename})
                data['dask'].update({'indexName': index_filename})

                data['dask'].update({'trajMem':traj_mem, 'trajName':traj_filename})

                data['dask'].update({'tprName': tpr_filename, 'tprMem': tpr_mem})
                data['dask'].update({'mdpName': mdp_filename, 'mdpMem': mdp_mem})
                data['dask'].update({'indexName': index_filename, 'indexMem': index_mem})


                test = 1
                # data['dask'].update({'cavFile':cav_file_mem })

                # self.state_data['dockSoftware']['LeDock']['simStates'][str(LeDock_sample_num)] = data
                test = 1
                run_g_mmpbsa.append(data)
        #         # result = run_dock_tools.run_LeDock_sim_parallel(LeDock_sample_num, data)
        #         # test = 1
        #
        # test = 1
        ###################################################################################################

        test = 1
        ####################################################################################################

        self.run_mmpbsa_dask = run_g_mmpbsa

        curr_LeDock = 0

        # very slow
        # while len(run_docking_queue) != 40:
        #     run_docking_queue += run_docking_LeDock[curr_LeDock:curr_LeDock + LeDock]
        #     curr_LeDock += LeDock
        #
        #     test = 1
        #     run_docking_queue += run_docking_rDock[curr_rDock:curr_rDock + rDock]
        #     curr_rDock += rDock
        #
        #     run_docking_queue += run_docking_FlexAid[curr_FlexAid:curr_FlexAid + FlexAid]
        #
        #     curr_FlexAid += FlexAid
        #
        #     run_docking_queue += run_docking_Vina[curr_Vina:curr_Vina + Vina]
        #     curr_Vina += Vina
        #
        #     test = 1
        #
        test = 1

        run_mmpbsa_queue = run_g_mmpbsa
        # run_docking_queue = run_docking_LeDock + run_docking_FlexAid + run_docking_Vina

        final_queue_job = []
        # Need to select those that are not finished
        for pre_job in run_mmpbsa_queue:
            # print(pre_job)
            if pre_job['runFinished'] is False:
                final_queue_job.append(pre_job)
            test = 1

        self.run_mmpbsa_dask = final_queue_job

        # random.shuffle(self.run_docking_queue)

        print('Finished preparing g_mmpbsa jobs')



    # TODO should I add json saving of information or not?
    def load_state_data_json(self, filename):
        '''

        :param filename: load json state data
        :return:
        '''
        # self.absolute_path = os.path.abspath(filename)
        self.load_state_called = True

        print(os.path.abspath(__file__))
        self.state_data = json.load(open(filename, "r"))

        # os.chdir('HSL_exhaustiveness')

        self.trajectory_file = self.state_data['trajectoryFile']
        self.mdp_file = self.state_data['mdpFile']
        self.tpr_file = self.state_data['tprFile']
        self.index_file = self.state_data['indexFile']






        self.folder_path = self.state_data['folderPath']
        self.run_type = self.state_data['runType']

        self.molecule_name = self.state_data['molName']
        self.receptor_name = self.state_data['receptorName']

        # TODO test
        self.sim_folder_run = self.state_data['simRunFolder']  # .split('/')[-1]
        self.directories = self.state_data['directory']
        self.folder_exists = self.state_data['folderCreated']

        self.absolute_json_state_file = self.state_data['absoluteJsonStates']
        self.g_mmpbsa_folder =  self.state_data['RunFolder']
        self.json_state_file = self.state_data['jsonStates']

        test = 1

        # self.rdock_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + 'rDock'
        # self.rdock_absolute_folder_name = self.uber_dock_folder + os.sep + self.rdock_folder_name

        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        # self.directories = folder_utils.find_folder_in_path(self.uber_dock_folder, self.rdock_folder_name)
        # print('TADA ', self.directories)

        test = 1
        # This will hold information about run states
        # self.uber_dock_folder = self.get_uber_dock_run_folder_name()








        ########################################################################################
        # # LeDock settings part
        #
        # self.ledock_data = self.state_data['dockSoftware']['LeDock']
        # test = 1
        #
        # # Try to load initial  LeDock
        try:
            self.mdtraj_frames = self.state_data['energySoftware']['g_mmpbsa']['frames']

            self.mdtraj_parts = self.state_data['energySoftware']['g_mmpbsa']['parts']

            self.file_save_list = self.state_data['energySoftware']['g_mmpbsa']['fileList']
            self.abs_file_save_list = self.state_data['energySoftware']['g_mmpbsa']['absFileList']

            self.simStates = self.state_data['energySoftware']['g_mmpbsa']['simStates']
            test = 1

            self.g_mmpbsa_prepared = self.state_data['energySoftware']['g_mmpbsa']['prepare']

            # self.state_data['energySoftware']['g_mmpbsa'].update({'frames': self.mdtraj_frames})
            # self.state_data['energySoftware']['g_mmpbsa'].update({'prepare': self.g_mmpbsa_prepared})
            # self.state_data['energySoftware']['g_mmpbsa'].update({'parts': self.mdtraj_parts})
            # self.state_data['energySoftware']['g_mmpbsa'].update({'fileList': self.file_save_list})
            # self.state_data['energySoftware']['g_mmpbsa'].update({'absFileList': self.abs_file_save_list})
            # self.state_data['energySoftware']['g_mmpbsa'].update(self.simStates)

        except:
            print('G_mmpbsa is empty verify yolo')
        #
        # test = 1
        #
        # try:
        #     self.setup_ledock_pameters = self.ledock_data['setup_LeDock']
        #     self.ledock_num_samples = self.ledock_data['num_samples']
        #     self.ledock_input_info = self.ledock_data['LeDockInputInfo']
        #     self.param_ledock_template = self.ledock_data['paramFull']
        # except:
        #     print('LeDock setting part is empty verify yolo')
        #
        # try:
        #     self.ledock_param_title = self.ledock_data['LeDock_params']['title']
        #     self.rdock_title = self.ledock_data['LeDock_params']['title']
        #
        #     self.receptor_file_ledock = self.ledock_data['LeDock_params']['receptorFile']
        #     self.ledock_rmsd = self.ledock_data['LeDock_params']['LeDockRMSD']
        #
        #     self.ledock_xmin = self.ledock_data['LeDock_params']['xmin']
        #     self.ledock_xmax = self.ledock_data['LeDock_params']['xmax']
        #     self.ledock_ymin = self.ledock_data['LeDock_params']['ymin']
        #     self.ledock_ymax = self.ledock_data['LeDock_params']['ymax']
        #     self.ledock_zmin = self.ledock_data['LeDock_params']['zmin']
        #     self.ledock_zmax = self.ledock_data['LeDock_params']['zmax']
        #
        # except:
        #     print('LeDock_params is empty verify yolo')
        #
        # try:
        #     self.LeDock_sim_states = self.state_data['dockSoftware']['LeDock']['simStates']
        #     self.ledock_samples = self.state_data['dockSoftware']['LeDock']['LeDockSample_list']
        #     print('No need to generate LeDock commands')
        #     self.prep_LeDock_run = True
        # except:
        #     print('LeDock_params simStates is empty verify yolo')
        #
        # test = 1

    def prepare_g_mmpbsa(self):
        '''
        Prepare g_mmpbsa run folder and initial json configuration

        :return:
        '''
        self.run_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + self.run_type

        self.sim_folder_run = self.folder_path + os.sep + self.run_folder_name
        # Create folder don't forget

        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.directories = folder_utils.find_folder_in_path(self.folder_path, self.run_folder_name)
        print('TADA ', self.directories)

        self.json_state_file = self.sim_folder_run + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type + '.json'

        # This will hold information about run states
        self.g_mmpbsa_folder = self.get_uber_g_mmpbsa_run_folder_name()
        self.absolute_json_state_file = self.g_mmpbsa_folder + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type + '.json'

        if len(self.directories) == 0:
            print('Creating folder for  g_mmpbsa run\n')
            print(self.sim_folder_run)
            folder_utils.create_folder(self.sim_folder_run)
            self.folder_exists = True

            programs_dict = {'energySoftware': {'g_mmpbsa': {}}}

            self.state_data.update({'trajectoryFile': self.trajectory_file,
                                    'mdpFile': self.mdp_file,
                                    'tprFile': self.tpr_file,
                                    'indexFile': self.index_file,
                                    'runFolderName': self.run_folder_name,
                                    'folderPath': self.folder_path,
                                    'jsonStates': self.json_state_file,
                                    'runType': self.run_type,
                                    'molName': self.molecule_name,
                                    'receptorName': self.receptor_name,
                                    'simRunFolder': self.sim_folder_run,
                                    'RunFolder': self.g_mmpbsa_folder,
                                    'absoluteJsonStates': self.absolute_json_state_file,
                                    'directory': self.directories,
                                    'folderCreated': self.folder_exists,
                                    'simStates': {}})

            self.state_data.update(programs_dict)

            # self.prepVinaSim_exhaust()

            self.save_state_data_json()
            self.load_state_called = False


        else:
            self.load_state_file = self.json_state_file
            self.load_state_called = True
            self.load_state_data_json(self.load_state_file)

    def prepare_ledock_settings(self):
        '''
        Prepare ultraDock folder and initial json configuration

                    >>> EPI_uber_dock.prepare_rdock_settings()

        Convert with pybel to mol2 for receptor and sd for ligand
        :return:
        '''

        # self.output_receptor_rdock = Outputfile("mol2", "{0}.mol2".format(self.receptor_name))
        # self.output_receptor_rdock.write(self.receptor_pybel)
        # self.output_receptor_rdock.close()
        #
        # self.output_ligand_rdock = Outputfile("sd", "{0}.sd".format(self.ligand_name))
        # self.output_ligand_rdock.write(self.ligand_pybel )
        # self.output_ligand_rdock.close()

        self.ledock_folder_name = self.receptor_name + '_' + self.molecule_name + '_' + 'LeDock'
        self.ledock_absolute_folder_name = self.uber_dock_folder + os.sep + self.ledock_folder_name

        test = 1
        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.ledock_directories = folder_utils.find_folder_in_path(self.uber_dock_folder, self.ledock_folder_name)
        print('TADA ', self.ledock_directories)

        test = 1
        # This will hold information about run states
        # self.uber_dock_folder = self.get_uber_dock_run_folder_name()

        if len(self.ledock_directories) == 0:
            print('Creating  rdock folder in  uberDocker folder \n')
            print(self.ledock_directories)
            folder_utils.create_folder(self.ledock_absolute_folder_name)

            test = 1

            self.receptor_ledock_pdb = "{0}.pdb".format(self.receptor_name)
            self.ligand_ledock_mol2 = "{0}.mol2".format(self.ligand_name)

            self.absolute_receptor_ledock_pdb = self.ledock_absolute_folder_name + os.sep + self.receptor_ledock_pdb
            self.absolute_ligand_ledock_mol2 = self.ledock_absolute_folder_name + os.sep + self.ligand_ledock_mol2

            self.receptor_pybel.write("pdb", self.absolute_receptor_ledock_pdb, overwrite=True)
            self.ligand_pybel.write("mol2", self.absolute_ligand_ledock_mol2, overwrite=True)

            self.ledock_folder_exists = True
            test = 1

            # TODO enter ledock folder and process structure for docking using lepro
            # ./lepro_linux_x86 LasR_flexaid.pdb

            os.chdir(self.ledock_absolute_folder_name)

            command_receptor = self.ledock_path + os.sep + 'lepro_linux_x86' + ' {0} '.format(self.receptor_ledock_pdb)
            os.system(command_receptor)

            self.lepro_pdb_file = 'pro.pdb'
            # Need to check whteter lepro ran fine
            print('Updated receptor with LePro\n')

            os.chdir(self.uber_dock_folder)

            self.state_data['dockSoftware']['LeDock'].update(
                {'receptor_pdb': self.receptor_ledock_pdb,
                 'ligand_mol2': self.ligand_ledock_mol2,
                 'lepro_pdb': self.lepro_pdb_file,
                 'lepro_abs_pdb': self.ledock_absolute_folder_name + os.sep + self.lepro_pdb_file,

                 'abs_receptor_pdb': self.absolute_receptor_ledock_pdb,
                 'abs_ligand_mol2': self.absolute_ligand_ledock_mol2,
                 'LeDockFolderStatus': self.ledock_folder_exists,
                 'LeDockAbsFolder': self.ledock_absolute_folder_name,
                 'LeDockFolderName': self.ledock_folder_name})

            self.save_state_data_json()
            self.load_state_called = False

            self.ledock_title = self.receptor_name + '_' + self.ligand_name + '_LeDock Parameter file'

            self.ledock_rmsd = 0.5

            self.set_up_ledock_dock_blind_parameters(title=self.ledock_title,
                                                     receptor_file=self.lepro_pdb_file,
                                                     ledock_rmsd=self.ledock_rmsd,
                                                     x_center=self.x_center,
                                                     y_center=self.y_center,
                                                     z_center=self.z_center)
        else:
            print('state has beeen loaded \n')

    ##############################################################################

    def flexaid_generate_ga_dat_parameters(self):
        '''
        Generate GA dat parameters for flexaid docking
        :return:
        '''
        self.flexaid_ga_dat_param_template = '''# Number of chromosomes (number individuals in the population)
# Integer in interval [1-N]
NUMCHROM 500

# Number of generations
# Integer in interval [1-N]
NUMGENER 500

# Use Adaptive Genetic-Algorithm
# Value of 0 or 1
ADAPTVGA 1

# Adaptive crossover and mutation probabilities
# Floats in interval [0.0,1.0]
ADAPTKCO 0.95 0.10 0.95 0.10

# Constant crossover probability
# Float in interval [0.0,1.0]
# Only considered when ADAPTVGA is 0
CROSRATE 0.90

# Constant mutation probability
# Float in interval [0.0,1.0]
# Only considered when ADAPTVGA is 0
MUTARATE 0.10

# Crossover operator
# Intragenic crossovers are possible
INTRAGEN

# Specifies that the initial population is generated randomly
POPINIMT RANDOM

# Fitness function
# Value in [LINEAR,PSHARE]
FITMODEL PSHARE

# Parameters of the shared fitness function
# Floats in interval [0.0,1000.0]
SHAREALF 4.0
SHAREPEK 5.0
SHARESCL 10.0

# Reproduction model
# Values in [BOOM,STEADY]
REPMODEL BOOM

# Fraction of population to create
# Only considered when REPMODEL is BOOM
BOOMFRAC 1.0

# Number of new individuals to generate at each generation
# Only considered when REPMODEL is STEADY
# Integer in interval [1,N-1] where N is NUMCHROM
STEADNUM 950

# Number of TOP individuals to print in console
# Integer in interval [1,N] where N is NUMCHROM
PRINTCHR 10
'''

        self.generate_ga_dat_pameters = True

        self.generate_ga_dat = 'ga_inp_' + self.receptor_name + '-' + self.ligand_name + '.dat'
        self.generate_ga_dat_name_abs = self.flexaid_absolute_input_folder + os.sep + self.generate_ga_dat

        self.generate_ga_dat_object_file = open(self.generate_ga_dat_name_abs, 'w')
        self.generate_ga_dat_object_file.write(self.flexaid_ga_dat_param_template)
        self.generate_ga_dat_object_file.close()

        self.state_data['dockSoftware']['FlexAid'].update({'GA_params': {}})

        self.state_data['dockSoftware']['FlexAid']['GA_params'].update(
            {'generateGA_param': self.generate_ga_dat_pameters,
             'GA_DataName': self.generate_ga_dat,
             'GA_DATA_Abs': self.generate_ga_dat_name_abs,
             'GA_ParamFull': self.flexaid_ga_dat_param_template})

        # self.state_data_samples = self.state_data.copy()

        self.save_state_data_json()

        # TODO this part needs to be thought out

    ####################################################################################################################

    def flexaid_generate_ga_dat_parameters_dask(self):
        '''
        Generate GA dat parameters for flexaid docking
        :return:
        '''
        self.flexaid_ga_dat_param_template = '''# Number of chromosomes (number individuals in the population)
    # Integer in interval [1-N]
    NUMCHROM 500

    # Number of generations
    # Integer in interval [1-N]
    NUMGENER 500

    # Use Adaptive Genetic-Algorithm
    # Value of 0 or 1
    ADAPTVGA 1

    # Adaptive crossover and mutation probabilities
    # Floats in interval [0.0,1.0]
    ADAPTKCO 0.95 0.10 0.95 0.10

    # Constant crossover probability
    # Float in interval [0.0,1.0]
    # Only considered when ADAPTVGA is 0
    CROSRATE 0.90

    # Constant mutation probability
    # Float in interval [0.0,1.0]
    # Only considered when ADAPTVGA is 0
    MUTARATE 0.10

    # Crossover operator
    # Intragenic crossovers are possible
    INTRAGEN

    # Specifies that the initial population is generated randomly
    POPINIMT RANDOM

    # Fitness function
    # Value in [LINEAR,PSHARE]
    FITMODEL PSHARE

    # Parameters of the shared fitness function
    # Floats in interval [0.0,1000.0]
    SHAREALF 4.0
    SHAREPEK 5.0
    SHARESCL 10.0

    # Reproduction model
    # Values in [BOOM,STEADY]
    REPMODEL BOOM

    # Fraction of population to create
    # Only considered when REPMODEL is BOOM
    BOOMFRAC 1.0

    # Number of new individuals to generate at each generation
    # Only considered when REPMODEL is STEADY
    # Integer in interval [1,N-1] where N is NUMCHROM
    STEADNUM 950

    # Number of TOP individuals to print in console
    # Integer in interval [1,N] where N is NUMCHROM
    PRINTCHR 10
    '''

        generate_ga_dat = 'ga_inp_' + self.receptor_name + '-' + self.ligand_name + '.dat'
        generate_ga_dat_name_abs = self.flexaid_absolute_input_folder + os.sep + self.generate_ga_dat

        return [generate_ga_dat, ]

        # self.generate_ga_dat_object_file = open(self.generate_ga_dat_name_abs, 'w')
        # self.generate_ga_dat_object_file.write(self.flexaid_ga_dat_param_template)
        # self.generate_ga_dat_object_file.close()
        #
        # self.state_data['dockSoftware']['FlexAid'].update({'GA_params': {}})
        #
        # self.state_data['dockSoftware']['FlexAid']['GA_params'].update(
        #     {'generateGA_param': self.generate_ga_dat_pameters,
        #      'GA_DataName': self.generate_ga_dat,
        #      'GA_DATA_Abs': self.generate_ga_dat_name_abs,
        #      'GA_ParamFull': self.flexaid_ga_dat_param_template})

        # self.state_data_samples = self.state_data.copy()

    ##############################################################################################

    def flexaid_generate_config_input_dask(self):
        '''
        Generate flexaid config input file

        Flexaid is very strict about spaces
        :return:
        '''
        flexaid_config_input_template = '''# Optimization method (genetic-algorithms)
METOPT GA

# The variation in degrees for the anchor angle of the ligand
# Float in interval [1.0-30.0]
VARANG 5.0

# The variation in degrees for the anchor dihedral of the ligand
# Float in interval [1.0-30.0]
VARDIH 5.0

# The variation in degrees for flexible dihedrals of the ligand
# Float in interval [1.0-30.0]
VARFLX 10.0

# Use Vcontacts in the calculations of surfaces in contact
COMPLF VCT

# Do not consider intramolecular interactions
NOINTR

# Side-chain rotamer acceptance threshold
# Float in interval [0.0-1.0]
DEECLA 0.8

# Use instances of side-chain conformers rather than using the Penultimate Rotamer Library
#ROTOBS

# Defines the grid spacing of the binding-site
# Float in interval [0.1,1.0]
SPACER 0.375

# Exclude hetero groups in the target (water,metal,modified amino acids,cofactors,ligands)
# To exclude these groups, uncomment the next line
#EXCHET

# Include water molecules in the target (always removed by default)
# Only considered if EXCHET is disabled
# To include water molecules, uncomment the next line
#INCHOH

# Permeability allowed between atoms
# Float in interval [0.0,1.0] from fully permeable to no permeability
PERMEA 0.9

# Permeability for side-chain rotamer acceptance
# Float in interval [0.0,1.0] from fully permeable to no permeability
ROTPER 0.8

# Solvent term penalty
# When the value is 0.0 the solvent interactions are derived from the interaction matrix
# Float in interval [-200.0,200.0]
SLVPEN 0.0

# Use Vcontacts indexing
VINDEX

# Vcontacts plane definition
# Value in [B,R,X] for Bissecting, Radical and Extended radical plane
# See McConkey et al. (2002) Bioinformatics. 18(10); 1365-1373
VCTPLA R

# Use normalized surfaces in contacts
NORMAR

# Define the RMSD cutoff between clusters
# Float in interval [0.5,3.0]
CLRMSD 2.0

# Number of results/docking poses to output
MAXRES 20

# Only output scored atoms in the final results
# Comment the next line if you wish to obtain the whole complex
SCOOUT

# Only calculate the CF for ligand atoms despite including flexible side-chains
#SCOLIG

# Ends reading of CONFIG file
ENDINP
'''
        final_str = ''''''

        # Specify the processed target file to use
        pdbnam = 'PDBNAM ' + '{0}\n\n'.format(
            self.receptor_flexaid_mol2)

        # Specify the processed ligand file to use
        # BTN.inp has the unique RESNUMC identifier LIG9999A
        inplig = 'INPLIG ' + '{0}.inp\n\n'.format(
            self.ligand_flexaid_initials)

        # Specify to use one or multiple cleft(s) as binding-site
        rgnopt_locclf = 'RNGOPT LOCCLF ' + 'global_binding_site.pdb\n\n'

        # Specify the degrees of freedom (DOF) of the processed ligand with residue number 9999 and chain A
        # Translational DOF of the ligand (-1)
        optimz1 = 'OPTIMZ 9999 {0} -1\n\n'.format(self.flexaid_res_chain)

        # Rotational DOF of the ligand (0)
        optimz2 = 'OPTIMZ 9999 {0} 0\n\n'.format(self.flexaid_res_chain)

        # Add one extra line for each flexible bond of the ligand
        # The allowable flexible bonds are listed as FLEDIH lines in Processed_files/BTN.inp
        # In our example, Biotin has 5 flexible bonds

        flexible_bonds_data = open(
            self.flexaid_absolute_processed_files_folder + os.sep + '{0}.inp'.format(self.ligand_flexaid_initials), 'r')

        flexible_bonds_data_text = flexible_bonds_data.read()
        flexible_bonds_data.close()

        flexible_bonds_data_text_list = flexible_bonds_data_text.split('\n')

        flexible_index_list_phrases = []
        flexible_index_list = []
        for i in flexible_bonds_data_text_list:
            if 'FLEDIH' in i:
                print(i)
                temp = i.split(' ')
                print(temp)
                flex_index = temp[-2]
                flexible_index_list.append(int(flex_index))
                temp_line = 'OPTIMZ {0} {1} {2}\n'.format(self.flexaid_res_number, self.flexaid_res_chain, flex_index)
                flexible_index_list_phrases.append(temp_line)

        test = 1

        final_str += pdbnam
        final_str += inplig
        final_str += rgnopt_locclf
        final_str += optimz1
        final_str += optimz2

        for y in flexible_index_list_phrases:
            final_str += y

        final_str += '\n'
        rmsdst = 'RMSDST ' + '{0}_ref.pdb\n\n'.format(
            self.ligand_flexaid_initials)

        final_str += rmsdst

        final_str += flexaid_config_input_template

        generate_config_input_file = 'CONFIG_' + self.receptor_name + '-' + self.ligand_name + '.inp'

        return generate_config_input_file, final_str
        # self.state_data['dockSoftware']['FlexAid'].update({'GA_params': {}})
        #
        # self.state_data['dockSoftware']['FlexAid']['GA_params'].update(
        #     {'generateGA_param': self.generate_ga_dat_pameters,
        #      'GA_DataName': self.generate_ga_dat,
        #      'GA_DATA_Abs': self.generate_ga_dat_name_abs,
        #      'GA_ParamFull': self.flexaid_ga_dat_param_template})
        #
        # # self.state_data_samples = self.state_data.copy()
        #
        # self.save_state_data_json()

        # TODO this part needs to be thought out

    ####################################################################################################################

    def prepare_samples_collection_run(self, standard_exhaust=128,
                                       num_samples_run=100,
                                       run_type='samples_run'):

        if self.setup_box is False:
            print('Please setup simulation box')
            sys.exit(0)

        self.run_type_samples = run_type

        self.prep_samples_run = True

        self.samples_exhaust = standard_exhaust
        self.samples_run = list(range(1, num_samples_run + 1))

        self.run_folder_name_samples = self.receptor_name + '_' + self.molecule_name + '_' + self.run_type_samples
        self.sim_folder_run_samples = self.folder_path + os.sep + self.run_folder_name_samples
        # Create folder don't forget

        # Exhaustiveness for all samples

        # self.directories = self.find_sample_folders(self.folder_path, dir_name=self.run_type)
        self.directories_samples = folder_utils.find_folder_in_path(self.folder_path, self.run_folder_name_samples)
        print('TADA ', self.directories_samples)

        self.json_samples_state_file = self.sim_folder_run_samples + os.sep + self.receptor_name + '_' + self.molecule_name + '_' + self.run_type_samples + '.json'

        # This will hold information about run states

        if len(self.directories_samples) == 0:
            print('Creating folder for vina samples run\n')
            print('Vina run type: {0}'.format(self.run_type_samples))
            print(self.sim_folder_run_samples)
            folder_utils.create_folder(self.sim_folder_run_samples)
            self.folder_exists_samples = True

            self.state_data_samples.update({'receptorFile': self.receptor_file,
                                            'ligandFile': self.ligand_file,
                                            'exhaustivenessList': self.exhaustiveness,
                                            'samples_exhaust': self.samples_exhaust,
                                            'samplesList': self.samples_run,
                                            'folderPath': self.folder_path,
                                            'runType': self.run_type_samples,
                                            'molName': self.molecule_name,
                                            'receptorName': self.receptor_name,
                                            'simRunFolder': self.sim_folder_run_samples,
                                            'directory': self.directories_samples,
                                            'setup': self.setup_box,
                                            'folderCreated': self.folder_exists_samples,
                                            'simStates': {}})

            self.prepVinaSim_samples()
            self.save_state_data_json(filedata=self.state_data_samples, filename=self.json_samples_state_file)

            self.load_state_called_samples = False

            self.prep_sample_run = True

        else:
            self.load_state_file_samples = self.json_samples_state_file
            self.load_state_called_samples = True
            self.load_samples_state_data_json(self.load_state_file_samples)
            self.prep_sample_run = True

    def get_exhaust_run_folder_name(self):
        curr_folder = os.getcwd()
        return curr_folder + os.sep + self.run_folder_name

    def get_samples_run_folder_name(self):
        curr_folder = os.getcwd()
        print("Yippie yi kay", curr_folder)
        return curr_folder + os.sep + self.run_folder_name_samples

    def save_state_data_json(self, filedata=None, filename=None):
        '''
        :param filename: Saves state file
        :return:
        '''
        # import json
        # with open(filename, 'w') as outfile:
        #     json.dump(self.cluster_models, outfile)
        # pickle.dump(self.cluster_models, open(filename, "wb"))
        # TODO create folder for run saving state run

        # filename = self.sim_folder_run + os.sep + self.receptor_name + '_' + self.molecule_name + '.json'
        if filename is None and filedata is None:
            # filename = self.json_state_file
            filename = self.absolute_json_state_file
            filedata = self.state_data
        # elif filedata is not None:
        #     filedata = filedata
        #     filename = self.absolute_json_state_file
        else:
            filedata = filedata
            filename = filename
        json.dump(filedata, open(filename, "w"), sort_keys=True, indent=4)

    # TODO should I add json saving of information or not?
    def load_samples_state_data_json(self, filename):
        '''

        :param filename: load json state data
        :return:
        '''
        # self.absolute_path = os.path.abspath(filename)
        self.load_state_called_samples = True

        print(os.path.abspath(__file__))
        self.state_data_samples = json.load(open(filename, "r"))

        # os.chdir('HSL_exhaustiveness')

        self.receptor_file = self.state_data_samples['receptorFile']
        self.ligand_file = self.state_data_samples['ligandFile']
        self.exhaustiveness = self.state_data_samples['exhaustivenessList']
        self.samples_run = self.state_data_samples['samplesList']
        self.folder_path = self.state_data_samples['folderPath']
        self.run_type = self.state_data_samples['runType']
        self.molecule_name = self.state_data_samples['molName']
        self.receptor_name = self.state_data_samples['receptorName']

        # TODO test
        self.samples_exhaust = self.state_data_samples['samples_exhaust']
        self.sim_folder_run_samples = self.state_data_samples['simRunFolder']  # .split('/')[-1]
        self.directories_samples = self.state_data_samples['directory']
        self.setup_box = self.state_data_samples['setup']
        self.folder_exists = self.state_data_samples['folderCreated']

        self.x_center = self.state_data_samples['boxSettings']['center_x']
        self.y_center = self.state_data_samples['boxSettings']['center_y']
        self.z_center = self.state_data_samples['boxSettings']['center_z']
        self.x_size = self.state_data_samples['boxSettings']['size_x']
        self.y_size = self.state_data_samples['boxSettings']['size_y']
        self.z_size = self.state_data_samples['boxSettings']['size_z']
        self.num_modes = self.state_data_samples['boxSettings']['numModes']

    def hold_nSec(self, n):
        for i in range(1, n + 1):
            print(i)
            time.sleep(1)  # Delay for 1 sec
        print('Ok %s secs have pass' % (n))

    @hlp.timeit
    def prepVinaSampleCommand(self, sample_num):
        # try:
        if self.setup_box is not False:
            # print("Running Vina")
            # TODO need to think about seed
            self.save_run_name = 'vina_' + self.run_type_samples + '_' + str(sample_num)
            command_to_run = "vina --receptor {0} " \
                             "--ligand {1} " \
                             "--center_x {2} " \
                             "--center_y {3} " \
                             "--center_z {4} " \
                             "--size_x {5} " \
                             "--size_y {6} " \
                             "--size_z {7} " \
                             "--exhaustiveness {8} " \
                             "--num_modes {9} " \
                             "--seed 10 " \
                             "--log {10}.txt " \
                             "--out {11}_out.pdbqt".format(self.receptor_file,
                                                           self.ligand_file,
                                                           self.x_center,
                                                           self.y_center,
                                                           self.z_center,
                                                           self.x_size,
                                                           self.y_size,
                                                           self.z_size,
                                                           self.samples_exhaust,
                                                           self.num_modes,
                                                           self.save_run_name,
                                                           self.save_run_name)
            print(command_to_run)
            self.command_samples_run_list.append(command_to_run)
            print("Launching new Sim")

            self.state_data_samples['simStates'].update({str(sample_num): {'save_run_name': self.save_run_name,
                                                                           'commandRun': command_to_run,
                                                                           'runFinished': False}})
            # try:
            #     os.system(command_to_run)
            # except KeyboardInterrupt:
            #     # quit
            #     sys.exit()
            print("Vina sample run command prep finished")
        else:
            print('Please setup vina box settings')
            # except Exception as e:
            #     print("error in Sample runSim: ", e)
            #     sys.exit(0)

    def get_molecule_name(self):
        return self.molecule_name

    def get_receptor_name(self):
        return self.receptor_name

    def set_molecule_name(self, mol_name):
        self.molecule_name = mol_name

    def set_receptor_name(self, receptor_name):
        self.receptor_name = receptor_name

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

    def find_sample_folders(self, folder_path='.', dir_name='vina_sample'):
        try:
            dir_names = []
            for dirname, dirnames, filenames in os.walk(folder_path):
                # print(dirname, '-')
                if dir_name in dirname:  #
                    # print(dir_name)
                    dir_names.append(dirname)
            # print sorted(dir_names)
            return sorted(dir_names)
        except Exception as e:
            print("Problem with finding folders : ", e)
            sys.exit(0)
