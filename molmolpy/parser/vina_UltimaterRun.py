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


import os
import shutil
import sys
import time

i = 0
commands = ['vina --config conf.txt --log res_log.txt']
print(len(commands))
print('----' * 20)


def hold_nSec(n):
    for i in range(1, n + 1):
        print(i)
        time.sleep(1)  # Delay for 1 sec
    print('Ok %s secs have pass' % (n))


def read_state_file(file):
    try:
        condition = []
        state = open(file, 'r')
        for i in state:
            temp = i.split(':')
            # print temp
            condition.append(temp)
            # if 'No' in temp[-1]:
            # print 'Yay'
        print(condition)
        return condition
    except Exception as e:
        print("error in read_state_file: ", e)
        sys.exit(0)


def check_condition(cond_file):
    print('COND_FILE is ', cond_file)
    try:
        i = 0
        print('tada ', cond_file)
        print('len ', len(cond_file))
        print('->-*' * 10)
        if 'Yes' in cond_file[0][1]:
            i += 1
        print('Okay run everything')
        return i
    except Exception as e:
        print("error in check_condition: ", e)
        sys.exit(0)


def replace_state(gas_file, index):
    try:
        import fileinput
        # import sys
        # x = fileinput.input(gas_file,inplace=1)
        # print x
        for i, line in enumerate(fileinput.input(gas_file, inplace=1)):
            # print('i is ',i)
            # print('line is ',line)
            # print('->'*10)
            if i == index:
                sys.stdout.write(line.replace(' No\n', ' Yes\n'))  # write a blank line after the 5th line
                # line = line.replace(' No\n',' Yes\n')
                # print line,
            else:
                sys.stdout.write(line)
    except Exception as e:
        print("error in replace_state: ", e)
        sys.exit(0)


def find_Local_folders(folderInit):
    try:
        dir_names = []
        path = '.'
        subdirectories = os.listdir(path)
        print(subdirectories)
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
            print(dirname, '-')
            if 'vina_sample' in dirname:  #
                dir_names.append(dirname)
        # print sorted(dir_names)
        return sorted(dir_names)
    except Exception as e:
        print("Problem with finding folders : ", e)
        sys.exit(0)


def find_folders():
    try:
        dir_names = []
        for dirname, dirnames, filenames in os.walk('.'):
            # print dirname, '-'
            if 'vina_sample' in dirname:
                dir_names.append(dirname)
        # print sorted(dir_names)
        return sorted(dir_names)
    except Exception as e:
        print("Problem with finding folders : ", e)
        sys.exit(0)


def copy_VIP_files(file_name, destination):
    import shutil
    shutil.copy2(file_name, destination)


def find_files(format):
    for dirname, dirnames, filenames in os.walk('.'):
        pass
    files = []

    for filename in filenames:
        # print filename
        if '.' + format in filename:
            if '~' not in filename:
                if '#' not in filename:
                    files.append(filename)

    return files


def find_state_file(folder):
    try:
        VIP = []
        for dirname, dirnames, filenames in os.walk(folder):
            # print dirname, '-'
            # print filenames
            for i in filenames:
                # print i
                if '.gas' in i:
                    VIP.append(i)
        return VIP
    except Exception as e:
        print("error in find_files: ", e)
        sys.exit(0)


def create_new_folder(name):
    # path = '.'
    # os.listdir(path)
    if not os.path.exists(name): os.makedirs(name)


def copy_parameter_files(parameter_folder, target_folder, name):
    os.chdir(parameter_folder)
    to_copy = []
    to_copy += find_files('txt') + find_files('pdbqt') + find_files('gas')
    print(to_copy)
    print(target_folder)
    os.chdir(target_folder)
    # os.rename('md_state.gas', '%s-state.gas' %(name))
    os.chdir('..')
    [shutil.copy2('./' + parameter_folder + '/' + i, target_folder) for i in to_copy]  # Here is the stuff


def prep_vina_files(sampleNum, parameter_folder):
    folder_name = 'vina_sample_'
    to_copy = []
    # to_copy += find_files('mdp') + find_files('itp') + find_files('top')  + find_files('ndx') #+ find_files('sh')
    for i in range(1, sampleNum + 1):
        right_folder = folder_name + str(i)
        print("processing Folder: --->  %s" % (right_folder))
        create_new_folder(right_folder)
        file_path = os.path.join('.', right_folder)
        target_path = os.getcwd() + '/' + right_folder
        copy_parameter_files(parameter_folder, target_path, right_folder)
    print('Tuto bene :D')


def runVina(working_folder):
    try:
        print("Running Vina")
        command_to_run = "vina --config conf.txt --log %s.txt --out  %s_out.pdbqt " % (working_folder, working_folder)
        print("Launching new Sim")
        command_to_run
        os.system(command_to_run)
        print("Vina run finished")
    except Exception as e:
        print("error in runSim: ", e)
        sys.exit(0)


def runVinaSim(folders):
    try:
        for folder in folders:
            working_folder = folder[2:]
            print('working folder is ', working_folder)
            VIP_files = sorted(find_state_file(folder))
            print('VIP files is ', VIP_files)
            os.chdir(working_folder)
            cond = read_state_file(VIP_files[-1])
            index = check_condition(cond)
            print('index is ', index)
            print('len is ', len(commands))
            print('okay ', index > len(commands))
            if index >= len(commands):
                print('Check new folder')
                os.chdir('..')
                continue
            else:
                runVina(working_folder)
                hold_nSec(5)
                print('Change in .gas file to Yes')
                replace_state(VIP_files[0], 1)  # Check this out
                print('Now continue :D')
            print('-*-' * 50)
            print('Lets go back')
            os.chdir('..')
    except Exception as e:
        print("Error with Main Function ", e)
        sys.exit(0)
        # finally:
        #     shutdown()


def runLocalFolders(localFolders, num_samples, parameters_folder):
    try:
        for folder in localFolders:
            working_folder = folder
            print("working_folder is ", working_folder)
            os.chdir(working_folder)  # Enters folder

            sample_folders = find_sample_folders()
            print('sample folder is ', sample_folders)
            prep_vina_files(num_samples, parameters_folder)
            runVinaSim(sample_folders)
            os.chdir('..')
    except Exception as e:
        print('Yikes this totally sucks my friends:--> ', e)


def auto_run(localFolders, num_samples, parameters_folder):
    try:
        runLocalFolders(localFolders, num_samples, parameters_folder)
        print('Tuto motherchecking bene :D')
    except (KeyboardInterrupt, SystemExit):
        print('Exiting automated script')
        sys.exit(0)


try:
    num_samples = 100
    parameters_folder = 'param'
    # prep_vina_files(num_samples,parameters_folder)


    localFolders = find_Local_folders('Mon')
    print(sorted(localFolders))
    print(len(localFolders))
    auto_run(localFolders, num_samples, parameters_folder)  # important


except Exception as e:
    print("error is ", e)
