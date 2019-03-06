# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# @file    folder_utils.py
# @brief   tools for folder/directory manipulations
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
import subprocess as sub
import pexpect

import sys
from subprocess import PIPE, Popen
from threading  import Thread

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x


def runCommandPopen(command, shell_do=False):
    # comm1 = 'jar'
    # comm2 = 'tvf'
    # comm3 = '{0}{1}{2}'.format(file_path, os.sep, jsbml_jar)
    # total_command = [comm1, comm2, comm3]
    total_command = command.split(' ')
    print(total_command)

    try:
        # command_run = sub.Popen(total_command,  shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
        test = 1
        # command_run = sub.Popen(total_command, shell=shell_do, stdout=sub.PIPE)

        command_run = sub.Popen(total_command , stdout=sub.PIPE)


        current_vina_pid = command_run.pid
        # stdout, stderr = command_run.communicate()


        # TODO not an ideal solution, but this solves continuation problem
        while True:
            output = command_run.stdout.readline().decode()
            if output == '' and command_run.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = command_run.poll()

        output_full = command_run.stdout

        return rc, output, output_full, current_vina_pid


        # if stdout:
        #     # For debugging purposes
        #     # print(stdout)
        #     stdout_value = stdout.decode()  # decode("utf-8")
        #     class_output = stdout_value.split('\n')
        #     # print(class_output)
        #
        #     # Check if it's in the output
        #     #'Writing output ... done.'
        #     # return class_data
        # elif stderr:
        #     error_txt = stderr.decode()
        #     print('error_txt ',error_txt)
        #     # print('ERROR is', error_txt)
        #     # if 'Error: class not found:' in error_txt:
        #     #     return
        #     # else:
        #     #     if extract_data is False:
        #     #         print('Check if Java SDK is installed, deviser requires javap')
        #     #         sys.exit(0)
        #     #     else:
        #     #         return
    except Exception as error:
        print('Error is ', error)
        print('Check something is wrong')
        sys.exit(0)



########################################################################################################################


def runCommandPexpect_gmmpbsa(command, firstIndex, secondIndex, part_num):
    import sys
    # comm1 = 'jar'
    # comm2 = 'tvf'
    # comm3 = '{0}{1}{2}'.format(file_path, os.sep, jsbml_jar)
    # total_command = [comm1, comm2, comm3]
    total_command = command.split(' ')
    print(total_command)

    print('Launching g_mmpbsa pexpect ----->')
    print(command)
    print('------------------------------------------------')

    try:
        # #Pexpect not so well
        a = pexpect.spawn(command, logfile='system.txt')
        log_name  = 'g_mmpbsa_output_{0}.log'.format(part_num)
        fout = open(log_name, "wb")
        a.logfile = fout
        print('g_mmpbsa PID is ', a.pid)


        a.expect(u'Select a group: ')
        print("---------start------------")
        pre_options = a.before

        print(pre_options)

        a.sendline('{0}'.format(firstIndex))

        # print pre_options

        a.expect(u'Select a group: ')
        print("---------start------------")
        pre_options = a.before

        print(str(pre_options))

        a.sendline('{0}'.format(secondIndex))

        # a.expect(pexpect.EOF)
        # print(a.before)
        prompt = u'Thanks for using g_mmpbsa.'
        a.expect_exact(prompt, timeout=5000)
        print(a.before)
        # test = 1
        #
        a.close()
        print( a.exitstatus, a.signalstatus)

        status = a.exitstatus

        if status == 0:
            return status

        test = 1
        # # # command_run = sub.Popen(total_command,  shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
        # command_run = sub.Popen(total_command, stdin=PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        #
        # grep_stdout = p.communicate(input=b'{')[0]
        # # stdout, stderr = command_run.communicate()
        #
        #
        # out, err = command_run.communicate()
        # result = out.decode()
        # print("Result : ", result)
        #
        # return out,err, result
        #
        # # TODO not an ideal solution, but this solves continuation problem
        # while True:
        #     output = command_run.stdout.readline().decode()
        #     if output == '' and command_run.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())
        # rc = command_run.poll()
        #
        # output_full = command_run.stdout
        #
        # return rc, output, output_full


        # if stdout:
        #     # For debugging purposes
        #     # print(stdout)
        #     stdout_value = stdout.decode()  # decode("utf-8")
        #     class_output = stdout_value.split('\n')
        #     # print(class_output)
        #
        #     # Check if it's in the output
        #     #'Writing output ... done.'
        #     # return class_data
        # elif stderr:
        #     error_txt = stderr.decode()
        #     print('error_txt ',error_txt)
        #     # print('ERROR is', error_txt)
        #     # if 'Error: class not found:' in error_txt:
        #     #     return
        #     # else:
        #     #     if extract_data is False:
        #     #         print('Check if Java SDK is installed, deviser requires javap')
        #     #         sys.exit(0)
        #     #     else:
        #     #         return
    except Exception as error:
        print('Error is ', error)
        print('Check something is wrong')
        sys.exit(0)












#########################################################################################################################





def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def runCommandPopen_thread(command):
    # comm1 = 'jar'
    # comm2 = 'tvf'
    # comm3 = '{0}{1}{2}'.format(file_path, os.sep, jsbml_jar)
    # total_command = [comm1, comm2, comm3]
    total_command = command.split(' ')
    print(total_command)

    ON_POSIX = 'posix' in sys.builtin_module_names

    p = Popen(total_command, stdout=PIPE, bufsize=1, close_fds=ON_POSIX)
    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True  # thread dies with the program
    t.start()

    # ... do other things here

    # read line without blocking
    try:
        line = q.get_nowait()  # or q.get(timeout=.1)
        print(line)
    except Empty:
        print('no output yet')
    else:  # got line
        print('darn')
    # ... do something with line