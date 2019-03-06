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


def create_folder(new_path='.'):
    # newpath = r'C:\Program Files\arbitrary'
    if not os.path.exists(new_path):
        os.mkdir(new_path)


def find_sample_folders(folder_path='.', dir_name='vina_sample'):
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


def find_folder_in_path(folder_path='.', dir_name='vina_sample'):
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


def find_folders(folderInit):
    try:
        dir_names = []
        path = '.'
        subdirectories = os.listdir(path)
        # print(subdirectories)
        for dirname in subdirectories:
            if folderInit in dirname:
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


def find_files_in_folder(folder, data_format, exclude=None, include=None):
    try:
        VIP = []
        for dirname, dirnames, filenames in os.walk(folder):
            # print dirname, '-'
            # print filenames
            for i in filenames:
                # print i
                if data_format in i:
                    if type(include) is list:
                        include = tuple(include)
                    else:
                        include =''
                    if type(exclude) is list:
                        exclude = tuple(exclude)
                    else:
                        exclude=''


                    if i.startswith(include):
                        if i.endswith(exclude):
                            pass
                        else:
                            VIP.append(i)
        return VIP
    except Exception as e:
        print("error in find_files: ", e)
        sys.exit(0)



def find_files_in_folder_uberDocker(folder, data_format, exclude=None, include=None):
    try:
        VIP = []
        for dirname, dirnames, filenames in os.walk(folder):
            # print dirname, '-'
            # print filenames
            for i in filenames:
                # print i
                if data_format in i:
                    if type(include) is list or type(include) is tuple:
                        include = tuple(include)
                    else:
                        include ='none'
                    if type(exclude) is list or type(exclude) is tuple:
                        exclude = tuple(exclude)
                    else:
                        exclude='none'

                    test = 1

                    if i.startswith(include):
                        if i.endswith(exclude):
                            pass
                        else:
                            VIP.append(i)
        return VIP
    except Exception as e:
        print("error in find_files: ", e)
        sys.exit(0)


def find_simple_file(folder, data_format):
    try:
        VIP = []
        for dirname, dirnames, filenames in os.walk(folder):
            # print dirname, '-'
            # print filenames
            for i in filenames:
                # print i
                if data_format in i:
                    VIP.append(i)
        return VIP
    except Exception as e:
        print("error in find_files: ", e)
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
