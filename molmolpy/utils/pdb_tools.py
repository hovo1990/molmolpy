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


def write_row_pdb(row, res_num=None):
    '''
    This function is for writing one row in a PDB file
    '''
    # print('row is ',row)
    try:
        s = ''
        s += "%-6s" % str(row[0])
        s += '%5d' % (int(row[1]))

        # s += ' %4s' %(str(row[2]))
        s += ' '
        s += '{:^4}'.format(str(row[2]))
        s += '%1s' % (' ')
        s += '%3s' % (str(row[3]))
        if row[4] is None:
            s += ' %1s' % ('Z')
        else:
            s += ' %1s' % (str(row[4]))
        # s += ' '
        if res_num is None:
            s += '%4d' % (int(row[5]))  #
            s += '%1s' % (' ')
        else:
            s += '{:>4}'.format(str(resNum))  #
        s += '   %8.3f' % (float(row[6]))
        s += '%8.3f' % (float(row[7]))
        s += '%8.3f' % (float(row[8]))
        s += '%6.2f' % (float((row[9])))
        s += '%6.2f' % (float(row[10]))
        if "\n" not in row[12]:
            row[12] += '\n'
        s += '           %2s' % (str(row[12]))
        # print('s is ',s)
        # if s[0] == ' ':
        #     print('check')
        #     s = s[1:]
        # print('len s is ',len(s))
        return s
    except Exception as e:
        print('error at write_rowPDB is ', e)


def write_row_pandas_pdb(row, res_num=None):
    '''
    This function is for writing one row in a PDB file
    '''
    # print('row is ',row)
    try:
        s = ''
        s += "%-6s" % str(row['ATOM'])
        s += '%5d' % (int(row['SerialNum']))

        # s += ' %4s' %(str(row[2]))
        s += ' '
        s += '{:^4}'.format(str(row['AtomName']))
        s += '%1s' % (' ')
        s += '%3s' % (str(row['ResidueName']))
        if row[4] is None:
            s += ' %1s' % ('Z')
        else:
            s += ' %1s' % (str(row['ChainId']))
        # s += ' '
        if res_num is None:
            s += '%4d' % (int(row['ChainNum']))  #
            s += '%1s' % (' ')
        else:
            s += '{:>4}'.format(str(res_num))  #
        s += '   %8.3f' % (float(row['X']))
        s += '%8.3f' % (float(row['Y']))
        s += '%8.3f' % (float(row['Z']))
        s += '%6.2f' % (float((row['Occupancy'])))
        s += '%6.2f' % (float(row['TempFactor']))
        # if "\n" not in row['ElemSymbol']:
        #     row[12] += '\n'
        s += '           %2s\n' % (str(row['ElemSymbol']))
        # print('s is ',s)
        # if s[0] == ' ':
        #     print('check')
        #     s = s[1:]
        # print('len s is ',len(s))
        return s
    except Exception as e:
        print('error at write_rowPDB is ', e)


def write_lig(model, res_num=None, file_to_write=None):
    # print("Hello write_lig")
    # print('data to write ',data)
    for index, row in model.iterrows():
        # print('row is ', row)
        s = write_row_pandas_pdb(row, res_num)
        file_to_write.write(s)


    # for line in model:
        # print('line is ',line)
        # s = write_row_pdb(line, res_num)
        # file_to_write.write(s)
