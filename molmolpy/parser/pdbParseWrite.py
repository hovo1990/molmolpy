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


def write_rowPDB(row, resNum=None):
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
        if row[4] == None:
            s += ' %1s' % ('Z')
        else:
            s += ' %1s' % (str(row[4]))
        # s += ' '
        # if resNum == None:
        s += '%4d' % (int(row[5]))  #
        s += '%1s' % (' ')
        # else:
        #     s += '{:>4}'.format(str(resNum)) #
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


def helper_function(line, n):
    if '\n' in line[n]:
        # print 'line[12] is ', line[12]
        last_elem = str(line[n][0:-1])
    else:
        last_elem = str(line[n])
    return last_elem


def write_Lig(data, file_to_write=None):
    # print("Hello write_lig")
    # print('data to write ',data)
    for line in data:
        # print('line is ',line)
        s = write_rowPDB(line)
        file_to_write.write(s)



        # def prepFile(filename = None):
        # shutil.copy2('./param/modelStab.pdb', './param/modelStabLig.pdb')


#
def prepData(data):
    print('=======================================' * 30)
    # print('new module ',data)
    dataToWrite = []
    atomNum = 1
    file_to_write = open('modelStabLig.pdb', 'a')
    for i in data:
        # print('i is ',i)
        col1 = 'ATOM'
        col2 = str(atomNum)
        col3 = 'C'
        col4 = 'DUM'
        col5 = 'A'
        col6 = '1'
        col7 = round(i[0], 3)
        col8 = round(i[1], 3)
        col9 = round(i[2], 3)
        col10 = '1.00'
        col11 = '0.00'
        col12 = 'C'
        temp_data = [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12]
        text_to_write = write_rowPDB(temp_data)
        # print('verify yolo Haha ',text_to_write)
        file_to_write.write(text_to_write)
    file_to_write.close()


def runMod(ligData=None):
    # prepFile()
    prepData(ligData)
