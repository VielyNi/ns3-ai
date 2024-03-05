# Copyright (c) 2020-2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>
#         Muyuan Shen <muyuan_shen@hust.edu.cn>

import os
import torch
import argparse
import numpy as np
import string
import matplotlib.pyplot as plt
from agents import DeepQAgent
import sys
import traceback
import ns3ai_DQN_msg_py as py_binding
from ns3ai_utils import Experiment


# initialize variable

parser = argparse.ArgumentParser()

parser.add_argument('--steps', type=int,
                    help='How many different distances to try')
parser.add_argument('--stepsTime', type=int,
                    help='Time on each step')
parser.add_argument('--stepsSize', type=int,
                    help='Distance between steps')

parser.add_argument('--AP1_x', type=int,
                    help='Position of AP1 in x coordinate')
parser.add_argument('--AP1_y', type=int,
                    help='Position of AP1 in y coordinate')
parser.add_argument('--STA1_x', type=int,
                    help='Position of STA1 in x coordinate')
parser.add_argument('--STA1_y', type=int,
                    help='Position of STA1 in y coordinate')

parser.add_argument('--show_log', action='store_true',
                    help='whether show observation and action')
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--result_dir', type=str,
                    default='./DQN_WLAN_results', help='output figures path')

args = parser.parse_args()

res_list = ['MCS', 'Distance', 'Throughput']
if args.result:
    for res in res_list:
        globals()[res] = []

stepIdx = 0

ns3Settings = {
    'apManager': 'DQN'}
exp = Experiment("ns3ai_DQN_msg", "../../../../../", py_binding, handleFinish=True)
msgInterface = exp.run(setting=ns3Settings, show_output=True)

try:

    Agent = DeepQAgent()
    while True:
        # receive observation from C++
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            print("Simulation ended")
            break
        #MCS
        MCS = msgInterface.GetCpp2PyStruct().MCS
        #the distance
        Distance = msgInterface.GetCpp2PyStruct().Distance
        #the put this time
        Throughput = msgInterface.GetCpp2PyStruct().Throughput
        #the put last time
        # Throughput_ = msgInterface.GetCpp2PyStruct().Throughput_
        msgInterface.PyRecvEnd()

        obs = [MCS, Distance, Throughput]
        if args.show_log:
            print("Recv obs:", obs)

        if args.result:
            for res in res_list:
                globals()[res].append(globals()[res])

        new_MCS = Agent.get_action(obs)

        # send action to C++
        msgInterface.PySendBegin()
        msgInterface.GetPy2CppStruct().new_MCS = new_MCS
        msgInterface.PySendEnd()



        if args.show_log:
            print("Step:", stepIdx)
            stepIdx += 1
            print("Send act:", act)

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

else:
    if args.result:
        if args.result_dir:
            if not os.path.exists(args.result_dir):
                os.mkdir(args.result_dir)
        for res in res_list:
            y = globals()[res]
            x = range(len(y))
            plt.clf()
            plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
            plt.xlabel('Step Number')
            plt.title('Information of {}'.format(res[:-2]))
            plt.savefig('{}.png'.format(os.path.join(args.result_dir, res[:-2])))

finally:
    print("Finally exiting...")
    del exp
