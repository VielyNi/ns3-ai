# Copyright (c) 2023 Huazhong University of Science and Technology
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
# Author: Muyuan Shen <muyuan_shen@hust.edu.cn>
#         Yunfei Ni <Viely@hust.edu.cn>

import torch
import numpy as np
import torch.nn as nn
import random


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(18, 126),
            nn.ReLU(),
            nn.Linear(126, 1),
        )

    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 32
        self.observer_shape = 3
        self.target_replace = 30
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2 * 3 + 2))  # s, a, r, s'
        self.epsilon = 1
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # x = torch.Tensor(x)
        if np.random.randint(0,1) > self.epsilon:  # choose best
            #one hot
            MCS = np.zeros(9)
            MCS[x[0]] = 1
            Dis = np.zeros(9)
            Dis[int(x[1]/5)%9] = 1
            s = np.concatenate((MCS,Dis),axis=0)
            s = torch.Tensor(s)
            # print(f"state:{s}")
            
            reward = self.eval_net.forward(s)
            # print(f"reward:{reward}")
            
            action = torch.argmax(reward[:8])
            # print(f"action:{action}")
            
        else:  # explore
            action = (int(x[0]) + np.random.randint(0, 2)- 1)%9
            # print(f"explore: {action}")
        self.epsilon -= 0.02
        return action#MCS   

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        # if index % 200 == 0:
        #     print('index: %d', index)
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        # print(f"sample list{sample_list}")
        
        sample = self.memory[sample_list, :]
        # print(f"sample{sample}")

        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape + 1])
        r = torch.Tensor(
            sample[:, self.observer_shape + 1:self.observer_shape + 2])
        s_ = torch.Tensor(sample[:, self.observer_shape + 2:])

        # print(f"s{s}")
        #one hot embedding
        sMCS = np.zeros((len(sample_list),9),int)
        for i in range(len(sample_list)):
            sMCS[i,int(s[i,0])] = 1
            
        s_MCS = np.zeros((len(sample_list),9),int)
        for i in range(len(sample_list)):
            s_MCS[i,int(s_[i,0])] = 1
        # print(f"sMCS{sMCS}")

        sDistance = np.zeros((len(sample_list),9),int)
        for i in range(len(sample_list)):
            sDistance[i,int(s[i,1]/5)%9] = 1
        s_Distance = np.zeros((len(sample_list),9),int)
        for i in range(len(sample_list)):
            s_Distance[i,int(s_[i,1]/5)%9] = 1

        state =  np.concatenate((sMCS,sDistance),axis=1)
        state_ = np.concatenate((s_MCS,s_Distance),axis=1)
        
        state = torch.Tensor(state)
        state_ = torch.Tensor(state_)
        # print(f"state{state}")
        
        # print(f"sample state{state}")
        
        q_eval = self.eval_net(state)
        q_next = self.target_net(state_).detach()

        q_target = r + 0.9 * q_next
        # print(f"q_tar: {q_target}")

        loss = self.loss_func(q_eval, q_target)
        # print(f"loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DeepQAgent:

    def __init__(self):
        self.dqn = DQN()
        self.Throughput = 0#datarate last time
        self.s = None   # state
        self.a = None   # action  one int -1,0,1
        self.r = -3200   # reward
        self.s_ = None  # next state

    def get_action(self, obs):

        # print(obs)
        MCS = obs[0]
        Distance = obs[1]
        Throughput = obs[2]
        
        Throughput_ = self.Throughput#the put last time
        self.Throughput = Throughput

        # update DQN
        self.s = self.s_
        self.s_ = [MCS, Distance, Throughput]
        if self.s is not None:  # not first time
            self.r = Throughput - Throughput_
            self.dqn.store_transition(self.s, self.a, self.r, self.s_)
            if self.dqn.memory_counter > self.dqn.memory_capacity:
                self.dqn.learn()

        # choose action
        self.a = self.dqn.choose_action(self.s_)
        
        return self.a
list

