/*
 * Copyright (c) 2023 Huazhong University of Science and Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author:  Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#include "DQN_WLAN.h"

#include <ns3/ai-module.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ns3ai_DQNWLAN_msg_py, m)
{
    py::class_<ns3::RateStats>(m, "DQNWLANRateStats")
        .def(py::init<>())
        .def_readwrite("nss", &ns3::RateStats::nss)
        .def_readwrite("channelWidth", &ns3::RateStats::channelWidth)
        .def_readwrite("guardInterval", &ns3::RateStats::guardInterval)
        .def_readwrite("dataRate", &ns3::RateStats::dataRate)
        .def("__copy__", [](const ns3::RateStats& self) {
            return ns3::RateStats(self);
        });

    py::class_<ns3::DQNWLANEnv>(m, "PyEnvStruct")
        .def(py::init<>())
        .def_readwrite("MCS", &ns3::DQNWLANEnv::MCS)
        .def_readwrite("Distance", &ns3::DQNWLANEnv::Distance)
        .def_readwrite("Throughput", &ns3::DQNWLANEnv::Throughput)
//        .def_readwrite("Throughput_", &ns3::DQNWLANEnv::Throughput_)


    py::class_<ns3::DQNWLANAct>(m, "PyActStruct")
        .def(py::init<>())
        .def_readwrite("new_MCS", &ns3::DQNWLANAct::new_MCS)


    py::class_<ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::PySendEnd)
        .def("PyGetFinished",
             &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::PyGetFinished)
        .def("GetCpp2PyStruct",
             &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::GetCpp2PyStruct,
             py::return_value_policy::reference)
        .def("GetPy2CppStruct",
             &ns3::Ns3AiMsgInterfaceImpl<ns3::DQNWLANEnv, ns3::DQNWLANAct>::GetPy2CppStruct,
             py::return_value_policy::reference);
}
