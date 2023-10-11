#build_lib_example(
#        NAME ns3ai_DQN_gym
#        SOURCE_FILES
#        use_gym/sim.cc
#        use_gym/DQN_WLAN.cc
#        LIBRARIES_TO_LINK
#        ${libai}
#        ${libcore}
#        ${libpoint-to-point}
#        ${libpoint-to-point-layout}
#        ${libnetwork}
#        ${libapplications}
#        ${libmobility}
#        ${libcsma}
#        ${libinternet}
#        ${libwifi}
#        ${libflow-monitor}
#)

build_lib_example(
        NAME ns3ai_DQN_msg
        SOURCE_FILES
        use_msg/DQN_WLAN.cc
        use_msg/sim.cc
        LIBRARIES_TO_LINK
        ${libai}
        ${libcore}
        ${libpoint-to-point}
        ${libpoint-to-point-layout}
        ${libnetwork}
        ${libapplications}
        ${libmobility}
        ${libcsma}
        ${libinternet}
        ${libwifi}

)

pybind11_add_module(ns3ai_DQN_msg_py use_msg/run_DQN_WLAN_py.cc)
set_target_properties(ns3ai_DQN_msg_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/use_msg)
target_link_libraries(ns3ai_DQN_msg_py PRIVATE ${libai})

# Build Python interface along with C++ lib
add_dependencies(ns3ai_DQN_msg ns3ai_DQN_msg_py)

