
#include "DQN_WLAN.h"
#include <ns3/core-module.h>
#include <ns3/double.h>
#include <ns3/log.h>
#include <ns3/packet.h>
#include <ns3/wifi-phy.h>

#include <iostream>
#include <vector>

namespace ns3{

NS_OBJECT_ENSURE_REGISTERED(DQNWifiManager);

NS_LOG_COMPONENT_DEFINE("DQNWifiManager");

/**
 * A structure containing parameters of a single rate and its
 * statistics.
 */
struct AiRateStats
{
    WifiMode mode;         ///< MCS
    uint16_t channelWidth; ///< channel width in MHz
    uint8_t nss;           ///< Number of spatial streams
//    uint8_t mcs;
};

/**
 * Holds station state and collected statistics.
 *
 * This struct extends from WifiRemoteStation to hold additional
 * information required by ThompsonSamplingWifiManager.
 */
struct DQNWifiRemoteStation : public WifiRemoteStation
{
    std::vector<AiRateStats> m_mcsStats; //!< Collected statistics

};

TypeId
DQNWifiManager::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::DQNWifiManager")
            .SetParent<WifiRemoteStationManager>()
            .SetGroupName("Wifi")
            .AddConstructor<DQNWifiManager>()
            .AddAttribute(
                "Distance",
                "Distance between ap and sta",
                DoubleValue(0.0),
                MakeDoubleAccessor(&DQNWifiManager::Distance),
                MakeDoubleChecker<double>(-2000))
            .AddTraceSource("Rate",
                            "Traced value for rate changes (b/s)",
                            MakeTraceSourceAccessor(&DQNWifiManager::m_currentRate),
                            "ns3::TracedValueCallback::Uint64");
    return tid;
}
DQNWifiManager::DQNWifiManager()
    : m_currentRate{0}
{
//    NS_LOG_FUNCTION(this);
//    auto interface = Ns3AiMsgInterface::Get();
//    interface->SetIsMemoryCreator(false);
//    interface->SetUseVector(false);
//    interface->SetHandleFinish(true);
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        interface->GetInterface<DQNWLANEnv, DQNWLANAct>();
//
//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x01;
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    m_ns3ai_manager_id = msgInterface->GetPy2CppStruct()->managerId;
//    msgInterface->CppRecvEnd();
}

DQNWifiManager::~DQNWifiManager()
{
    NS_LOG_FUNCTION(this);
}

WifiRemoteStation*
DQNWifiManager::DoCreateStation() const
{
    NS_LOG_FUNCTION(this);
    DQNWifiRemoteStation* station = new DQNWifiRemoteStation();
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();
//
//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x02;
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    station->m_ns3ai_station_id = msgInterface->GetPy2CppStruct()->stationId;
//    msgInterface->CppRecvEnd();

    return station;
}

void
DQNWifiManager::InitializeStation(WifiRemoteStation* st) const
{
    auto station = static_cast<DQNWifiRemoteStation*>(st);
    if (!station->m_mcsStats.empty())
    {
        return;
    }

    // Add HT, VHT or HE MCSes
    for (const auto& mode : GetPhy()->GetMcsList())
    {
        for (uint16_t j = 20; j <= GetPhy()->GetChannelWidth(); j *= 2)
        {
            WifiModulationClass modulationClass = WIFI_MOD_CLASS_VHT;
//            if (GetVhtSupported())
//            {
//                modulationClass = WIFI_MOD_CLASS_VHT;
//            }
//            if (GetHeSupported())
//            {
//                modulationClass = WIFI_MOD_CLASS_HE;
//            }
            if (mode.GetModulationClass() == modulationClass)
            {
//                for (uint8_t k = 1; k <= GetPhy()->GetMaxSupportedTxSpatialStreams(); k++)
//                {
                    if (mode.IsAllowed(j, 1))
                    {
                        AiRateStats init_stats;
                        init_stats.mode = mode;
                        init_stats.channelWidth = j;
                        init_stats.nss = 1;
//                        init_stats.mcs = station->m_mcsStats.size();
                        station->m_mcsStats.push_back(init_stats);
                    }
//                }
            }
        }
    }
    //useless in this example
//    if (station->m_mcsStats.empty())
//    {
//        // Add legacy non-HT modes.
//        for (uint8_t i = 0; i < GetNSupported(station); i++)
//        {
//            AiRateStats stats;
//            stats.mode = GetSupported(station, i);
//            if (stats.mode.GetModulationClass() == WIFI_MOD_CLASS_DSSS ||
//                stats.mode.GetModulationClass() == WIFI_MOD_CLASS_HR_DSSS)
//            {
//                stats.channelWidth = 22;
//            }
//            else
//            {
//                stats.channelWidth = 20;
//            }
//            stats.nss = 1;
//            station->m_mcsStats.push_back(stats);
//        }
//    }

    NS_ASSERT_MSG(!station->m_mcsStats.empty(), "No usable MCS found");

//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();

//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x03;
//    msgInterface->GetCpp2PyStruct()->managerId = m_ns3ai_manager_id;
//    msgInterface->GetCpp2PyStruct()->stationId = station->m_ns3ai_station_id;

    NS_ASSERT_MSG(station->m_mcsStats.size() <= 9, "m_mcsStats too long");

//    auto& s = msgInterface->GetCpp2PyStruct()->stats;
//    for (size_t i = 0; i < station->m_mcsStats.size(); i++)
//    {
//        const WifiMode mode{station->m_mcsStats.at(i).mode};
//        s.at(i).nss = station->m_mcsStats.at(i).nss;
//        s.at(i).channelWidth = station->m_mcsStats.at(i).channelWidth;
//        s.at(i).guardInterval = GetModeGuardInterval(st, mode);
//        s.at(i).dataRate =
//            mode.GetDataRate(s.at(i).channelWidth, s.at(i).guardInterval, s.at(i).nss);
//    }
//    s[station->m_mcsStats.size()].lastDecay = -1.0;
//    msgInterface->CppSendEnd();

//    msgInterface->CppRecvBegin();
//    NS_ASSERT_MSG(msgInterface->GetPy2CppStruct()->stationId ==
//                      msgInterface->GetCpp2PyStruct()->stationId,
//                  "Error 0x03");
//    msgInterface->CppRecvEnd();

    UpdateNextMode(st);
}

void
DQNWifiManager::DoReportRxOk(WifiRemoteStation* station,
                                            double rxSnr,
                                            WifiMode txMode)
{
    NS_LOG_FUNCTION(this << station << rxSnr << txMode);
}

void
DQNWifiManager::DoReportRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
DQNWifiManager::DoReportDataFailed(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    InitializeStation(st);
//    auto station = static_cast<DQNWifiRemoteStation*>(st);
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();

//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x05;
//    msgInterface->GetCpp2PyStruct()->managerId = m_ns3ai_manager_id;
//    msgInterface->GetCpp2PyStruct()->stationId = station->m_ns3ai_station_id;
//    msgInterface->GetCpp2PyStruct()->data.decay.decay = m_decay;
//    msgInterface->GetCpp2PyStruct()->data.decay.now = Simulator::Now().GetSeconds();
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    NS_ASSERT_MSG(msgInterface->GetPy2CppStruct()->stationId ==
//                      msgInterface->GetCpp2PyStruct()->stationId,
//                  "Error 0x05");
//    msgInterface->CppRecvEnd();
}

void
DQNWifiManager::DoReportRtsOk(WifiRemoteStation* st,
                                             double ctsSnr,
                                             WifiMode ctsMode,
                                             double rtsSnr)
{
    NS_LOG_FUNCTION(this << st << ctsSnr << ctsMode.GetUniqueName() << rtsSnr);
}

void
DQNWifiManager::UpdateNextMode(WifiRemoteStation* st) const
{
    InitializeStation(st);
    auto station = static_cast<DQNWifiRemoteStation*>(st);
    NS_ASSERT(!station->m_mcsStats.empty());
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();
//
//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x0a;
//    msgInterface->GetCpp2PyStruct()->managerId = m_ns3ai_manager_id;
//    msgInterface->GetCpp2PyStruct()->stationId = station->m_ns3ai_station_id;
////    msgInterface->GetCpp2PyStruct()->data.decay.decay = m_decay;
////    msgInterface->GetCpp2PyStruct()->data.decay.now = Simulator::Now().GetSeconds();
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    NS_ASSERT_MSG(msgInterface->GetPy2CppStruct()->stationId ==
//                      msgInterface->GetCpp2PyStruct()->stationId,
//                  "Error 0x0a");
//    msgInterface->CppRecvEnd();
}

void
DQNWifiManager::DoReportDataOk(WifiRemoteStation* st,
                                              double ackSnr,
                                              WifiMode ackMode,
                                              double dataSnr,
                                              uint16_t dataChannelWidth,
                                              uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << ackSnr << ackMode.GetUniqueName() << dataSnr);
    InitializeStation(st);
//    auto station = static_cast<DQNWifiRemoteStation*>(st);
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();

//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x06;
//    msgInterface->GetCpp2PyStruct()->managerId = m_ns3ai_manager_id;
//    msgInterface->GetCpp2PyStruct()->stationId = station->m_ns3ai_station_id;
//    msgInterface->GetCpp2PyStruct()->data.decay.decay = m_decay;
//    msgInterface->GetCpp2PyStruct()->data.decay.now = Simulator::Now().GetSeconds();
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    NS_ASSERT_MSG(msgInterface->GetPy2CppStruct()->stationId ==
//                      msgInterface->GetCpp2PyStruct()->stationId,
//                  "Error 0x06");
//    msgInterface->CppRecvEnd();
}

void
DQNWifiManager::DoReportAmpduTxStatus(WifiRemoteStation* st,
                                                     uint16_t nSuccessfulMpdus,
                                                     uint16_t nFailedMpdus,
                                                     double rxSnr,
                                                     double dataSnr,
                                                     uint16_t dataChannelWidth,
                                                     uint8_t dataNss)
{
    NS_LOG_FUNCTION(this << st << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr);
    InitializeStation(st);
//    auto station = static_cast<DQNWifiRemoteStation*>(st);
//    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
//        Ns3AiMsgInterface::Get()
//            ->GetInterface<DQNWLANEnv, DQNWLANAct>();
//
//    msgInterface->CppSendBegin();
//    msgInterface->GetCpp2PyStruct()->type = 0x07;
//    msgInterface->GetCpp2PyStruct()->managerId = m_ns3ai_manager_id;
//    msgInterface->GetCpp2PyStruct()->stationId = station->m_ns3ai_station_id;
//    msgInterface->GetCpp2PyStruct()->var = (uint64_t)nSuccessfulMpdus << 32 | nFailedMpdus;
//    msgInterface->GetCpp2PyStruct()->data.decay.decay = m_decay;
//    msgInterface->GetCpp2PyStruct()->data.decay.now = Simulator::Now().GetSeconds();
//    msgInterface->CppSendEnd();
//
//    msgInterface->CppRecvBegin();
//    NS_ASSERT_MSG(msgInterface->GetPy2CppStruct()->stationId ==
//                      msgInterface->GetCpp2PyStruct()->stationId,
//                  "Error 0x07");
//    msgInterface->CppRecvEnd();
}

void
DQNWifiManager::DoReportFinalRtsFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

void
DQNWifiManager::DoReportFinalDataFailed(WifiRemoteStation* station)
{
    NS_LOG_FUNCTION(this << station);
}

uint16_t
DQNWifiManager::GetModeGuardInterval(WifiRemoteStation* st, WifiMode mode) const
{
//    if (mode.GetModulationClass() == WIFI_MOD_CLASS_HE)
//    {
//        return std::max(GetGuardInterval(st), GetGuardInterval());
//    }
//    else if ((mode.GetModulationClass() == WIFI_MOD_CLASS_HT) ||
//             (mode.GetModulationClass() == WIFI_MOD_CLASS_VHT))
//    {
//        return std::max<uint16_t>(GetShortGuardIntervalSupported(st) ? 400 : 800,
//                                  GetShortGuardIntervalSupported() ? 400 : 800);
//    }
//    else
//    {
//        return 800;
//    }
        //because I only use 802.11ac with guard interval 0.8us
        return 800;
}

WifiTxVector
DQNWifiManager::DoGetDataTxVector(WifiRemoteStation* st, uint16_t allowedWidth)
{
    NS_LOG_FUNCTION(this << st);
    InitializeStation(st);
    auto station = static_cast<DQNWifiRemoteStation*>(st);
    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
        Ns3AiMsgInterface::Get()
            ->GetInterface<DQNWLANEnv, DQNWLANAct>();

    msgInterface->CppSendBegin();
    msgInterface->GetCpp2PyStruct()->MCS = MCS;
    printf("Manager:%f\n",Distance);
    msgInterface->GetCpp2PyStruct()->Distance = Distance;
    msgInterface->GetCpp2PyStruct()->Throughput = station->m_mcsStats.at(MCS).mode.
                                                  GetDataRate(station->m_mcsStats.at(MCS).channelWidth,
                                                              800,
                                                              station->m_mcsStats.at(MCS).nss)/100000;
//    msgInterface->GetCpp2PyStruct()->Throughput_ = Throughput;
    msgInterface->CppSendEnd();

    msgInterface->CppRecvBegin();
    WifiMode mode = station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).mode;
    uint8_t nss = station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).nss;//msgInterface->GetPy2CppStruct()->stats.nss;
    uint16_t channelWidth =
        std::min(station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).channelWidth, GetPhy()->GetChannelWidth());
    uint16_t guardInterval = 800;
    MCS = msgInterface->GetPy2CppStruct()->new_MCS;
    msgInterface->CppRecvEnd();

    uint64_t rate = mode.GetDataRate(channelWidth, guardInterval, nss);
    if (m_currentRate != rate)
    {
        NS_LOG_DEBUG("New datarate: " << rate);
        m_currentRate = rate;
    }

    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        guardInterval,
        GetNumberOfAntennas(),
        nss,
        0, // NESS
        GetPhy()->GetTxBandwidth(mode, GetChannelWidth(st)),
        GetAggregation(station),
        false);
}

WifiTxVector
DQNWifiManager::DoGetRtsTxVector(WifiRemoteStation* st)
{
    NS_LOG_FUNCTION(this << st);
    InitializeStation(st);
    auto station = static_cast<DQNWifiRemoteStation*>(st);
    Ns3AiMsgInterfaceImpl<DQNWLANEnv, DQNWLANAct>* msgInterface =
        Ns3AiMsgInterface::Get()
            ->GetInterface<DQNWLANEnv, DQNWLANAct>();

    msgInterface->CppSendBegin();
    msgInterface->GetCpp2PyStruct()->MCS = MCS;
    printf("Manager:%f\n",Distance);
    msgInterface->GetCpp2PyStruct()->Distance = Distance;
    msgInterface->GetCpp2PyStruct()->Throughput = station->m_mcsStats.at(MCS).mode.
                                                  GetDataRate(station->m_mcsStats.at(MCS).channelWidth,
                                                              800,
                                                              station->m_mcsStats.at(MCS).nss)/100000;
//    msgInterface->GetCpp2PyStruct()->Throughput_ = Throughput;
    msgInterface->CppSendEnd();

    msgInterface->CppRecvBegin();
    WifiMode mode = station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).mode;
    uint8_t nss = station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).nss;//msgInterface->GetPy2CppStruct()->stats.nss;
    uint16_t channelWidth =
        std::min(station->m_mcsStats.at(msgInterface->GetPy2CppStruct()->new_MCS).channelWidth,
                 GetPhy()->GetChannelWidth());
    uint16_t guardInterval = 800;
    MCS = msgInterface->GetPy2CppStruct()->new_MCS;
//    uint8_t nss = msgInterface->GetPy2CppStruct()->stats.nss;
//    uint16_t channelWidth =
//        std::min(msgInterface->GetPy2CppStruct()->stats.channelWidth, GetPhy()->GetChannelWidth());
//    uint16_t guardInterval = msgInterface->GetPy2CppStruct()->stats.guardInterval;
    msgInterface->CppRecvEnd();

    // Make sure control frames are sent using 1 spatial stream.
    NS_ASSERT(nss == 1);

    return WifiTxVector(
        mode,
        GetDefaultTxPowerLevel(),
        GetPreambleForTransmission(mode.GetModulationClass(), GetShortPreambleEnabled()),
        guardInterval,
        GetNumberOfAntennas(),
        nss,
        0, // NESS
        channelWidth,
        GetAggregation(station),
        false);
}

} // namespace ns3

