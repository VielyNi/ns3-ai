
#ifndef NS3_DQN_WLAN_H
#define NS3_DQN_WLAN_H

#include <ns3/ai-module.h>
#include <ns3/random-variable-stream.h>
#include <ns3/traced-value.h>
#include <ns3/wifi-remote-station-manager.h>

namespace ns3
{

struct RateStats
{
    uint8_t nss;
    uint16_t channelWidth;
    uint16_t guardInterval;
    uint64_t dataRate;
//    double success;
//    double fails;
//    double lastDecay;

    RateStats()
        : nss(0),
          channelWidth(0),
          guardInterval(0),
          dataRate(0){}
//          success(0),
//          fails(0),
//          lastDecay(0)


};

struct DQNWLANEnv{
    int8_t  type;
    int8_t MCS;
    u_int8_t Distance;
    double Throughput;
    double Throughput_;
    std::array<RateStats, 64> stats;
    int8_t managerId;
    int8_t stationId;
};

struct DQNWLANAct{
    int8_t  type;
    int8_t new_MCS;
    RateStats stats;
    int8_t managerId;
    int8_t stationId;
};


class DQNWifiManager: public WifiRemoteStationManager{
  public:
    static TypeId GetTypeId();
    DQNWifiManager();
    ~DQNWifiManager() override;

  private:
    WifiRemoteStation* DoCreateStation() const override;
    void DoReportRxOk(WifiRemoteStation* station, double rxSnr, WifiMode txMode) override;
    void DoReportRtsFailed(WifiRemoteStation* station) override;
    void DoReportDataFailed(WifiRemoteStation* station) override;
    void DoReportRtsOk(WifiRemoteStation* station,
                       double ctsSnr,
                       WifiMode ctsMode,
                       double rtsSnr) override;
    void DoReportDataOk(WifiRemoteStation* station,
                        double ackSnr,
                        WifiMode ackMode,
                        double dataSnr,
                        uint16_t dataChannelWidth,
                        uint8_t dataNss) override;
    void DoReportAmpduTxStatus(WifiRemoteStation* station,
                               uint16_t nSuccessfulMpdus,
                               uint16_t nFailedMpdus,
                               double rxSnr,
                               double dataSnr,
                               uint16_t dataChannelWidth,
                               uint8_t dataNss) override;
    void DoReportFinalRtsFailed(WifiRemoteStation* station) override;
    void DoReportFinalDataFailed(WifiRemoteStation* station) override;
    WifiTxVector DoGetDataTxVector(WifiRemoteStation* station, uint16_t allowedWidth) override;
    WifiTxVector DoGetRtsTxVector(WifiRemoteStation* station) override;

    /**
     * Initializes station rate tables. If station is already initialized,
     * nothing is done.
     *
     * \param station Station which should be initialized.
     */
    void InitializeStation(WifiRemoteStation* station) const;

    /**
     * Draws a new MCS and related parameters to try next time for this
     * station.
     *
     * This method should only be called between TXOPs to avoid sending
     * multiple frames using different modes. Otherwise it is impossible
     * to tell which mode was used for succeeded/failed frame when
     * feedback is received.
     *
     * \param station Station for which a new mode should be drawn.
     */
    void UpdateNextMode(WifiRemoteStation* station) const;

    /**
     * Returns guard interval in nanoseconds for the given mode.
     *
     * \param st Remote STA.
     * \param mode The WifiMode.
     * \return the guard interval in nanoseconds
     */
    uint16_t GetModeGuardInterval(WifiRemoteStation* st, WifiMode mode) const;


    TracedValue<uint64_t> m_currentRate; //!< Trace rate changes

    int8_t m_ns3ai_manager_id;

};

};

#endif // NS3_DQN_WLAN_H
