import numpy as np
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, fft
import uproot
class CATH:  # matching description from gen2 proposal

    def __init__(self, name='', coordinates=np.array([0, 0, 0])):

        self._name = name
        self._coordinates = coordinates
        self._coordinates_surface_antenna = np.array([coordinates[0], coordinates[1], +1])
        self._coordinates_deep_antenna_vpol = np.array([coordinates[0], coordinates[1], -349])
        self._coordinates_deep_antenna_hpol = np.array([coordinates[0], coordinates[1], -350])
        self.pulse = np.array([])  # 0,1 and 2 are surface, vpol and hpol emitters
        self.pulse_fft = np.array([])
        self.pulse_fft_freq = np.array([])

    antenna_info = antennapattern.AntennaPatternProvider()
    antenna_surface_Tx = antenna_info.load_antenna_pattern('bicone_v8_InfAir')
    antenna_inIce_Tx_vpol = antenna_info.load_antenna_pattern('RNOG_vpol_4inch_center_n1.73')
    antenna_inIce_Tx_hpol = antenna_info.load_antenna_pattern('RNOG_quadslot_v2_n1.74')

    surface_Tx_orientation = np.array([0, 0, np.pi / 2, 0])
    inIce_Tx_vpol_orientation = np.array([0, 0, np.pi / 2, 0])
    inIce_Tx_hpol_orientation = np.array([0, 0, np.pi / 2, np.pi / 2])

    # antennas orientation and rotation chosen similarly to https://github.com/nu-radio/NuRadioMC/blob/develop/NuRadioReco/detector/RNO_G/RNO_season_2021.json

    def set_pulse(self, PulserType='IDL'):
        # print(PulserType)
        if PulserType == 'IDL':
            file = uproot.open('/Users/alisanozdrina/Documents/phys/exp/icecube_gen2/cal_tower/data/idl2.root')
            time_ns = file['waveforms_ffts/Wav_1400'].values()[0] * 1e9
            v_V = file['waveforms_ffts/Wav_1400'].values()[1]
            v_V = v_V * 1e3  # increase amplitude 1000 times to make up 60bB attenuation
            v_V = v_V * (np.sqrt(2) / 2)  # add 3db attenuation coming from 1:1 power splitter
            N = len(v_V)
            dt = 0.4  # ns
            sampling_rate = 1 / dt  # GHz
            freq_axis = np.fft.rfftfreq(N, dt)
            fft_idl = fft.time2freq(v_V, sampling_rate)
        # add 3db attenuation coming from 1:1 power splitter
        if PulserType == 'ICL':
            array = np.genfromtxt('/Users/alisanozdrina/Downloads/Default Dataset.csv', delimiter=',')
            v_V = array[:, 1] + 2000
            dt = array[1, 0] * 1e-3 - array[0, 0] * 1e-3  # ns
            v_V = v_V * (np.sqrt(2) / 2)  # add 3db attenuation coming from 1:1 power splitter
            N = len(v_V)

            sampling_rate = 1 / dt  # GHz
            freq_axis = np.fft.rfftfreq(N, dt)
            fft_idl = fft.time2freq(v_V, sampling_rate)

            fft_idl = np.interp(np.arange(0.0, 1.250, 0.005), freq_axis, fft_idl)
            freq_axis = np.arange(0.0, 1.250, 0.005)

        self.pulse = v_V
        self.pulse_fft = fft_idl
        self.pulse_fft_freq = freq_axis

    def get_trace(self):
        self.set_pulse()
        return self.pulse

    def get_pulse_fft(self):
        self.set_pulse()
        return self.pulse_fft

    def get_pulse_fft_freq(self):
        self.set_pulse()
        return self.pulse_fft_freq
