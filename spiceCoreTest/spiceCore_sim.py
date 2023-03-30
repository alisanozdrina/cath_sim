from IceCube_gen2_radio.AraDetector import AraDetector
from IceCube_gen2_radio.CATH import CATH
from IceCube_gen2_radio.EventTrace import EventTrace
from IceCube_gen2_radio.tools import *

from NuRadioReco.detector import antennapattern
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units, fft
from radiotools import helper as hp

import numpy as np
from scipy import constants
from scipy.signal import find_peaks,peak_widths,peak_prominences
from NuRadioReco.detector.ARA.analog_components import get_system_response
import matplotlib.pyplot as plt

import pandas as pd

#################
# A1 coordinates taken from https://aradocs.wipac.wisc.edu/0019/001993/001/SPS_ARA_Hot_WaterDrill_Power_Distribution_As_Built-Rev2.pdf
# 51.066 ft N, 38.800 ft E i.e. 15564.917 m N, 11826.24 m E
# A2 coordinates:
# 45,382 ft N, 35,528 ft E i.e. 13832.434 m N, 10828.934 m E
# Spice Core coordinates taken from https://aradocs.wipac.wisc.edu/0025/002565/002/SPICE%20coordinates.pdf
# 48.832 ft N, 42.455 ft E i.e. 14883.994 m N, 12940.284 m E
################
c = constants.c #m/sec
n_firn = 1.3

prop = propagation.get_propagation_module('analytic')

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'SP1'

rays = prop(ice, attenuation_model,
            n_frequencies_integration=25,
            n_reflections=0)


# Initialize ARA detector, Spice Core transmitter
Tx_location = np.array([1062.08216737, -2107.26320548,    12.15159203 -1100])

ara2 = AraDetector(coordinates=np.array([0, 0, 0]), st_id = 2)
ara2.set_tx_coordinates(Tx_location)
sampling_rate_ara1 = 3.2

spunk = CATH(coordinates=Tx_location)
# Specify Tx antenna model, for now let's use ara bicone:
antenna_info = antennapattern.AntennaPatternProvider()
spunk_antenna = antenna_info.load_antenna_pattern('bicone_v8_inf_n1.78')
spunk_antenna_orientation = np.array([0,0,np.pi/2,0])
freq_range = spunk.get_pulse_fft_freq() # get freq domain corresponding to IDL pulser power spectrum

Event = EventTrace(ev_num=0, station_name='ARA1')

for ch_id in range(0,8):
    final_point = ara2.getAntCoordinate()[ch_id]
    rays.set_start_and_end_point(Tx_location, final_point)
    rays.find_solutions()
    n = rays.get_number_of_solutions()
    if n == 2:
        print('ray tracing solution exist')
        WF = [np.array([]) for i in range(n)]
        default_trigger_time = 14900 # pick arbitary triggering time for plotting

        dt_channel = 0
        dt_DnR = 0
        for i_solution in range(0, n):
            # get ray tracing trajectory info
            launch_vector = rays.get_launch_vector(i_solution)

            path_length = rays.get_path_length(i_solution)
            if i_solution == 0:
                dt_channel = rays.get_travel_time(i_solution) - default_trigger_time

            if i_solution == 1:
                dt_DnR = rays.get_travel_time(i_solution) - travel_time

            travel_time = rays.get_travel_time(i_solution)

            zenith_emitter, azimuth_emitter = hp.cartesian_to_spherical(*launch_vector)

            VEL = spunk_antenna.get_antenna_response_vectorized(freq_range, zenith_emitter, azimuth_emitter,
                                                                        *spunk_antenna_orientation)
            n_index = ice.get_index_of_refraction(Tx_location)

            eTheta = VEL['theta'] * (-1j) * spunk.get_pulse_fft() * freq_range * n_index / c
            ePhi = VEL['phi'] * (-1j) * spunk.get_pulse_fft() * freq_range * n_index / c

            eTheta *= 1 / path_length
            ePhi *= 1 / path_length

            # apply ice attenuation effect
            attenuation_ice = rays.get_attenuation(i_solution, freq_range, 2*np.max(freq_range))
            # sampling rate is twice the length of original pulse freq band..

            receive_vector = rays.get_receive_vector(i_solution)
            zenith_Rx, azimuth_Rx = hp.cartesian_to_spherical(*receive_vector)
            rx_antenna = ara2.get_antenna_info(ch_id)
            rx_antenna_orientation = ara2.getAntOrientationRotation(ch_id)

            VEL = rx_antenna.get_antenna_response_vectorized(freq_range, zenith_Rx, azimuth_Rx,
                                                                *rx_antenna_orientation)

            pwr_spectrum_rx = np.array([VEL['theta'], VEL['phi']]) * np.array([eTheta, ePhi])

            pwr_spectrum_rx = np.sum(pwr_spectrum_rx, axis=0) * attenuation_ice

            system_response = get_system_response(freq_range)
            system_response = system_response['gain'] * system_response['phase']
            pwr_spectrum_rx *= system_response
            # here go antenna - DAQ amplification chain..

            WF[i_solution] = fft.freq2time(pwr_spectrum_rx, 2*np.max(freq_range))
            # bandpass filter
            WF[i_solution] = butter_bandpass_filter(WF[i_solution], 0.15, 0.85, sampling_rate_ara1, order=4)

        WF_dnR = superimposeWF(WF, dt_DnR, sampling_rate_ara1)
        Wf_shifted = shift_trace(WF_dnR, dt_channel, sampling_rate_ara1)

        Event.set_trace(ch_id, Wf_shifted)
    else:
        Event.set_trace(ch_id, np.zeros(2048))

fig, axs = plt.subplots(2, 4)
num_of_samples_per_trace = 2048
maxT = int( num_of_samples_per_trace / (sampling_rate_ara1))
step = maxT / num_of_samples_per_trace
time_axis = np.arange(0, maxT, step)
dt_DnR_sim = np.zeros(8)
dt_ch_top_sim = np.zeros(4)
dt_ch_b_sim = np.zeros(4)

#x,y row, column
for ch_id in range(0,8):
    y = ch_id % 4
    x = (ch_id)//4
    peaks= find_peak_start(Event.get_trace(ch_id), 0.25e-10, distance=100)
    dt_DnR_sim[ch_id] = (peaks[1] - peaks[0]) * step

    axs[x, y].plot(time_axis,Event.get_trace(ch_id))
    axs[x, y].set_title('ch' + str(ch_id) )
    if ch_id <4:
        dt_ch_top_sim[ch_id] = peaks[1] * step
    else:
        dt_ch_b_sim[ch_id-4] = peaks[1] * step
dt_ch_sim = dt_ch_top_sim - dt_ch_b_sim
plt.show()

#compare with the spice core data
# fig, axs = plt.subplots(2, 4)
# for ch_id in range(0,8):
#     y = ch_id % 4
#     x = (ch_id)//4
#     dt_DnR_spice = ( get_spiceCore_DnR(ch_id) [1] - get_spiceCore_DnR(ch_id) [0] )
#     #axs[x, y].axvline(dt_DnR_spice, color='r')
#     axs[x, y].axvline(dt_DnR_spice - dt_DnR_sim[ch_id], color='k', linestyle='dashed', linewidth=1)
#     min_ylim, max_ylim = axs[x, y].get_ylim()
#     #plt.text(dt_DnR_sim[ch_id] * 1.1, max_ylim * 0.9, 'sim'.format(dt_DnR_sim[ch_id]))
#     axs[x, y].set_title('ch' + str(ch_id))

# plt.show()

# timing diagnostic plots
dt_DnR = np.zeros(8)
for ch_id in range(0,8):
    dt_DnR_spice = get_spiceCore_DnR(ch_id)[1] - get_spiceCore_DnR(ch_id)[0]
    dt_DnR[ch_id] = dt_DnR_spice - dt_DnR_sim[ch_id]
figure, axis = plt.subplots(2, 1)
figure.tight_layout()
#ax = fig.add_axes([0,0,1,1])
ch_n = np.arange(0,8,1)
axis[0].bar(ch_n, -dt_DnR, color = 'b', width = 0.25, label = 'dt direct & reflected')
axis[0].legend()
axis[0].set_title('CATH sim vs Spice Core A2 data')
axis[0].set(xlabel='channel number',ylabel= 'dt, ns')
axis[0].grid()
#plt.show()

dt_ch = np.zeros(4)
for ch_id in range(0,4):
    dt_ch_spice = (get_spiceCore_DnR(ch_id)[1] + get_channel_delay(ch_id)) -\
                  (get_spiceCore_DnR(ch_id+4)[1] +get_channel_delay(ch_id+4))
    dt_ch[ch_id] = dt_ch_spice - dt_ch_sim[ch_id]

#ax = fig.add_axes([0,0,1,1])
ch_n = np.arange(0,4,1)
axis[1].bar(ch_n, -dt_ch, color = 'b', width = 0.25, label = 'dt top and bottom ch')
axis[1].legend()
axis[1].set_title('CATH sim vs Spice Core A2 data')
axis[1].set(xlabel='string number',ylabel= 'dt, ns')
axis[1].grid()
plt.show()

