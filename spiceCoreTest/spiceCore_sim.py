from IceCube_gen2_radio.Ara1Detector import Ara1Detector
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
import matplotlib.pyplot as plt


#################
# A1 coordinates taken from https://aradocs.wipac.wisc.edu/0019/001993/001/SPS_ARA_Hot_WaterDrill_Power_Distribution_As_Built-Rev2.pdf
# 51.066 ft N, 38.800 ft E i.e. 15564.917 m N, 11826.24 m E
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
Tx_location = np.array([14883.994, 12940.284, -1100])

ara1 = Ara1Detector(coordinates=np.array([15564.917,11826.24, 0]))
ara1.set_tx_coordinates(Tx_location)
sampling_rate_ara1 = 3.2

spunk = CATH(coordinates=Tx_location)
# Specify Tx antenna model, for now let's use ara bicone:
antenna_info = antennapattern.AntennaPatternProvider()
spunk_antenna = antenna_info.load_antenna_pattern('bicone_v8_inf_n1.78')
spunk_antenna_orientation = np.array([0,0,np.pi/2,0])
freq_range = spunk.get_pulse_fft_freq() # get freq domain corresponding to IDL pulser power spectrum

Event = EventTrace(ev_num=0, station_name='ARA1')

for ch_id in range(0,8):
    rays.set_start_and_end_point(Tx_location, ara1.getAntCoordinate()[ch_id])
    rays.find_solutions()
    n = rays.get_number_of_solutions()

    WF = [np.array([]) for i in range(n)]
    default_trigger_time = 9800 # pick arbitary triggering time for plotting

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
        rx_antenna = ara1.get_antenna_info(ch_id)
        rx_antenna_orientation = ara1.getAntOrientationRotation(ch_id)

        VEL = rx_antenna.get_antenna_response_vectorized(freq_range, zenith_Rx, azimuth_Rx,
                                                            *rx_antenna_orientation)

        pwr_spectrum_rx = np.array([VEL['theta'], VEL['phi']]) * np.array([eTheta, ePhi])

        pwr_spectrum_rx = np.sum(pwr_spectrum_rx, axis=0) * attenuation_ice


        # here go antenna - DAQ amplification chain..

        WF[i_solution] = fft.freq2time(pwr_spectrum_rx, 2*np.max(freq_range))
        # bandpass filter
        WF[i_solution] = butter_bandpass_filter(WF[i_solution], 0.15, 0.85, sampling_rate_ara1, order=4)

    WF_dnR = superimposeWF(WF, dt_DnR, sampling_rate_ara1)
    Wf_shifted = shift_trace(WF_dnR, dt_channel, sampling_rate_ara1)

    Event.set_trace(ch_id, Wf_shifted)

fig, axs = plt.subplots(2, 4)
num_of_samples_per_trace = 2048
maxT = int( num_of_samples_per_trace / (sampling_rate_ara1))
step = maxT / num_of_samples_per_trace
time_axis = np.arange(0, maxT, step)

#x,y row, column
for ch_id in range(0,8):
    y = ch_id % 4
    x = (ch_id)//4
    axs[x, y].plot(time_axis, Event.get_trace(ch_id))
    axs[x, y].set_title('ch' + str(ch_id) )

plt.show()