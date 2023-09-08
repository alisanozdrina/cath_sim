import matplotlib.pyplot as plt
from IceCube_gen2_radio.IC_hybrid_station import *
from IceCube_gen2_radio.rno_g_st import rno_g_st
from IceCube_gen2_radio.tools import *
from IceCube_gen2_radio.tools_rno import *
import scipy
from IceCube_gen2_radio.CATH import CATH
from IceCube_gen2_radio.EventTrace import EventTrace
from IceCube_gen2_radio.noise import *
# define ice parameters
n_index = 1
c = constants.c * units.m / units.s
n_firn = 1.3
n_air = 1.00029
n_ice = 1.74
prop = propagation.get_propagation_module('analytic')

#for the Summit station
ref_index_model = 'greenland_simple'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'GL1'


rays = prop(ice, attenuation_model,
            n_frequencies_integration=25,
            n_reflections=0)

delayBW_SurfDeep = 0.455 * 1e3  # ns

# define detector layout
#Num_of_stations = 4
Num_of_stations = 2
array_of_st = np.empty(Num_of_stations, dtype=rno_g_st)
array_of_st[0] = rno_g_st('rno 1', np.array([-625, 0, 0]))
array_of_st[1] = rno_g_st('rno 2', np.array([+625, 0, 0]))

# array_of_st[0] = rno_g_st('rno 1', np.array([-400, +400, 0]))
# array_of_st[1] = rno_g_st('rno 2', np.array([+400, 400, 0]))
# array_of_st[2] = rno_g_st('rno 3', np.array([-400, -400, 0]))
# array_of_st[3] = rno_g_st('rno 4', np.array([400, -400, 0]))

# define CATH coordinates, antennas, antenna orientation, pulser trace - needs separate class

cath_site = CATH(name='cath 1', coordinates=np.array([0, 0, 5]))
freq_range = cath_site.get_pulse_fft_freq()

freq_range_noise = np.arange(freq_range[0], freq_range[-1],
                             (freq_range[-1] - freq_range[0]) / 1025)

# define a single event - need event class, would contain traces for each channel and timing information
Event = np.empty(Num_of_stations,
                 dtype=EventTrace)  # for now, we only work with a single event, but each station has it's own Event object

##################################
# Pick CATH Pulsing Mode:
#pulsingMode = Vpol and Hpol # maybe will add CW mode in the future
pulsingMode = 'Vpol'
#################################
surf_ch_num = np.array([12,13,14,15,16,17,18,19,20])
# loop through every station
for st_id in range(0, Num_of_stations):
    ev_id = st_id
    Event[ev_id] = EventTrace(ev_id, array_of_st[st_id]._name)

    # loop through surface channels:
    for count, ch_id in enumerate(surf_ch_num):
        # find travel distance: surface Tx between surface channels

        array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_surface_antenna)
        travel_distance = array_of_st[st_id].getAntennaCoordinate()[ch_id][0]  # travel distance - array of 16 elements
        # find angle for radiation pattern
        rad_pattern = array_of_st[st_id].getRelRadiationAngle()
        # calculate antenna radiation patter
        zenith = rad_pattern[ch_id][0]
        azimuth = rad_pattern[ch_id][1]

        antenna_orientation = cath_site.surface_Tx_orientation

        VEL = cath_site.antenna_surface_Tx.get_antenna_response_vectorized(freq_range, zenith, -azimuth,
                                                                           *antenna_orientation)
        # propagate through the air - 1/R factor, save propagation time
        # CATHSim1 configuration
        idl_pulse_attenuated = cath_site.get_pulse_fft() * \
                               cable_attenuation(freq_range, -cath_site._coordinates_deep_antenna_vpol[2])
        eTheta = VEL['theta'] * (-1j) * idl_pulse_attenuated * freq_range * n_index / c
        ePhi = VEL['phi'] * (-1j) * idl_pulse_attenuated * freq_range * n_index / c

        eTheta = eTheta / travel_distance
        ePhi = ePhi / travel_distance
        # if the wave passes air/ice boundary - correct on fresnel coefficient

        if array_of_st[st_id].getAntennaCoordinate()[ch_id][2] < 0:
            zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_firn, n_index)
            t_theta = geo_utl.get_fresnel_t_p(zenith, n_firn, n_index)
            t_phi = geo_utl.get_fresnel_t_s(zenith, n_firn, n_index)
        else:
            t_theta = 1.
            t_phi = 1.
            zenith_antenna = zenith
        # fold signal with receiving antenna response
        RxAntenna_type = array_of_st[st_id].antenna_surface_lpda
        RxAntenna_orientation = array_of_st[st_id].getAntennaRotation()[ch_id]

        if ch_id == 13:
            RxAntenna_orientation = np.array([np.pi / 2, azimuth, np.pi / 2, -np.pi / 2 + azimuth])
        # print(array_of_st[st_id]._name, np.rad2deg(azimuth))

        VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith_antenna, azimuth,
                                                             *RxAntenna_orientation)

        efield_antenna_factor = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
        power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
        power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0)
        # USAGE OF SURFACE AMPLIFIER IS IMPOSIBLE

        sampling_rate_for_IFFT = 2 * np.max(freq_range)
        #power_spectrum_atRx = np.interp(freq_range_noise, freq_range, power_spectrum_atRx)

        amplifier_s11 = array_of_st[st_id].amp_response_surface
        amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
        amp_response_noise = amplifier_s11['gain'](freq_range_noise) * amplifier_s11['phase'](freq_range_noise)

        ampl_noise = get_noise_figure_IGLU_DRAP(freq_range_noise)

        voltage_trace_atRx = fft.freq2time(power_spectrum_atRx*amp_response,
                                           sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
        galactic_noise_spectrum = print_noise_temperture_from_pygdsm(freq_range_input = freq_range_noise*1e3,
                                                                     antenna_orientation = RxAntenna_orientation)
        galactic_noise_timeTrace = fft.freq2time(galactic_noise_spectrum*amp_response_noise, 2 * np.max(freq_range_noise))

        thermal_noise_spectrum = generate_thermal_noise(freq_range_noise, depth=0, antenna_type = 'lpda')
        thermal_noise_trace = fft.freq2time(thermal_noise_spectrum*amp_response_noise, 2 * np.max(freq_range_noise))

        ampl_noise_trace = fft.freq2time(ampl_noise*amp_response_noise, 2 * np.max(freq_range_noise))
        # check trigger conditions, save trigger info
        trace = galactic_noise_timeTrace + ampl_noise_trace + \
                np.append(voltage_trace_atRx, np.zeros( len(galactic_noise_timeTrace) - len(voltage_trace_atRx) )) \
                + thermal_noise_trace

        time = np.arange(0, 640, 640 / len(trace))
        # if ch_id == 13:
        #     time = np.arange(0, 640, 640 / len(thermal_noise_trace))
        #     #thermal_noise_trace = fft.time2freq(thermal_noise_trace, 2 * np.max(freq_range_noise))
        #     plt.rcParams["figure.figsize"] = (10, 5)
        #     #plt.plot(freq_range_noise, 20*np.log10(thermal_noise_trace ), linewidth=3.0, label = 'After The Amplification' )
        #     plt.plot(time, galactic_noise_timeTrace + ampl_noise_trace + thermal_noise_trace, linewidth=3.0, label='After The Signal Amplification')
        #
        #     #plt.xlim(0, 640)
        #     # plt.ylim(-120, -10)
        #     plt.title('Time Trace of the Noise at the Surface Channel 0', fontsize = 18)
        #     plt.legend(fontsize = 18)
        #     plt.xlabel('Time, [ns]', fontsize = 18)
        #     plt.ylabel('Amplitude, [V]', fontsize=18)
        #     plt.xticks(fontsize=17)
        #     plt.yticks(fontsize=17)
        #     plt.savefig('/Users/alisanozdrina/Desktop/pics_for_work/surface_noise_trace_Gl.png', dpi=100, bbox_inches='tight')
        #     plt.show()

        Event[ev_id].set_trace(ch_id, trace )
    #time needed for trigger signal to pass CATH --> ICg2 station in the air
    travel_time_inAir = array_of_st[st_id].getDistanceToAntenna()[13][0] / (scipy.constants.c / n_air) * 1e9
    # let's use 3.8 ns per m propagation time.
    cable_prop_time = 3.8 * 200

    fiber_prop_time = 5 * 100
    trigger_time = travel_time_inAir + cable_prop_time # take snapshot from the deep channels
    # loop through deep channels: **note maybe start with a single one?

    v_pols = np.array([0,1,2,3,5,6,7,9,10,22,23])
    h_pols = np.array([4,8,11,21])

    deep_channels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23])

    #deep_channels = np.concatenate((v_pols, h_pols), axis=0)
    # has to calculate trigger time ...
    for count, ch_id in enumerate(deep_channels):
        # find travel distance: deep Tx and deep channel coordinate difference --
        if pulsingMode == 'Vpol':
            array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_vpol)
        if pulsingMode == 'Hpol':
            array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_hpol)

        initial_point = cath_site._coordinates_deep_antenna_vpol
        final_point = array_of_st[st_id].getAntennaCoordinate()[ch_id]
        # run analytic ray tracing
        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()
        n = rays.get_number_of_solutions()

        WF = [np.array([]) for i in range(n)]

        travel_time = 0
        dt_channel = 0

        # loop through ray tracing solutions: *start with direct ray
        if n != 0:
            for iS in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(iS)
                solution_type = solution_types[solution_int]

                # save travel distance and time for each solution
                path_length = rays.get_path_length(iS)
                timeOfFlight = rays.get_travel_time(iS)

                if iS == 0:
                    travel_time = rays.get_travel_time(iS)
                    dt_channel = travel_time - trigger_time - delayBW_SurfDeep
                    if dt_channel <=0 :
                        print('Warning! signal reached the deep channel ' + str(ch_id) + ' before the trigger ')
                        print('Try set delayBW_SurfDeep closer to :', travel_time - trigger_time)
                if iS == 1:
                    dt_DnR = rays.get_travel_time(iS) - travel_time

                # find launching and receiving vectors for antenna radiation pattern
                launch_vector = rays.get_launch_vector(iS)
                receive_vector = rays.get_receive_vector(iS)

                zenith = hp.cartesian_to_spherical(*launch_vector)[0]
                azimuth = hp.cartesian_to_spherical(*launch_vector)[1]

                # calculate antenna radiation patter
                # apply cable attenuation on power spectrum
                fft_idl = cath_site.get_pulse_fft()

                if pulsingMode == 'Vpol':
                    antenna_orientation = cath_site.inIce_Tx_vpol_orientation
                    VEL = cath_site.antenna_inIce_Tx_vpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                          *antenna_orientation)
                if pulsingMode == 'Hpol':
                    antenna_orientation = cath_site.inIce_Tx_hpol_orientation
                    VEL = cath_site.antenna_inIce_Tx_hpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                          *antenna_orientation)

                eTheta = VEL['theta'] * (-1j) * fft_idl * freq_range * n_index / (c)
                ePhi = VEL['phi'] * (-1j) * fft_idl * freq_range * n_index / (c)

                eTheta = eTheta / path_length
                ePhi = ePhi / path_length

                # propagate through the ice - apply ice attenuation and distance factors
                attenuation_ice = rays.get_attenuation(iS, freq_range, sampling_rate_for_IFFT)

                zenith = hp.cartesian_to_spherical(*receive_vector)[0]
                azimuth = hp.cartesian_to_spherical(*receive_vector)[1]

                if ch_id in v_pols:
                    RxAntenna_type = array_of_st[st_id].antenna_inIce_vpol
                    RxAntenna_orientation = array_of_st[st_id].getAntennaRotation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                else:
                    RxAntenna_type = array_of_st[st_id].antenna_inIce_hpol
                    RxAntenna_orientation = array_of_st[st_id].getAntennaRotation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                efield_antenna_factor = np.array([VEL['theta'], VEL['phi']])

                #Fold E-field with in ice Rx antennas
                power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
                power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0) * attenuation_ice
                # Add thermal noise
                # arange new array of frequencies to generate 2048 samples of thermal noise
                # freq_range_noise = np.arange(freq_range[0], freq_range[-1],
                #                              (freq_range[-1] - freq_range[0])/1025 )
                thermal_noise_spectrum = generate_thermal_noise(freq_range_noise, depth = -final_point[2])

                #power_spectrum_atRx += thermal_noise_spectrum
                # apply amplifier
                amplifier_s11 = array_of_st[st_id].amp_response_iglu
                amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
                amp_response_noise = amplifier_s11['gain'](freq_range_noise) * amplifier_s11['phase'](freq_range_noise)
                # Add Amplifier noise
                ampl_noise =  get_noise_figure_IGLU_DRAP(freq_range_noise)
                #power_spectrum_atRx += ampl_noise
                # Fiber Link passes signal to the DAQ
                power_spectrum_atRx *= amp_response * fiber_link(freq_range, -final_point[2])
                thermal_noise_spectrum *= amp_response_noise * fiber_link(freq_range_noise, -final_point[2])
                ampl_noise *= amp_response_noise * fiber_link(freq_range_noise, -final_point[2])

                # fold signal with receiving antenna response
                WF[iS] = fft.freq2time(power_spectrum_atRx,
                                       sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
                # thermal_noise = fft.freq2time(thermal_noise_spectrum, sampling_rate_for_IFFT)

                # WF[iS] += thermal_noise
            # Superimpose direct and reflected rays
            WF_DnR_superimposed = superimposeWF(WF, dt_DnR)
            # Shift traces in time with respect to trigger time
            WF_superimposed_shifted = shift_trace(WF_DnR_superimposed, dt_channel, sampling_rate_for_IFFT)

            WF_noise = fft.freq2time(thermal_noise_spectrum, 2 * np.max(freq_range_noise)) + \
                       fft.freq2time(ampl_noise, 2 * np.max(freq_range_noise))

            Event[ev_id].set_trace( ch_id, WF_superimposed_shifted+WF_noise)


# draw diagnostic plots - amplitude of signal in surface and deep channels - 2 plots, time delay between hit time in
plot_surface_array(array_of_st, cath_site, Event, trigg_ch=13)

ev_id = 1
st_id = 0
drawTraceSurfaceArrayRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)


drawFFTSurfaceArrayRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)

drawTraceDeepChannelsRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
                      Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length)

drawFFTDeepChannelsRNO(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
                      Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length)