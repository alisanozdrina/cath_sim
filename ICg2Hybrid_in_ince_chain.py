from IceCube_gen2_radio.IC_hybrid_station import *
from IceCube_gen2_radio.tools import *
import scipy

# define ice parameters
n_index = 1
c = constants.c * units.m / units.s
n_firn = 1.3
n_air = 1.00029
n_ice = 1.74
prop = propagation.get_propagation_module('analytic')

# for the Summit station
# ref_index_model = 'greenland_simple'
# ice = medium.get_ice_model(ref_index_model)
# attenuation_model = 'GL1'

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'SP1'
rays = prop(ice, attenuation_model,
            n_frequencies_integration=25,
            n_reflections=0)

delayBW_SurfDeep = 3.05 * 1e3  # ns
delays = []
ch_del = []
# define detector layout
Num_of_stations = 2
# array_of_st = np.empty(Num_of_stations, dtype=IC_hybrid_station)
array_of_st = get_ICH_coordinates(Num_of_stations, separation=1750, grid_type='square',
                                  coordinate_center=np.array([0, 0]), station_type='hybrid')

# define CATH coordinates, antennas, antenna orientation, pulser trace - needs separate class

cath_site = CATH_site(name='cath 1', coordinates=np.array([0, -875, 5]))
freq_range = cath_site.get_pulse_fft_freq()

# define a single event - need event class, would contain traces for each channel and timing information
st_id = 1
Event = np.empty(Num_of_stations,
                 dtype=EventTraces)  # for now, we only work with a single event, but each station has it's own Event object

# for st_id in range(0, Num_of_stations):
for st_id in range(0, 2):

    ev_id = st_id

    Event[ev_id] = EventTraces(ev_id, array_of_st[st_id]._name)
    # loop through every station **note maybe start with a single one?

    # loop through surface channels: **note maybe start with a single one?
    # ch_id = 0
    for ch_id in range(0, 7):
        # find travel distance: surface Tx between surface channels

        array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_surface_antenna)
        travel_distance = array_of_st[st_id].getDistanceToAntenna()[ch_id][0]  # travel distance - array of 16 elements
        travel_time_inAir = travel_distance / (scipy.constants.c / n_air) * 1e9

        # find angle for radiation pattern
        rad_pattern = array_of_st[st_id].getRelRadiationAngle()
        # calculate antenna radiation patter
        zenith = rad_pattern[ch_id][0]
        azimuth = rad_pattern[ch_id][1]

        antenna_orientation = cath_site.surface_Tx_orientation

        VEL = cath_site.antenna_surface_Tx.get_antenna_response_vectorized(freq_range, zenith, -azimuth,
                                                                           *antenna_orientation)
        # propagate through the air - 1/R factor, save propagation time
        eTheta = VEL['theta'] * (-1j) * cath_site.get_pulse_fft() * freq_range * n_index / c
        ePhi = VEL['phi'] * (-1j) * cath_site.get_pulse_fft() * freq_range * n_index / c

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
        RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]

        if ch_id == 0:
            RxAntenna_orientation = np.array([np.pi / 2, azimuth, np.pi / 2, -np.pi / 2 + azimuth])
        # print(array_of_st[st_id]._name, np.rad2deg(azimuth))

        VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith_antenna, azimuth,
                                                             *RxAntenna_orientation)

        efield_antenna_factor = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
        power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
        power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0)
        # apply amplifier
        # amplifier_s11=array_of_st[st_id].amp_response_surface
        # amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
        amp_response = 1
        sampling_rate_for_IFFT = 2 * np.max(freq_range)
        voltage_trace_atRx = fft.freq2time(power_spectrum_atRx * amp_response,
                                           sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..

        # check trigger conditions, save trigger info
        Event[ev_id].set_trace(ch_id, voltage_trace_atRx)

    hit_time = np.argmax(Event[st_id].get_trace(0) > 3 * 0.015) * 0.4  # ns

    trigger_time = travel_time_inAir + hit_time  # ch1 trigger station. take snapshot from the deep channels
    # loop through deep channels: **note maybe start with a single one?

    v_pols = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    # has to calculate trigger time ...

    for count, ch_id in enumerate(v_pols):

        # find travel distance: deep Tx and deep channel coordinate difference --
        array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_vpol)

        initial_point = cath_site._coordinates_deep_antenna_vpol
        final_point = array_of_st[st_id].getAntennaCoordinate()[ch_id]
        # print(ch_id, initial_point , final_point)
        # run analytic ray tracing

        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()
        n = rays.get_number_of_solutions()

        WF = [np.array([]) for i in range(n)]
        delta_t = np.zeros(n)
        travel_time = 0
        delta_time = 0

        # loop through ray tracing solutions: *start with direct ray
        if n != 0:
            # iS = 0
            for iS in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(iS)
                solution_type = solution_types[solution_int]

                # save travel distance and time for each solution
                path_length = rays.get_path_length(iS)

                timeOfFlight = rays.get_travel_time(iS)
                # print(ch_id, iS, final_point[2], '  ', np.round(path_length,2), np.round(timeOfFlight,2))

                # delays.append( trigger_time-timeOfFlight )
                # ch_del.append(ch_id)

                # print(ch_id, trigger_time-timeOfFlight)

                if (travel_time == 0):
                    travel_time = rays.get_travel_time(iS)
                else:
                    delta_t[iS] = travel_time - rays.get_travel_time(iS)

                delay_ch = travel_time - trigger_time - delayBW_SurfDeep
                delta_t[iS] += delay_ch
                # print(delay_ch)
                # print(ch_id, final_point[2], iS, delta_t[iS])
                # find launching and receiving vectors for antenna radiation pattern
                launch_vector = rays.get_launch_vector(iS)
                receive_vector = rays.get_receive_vector(iS)

                zenith = hp.cartesian_to_spherical(*launch_vector)[0]
                azimuth = hp.cartesian_to_spherical(*launch_vector)[1]
                antenna_orientation = cath_site.inIce_Tx_vpol_orientation
                # calculate antenna radiation patter
                # apply cable attenuation on power spectrum

                # fft_idl= cath_site.get_pulse_fft() * cable_attenuation(freq_range, -cath_site._coordinates_deep_antenna_vpol[2])
                fft_idl = cath_site.get_pulse_fft() * fiber_link(freq_range,
                                                                 -cath_site._coordinates_deep_antenna_vpol[2])

                VEL = cath_site.antenna_inIce_Tx_vpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                      *antenna_orientation)

                eTheta = VEL['theta'] * (-1j) * fft_idl * freq_range * n_index / (c)
                ePhi = VEL['phi'] * (-1j) * fft_idl * freq_range * n_index / (c)

                eTheta = eTheta / path_length
                ePhi = ePhi / path_length

                # propagate through the ice - apply ice attenuation and distance factors
                attenuation_ice = rays.get_attenuation(iS, freq_range, sampling_rate_for_IFFT)

                RxAntenna_type = array_of_st[st_id].antenna_inIce_vpol
                RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]
                zenith = hp.cartesian_to_spherical(*receive_vector)[0]
                azimuth = hp.cartesian_to_spherical(*receive_vector)[1]

                VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                     *RxAntenna_orientation)

                efield_antenna_factor = np.array([VEL['theta'], VEL['phi']])
                power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])

                power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0) * attenuation_ice
                # apply amplifier
                amplifier_s11 = array_of_st[st_id].amp_response_iglu
                amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)

                power_spectrum_atRx = power_spectrum_atRx * amp_response * cable_attenuation(freq_range,
                                                                                             -final_point[2])
                # fold signal with receiving antenna response
                WF[iS] = fft.freq2time(power_spectrum_atRx,
                                       sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
            # voltage_trace_atRx = fft.freq2time(power_spectrum_atRx, sampling_rate_for_IFFT) #sampling rate is twice the length of original pulse freq band..

            # print(hit_time)
            # print(delta_t)
            Event[ev_id].set_trace(ch_id, superimposeWF(WF, delta_t))

        # save amplitude and hit time

        # if several solutions are within 500ns time window, superimpose traces in event object
        # check trigger condition py
    # hpols:
    # fairing from V pol emitter!!
    h_pols = np.array([18, 19, 20, 21, 22, 23])

    WF = [np.array([]) for i in range(n)]
    delta_t = np.zeros(n)
    travel_time = 0
    delta_time = 0

    for count, ch_id in enumerate(h_pols):
        array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_deep_antenna_vpol)
        initial_point = cath_site._coordinates_deep_antenna_vpol
        final_point = array_of_st[st_id].getAntennaCoordinate()[ch_id]
        # run analytic ray tracing

        rays.set_start_and_end_point(initial_point, final_point)
        rays.find_solutions()
        n = rays.get_number_of_solutions()
        # print('num of solutions and ch_id:', n, ch_id)
        # loop through ray tracing solutions: *start with direct ray

        if n != 0:
            for iS in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(iS)
                solution_type = solution_types[solution_int]

                path_length = rays.get_path_length(iS)

                timeOfFlight = rays.get_travel_time(iS)
                # print(ch_id, path_length, timeOfFlight)

                # delays.append( trigger_time-timeOfFlight )
                # ch_del.append(ch_id)

                # print(ch_id, trigger_time-timeOfFlight)

                if (travel_time == 0):
                    travel_time = rays.get_travel_time(iS)
                else:
                    delta_t[iS] = travel_time - rays.get_travel_time(iS)

                delay_ch = travel_time - trigger_time - delayBW_SurfDeep
                delta_t[iS] += delay_ch

                # print('direct solution exist for ch & st_id', ch_id, st_id)
                # save travel distance and time for each solution

                # find launching and receiving vectors for antenna radiation pattern
                launch_vector = rays.get_launch_vector(iS)
                receive_vector = rays.get_receive_vector(iS)

                zenith = hp.cartesian_to_spherical(*launch_vector)[0]
                azimuth = hp.cartesian_to_spherical(*launch_vector)[1]
                antenna_orientation = cath_site.inIce_Tx_vpol_orientation

                fft_idl = cath_site.get_pulse_fft() * cable_attenuation(freq_range,
                                                                        -cath_site._coordinates_deep_antenna_hpol[2])
                VEL = cath_site.antenna_inIce_Tx_vpol.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                                      *antenna_orientation)

                eTheta = VEL['theta'] * (-1j) * fft_idl * freq_range * n_index / (c)
                ePhi = VEL['phi'] * (-1j) * fft_idl * freq_range * n_index / (c)

                eTheta = eTheta / path_length
                ePhi = ePhi / path_length

                attenuation_ice = rays.get_attenuation(iS, freq_range, sampling_rate_for_IFFT)

                RxAntenna_type = array_of_st[st_id].antenna_inIce_hpol
                RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]
                zenith = hp.cartesian_to_spherical(*receive_vector)[0]
                azimuth = hp.cartesian_to_spherical(*receive_vector)[1]

                VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                     *RxAntenna_orientation)

                efield_antenna_factor = np.array([VEL['theta'], VEL['phi']])
                power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])

                power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0) * attenuation_ice
                # apply amplifier
                amplifier_s11 = array_of_st[st_id].amp_response_iglu
                amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)

                power_spectrum_atRx = power_spectrum_atRx * amp_response * cable_attenuation(freq_range,
                                                                                             -final_point[2])
                # fold signal with receiving antenna responce
                WF[iS] = fft.freq2time(power_spectrum_atRx,
                                       sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
            # voltage_trace_atRx = fft.freq2time(power_spectrum_atRx, sampling_rate_for_IFFT) #sampling rate is twice the length of original pulse freq band..

            Event[ev_id].set_trace(ch_id, superimposeWF(WF, delta_t))

# draw diagnostic plots - amplitude of signal in surface and deep channels - 2 plots, time delay between hit time in

plot_surface_array(array_of_st, cath_site, Event)

ev_id = 0
st_id = 0
# drawTraceSurfaceArray(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)


# plt.plot(Event[st_id].get_trace(0))
# plt.title("Rx signal without surface amplifier, ch 0 - LPDA 1 m above the snow")
# plt.ylabel('Voltage, [V]')
# plt.xlabel('Time, [ns]')
# plt.tight_layout()
# plt.show()
# drawTraceDeepChannels(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
#                       Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length)

# programm timing
