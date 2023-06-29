from IceCube_gen2_radio.IC_hybrid_station import *
from IceCube_gen2_radio.tools import *
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

delayBW_SurfDeep = 0.8 * 1e3  # ns

# define detector layout
Num_of_stations = 2
array_of_st = np.empty(Num_of_stations, dtype=IC_hybrid_station)

array_of_st[0] = IC_hybrid_station('ICR 1', np.array([-850, 0, 0]))
array_of_st[1] = IC_hybrid_station('ICR 2', np.array([+850, 0, 0]))

# array_of_st = get_ICH_coordinates(Num_of_stations, separation=1250, grid_type='square',
#                                   coordinate_center=np.array([0, 0]), station_type='hybrid')

# define CATH coordinates, antennas, antenna orientation, pulser trace - needs separate class
#cath_site = CATH(name='cath 1', coordinates=np.array([-928.0, 1682.8, 5]))

cath_site = CATH(name='cath 1', coordinates=np.array([0, 0, 5]))
freq_range = cath_site.get_pulse_fft_freq()

# define a single event - need event class, would contain traces for each channel and timing information
Event = np.empty(Num_of_stations,
                 dtype=EventTrace)  # for now, we only work with a single event, but each station has it's own Event object

##################################
# Pick CATH Pulsing Mode:
# pulsingMode = Vpol and Hpol # maybe will add CW mode in the future
pulsingMode = 'Vpol'
#################################

# loop through every station
for st_id in range(0, Num_of_stations):
    ev_id = st_id
    Event[ev_id] = EventTrace(ev_id, array_of_st[st_id]._name)

    # loop through surface channels:
    for ch_id in range(0, 7):
        # find travel distance: surface Tx between surface channels

        array_of_st[st_id].set_tx_coordinates(cath_site._coordinates_surface_antenna)
        travel_distance = array_of_st[st_id].getDistanceToAntenna()[ch_id][0]  # travel distance - array of 16 elements
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
        RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]

        if ch_id == 0:
            RxAntenna_orientation = np.array([np.pi / 2, azimuth, np.pi / 2, -np.pi / 2 + azimuth])
        # print(array_of_st[st_id]._name, np.rad2deg(azimuth))

        VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith_antenna, azimuth,
                                                             *RxAntenna_orientation)

        efield_antenna_factor = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
        power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
        power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0)
        # USAGE OF SURFACE AMPLIFIER IS IMPOSIBLE
        # amplifier_s11=array_of_st[st_id].amp_response_surface
        # amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
        sampling_rate_for_IFFT = 2 * np.max(freq_range)
        voltage_trace_atRx = fft.freq2time(power_spectrum_atRx,
                                           sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..

        # check trigger conditions, save trigger info
        Event[ev_id].set_trace(ch_id, voltage_trace_atRx)
    #time needed for trigger signal to pass CATH --> ICg2 station in the air
    travel_time_inAir = array_of_st[st_id].getDistanceToAntenna()[0][0] / (scipy.constants.c / n_air) * 1e9
    # let's use 3.8 ns per m propagation time.
    cable_prop_time = 3.8 * 350

    fiber_prop_time = 5 * 150
    trigger_time = travel_time_inAir + cable_prop_time # take snapshot from the deep channels
    # loop through deep channels: **note maybe start with a single one?

    v_pols = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    h_pols = np.array([18, 19, 20, 21, 22, 23])

    deep_channels = np.concatenate((v_pols, h_pols), axis=0)
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
                    RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                else:
                    RxAntenna_type = array_of_st[st_id].antenna_inIce_hpol
                    RxAntenna_orientation = array_of_st[st_id].getAntenna_orientation()[ch_id]
                    VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith, azimuth,
                                                                         *RxAntenna_orientation)
                efield_antenna_factor = np.array([VEL['theta'], VEL['phi']])

                #Fold E-field with in ice Rx antennas
                power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi])
                power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0) * attenuation_ice
                # Add thermal noise
                thermal_noise_spectrum = generate_thermal_noise(freq_range, depth = -final_point[2])

                #power_spectrum_atRx += thermal_noise_spectrum
                # apply amplifier
                amplifier_s11 = array_of_st[st_id].amp_response_iglu
                amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)
                # Add Amplifier noise
                ampl_noise =  get_noise_figure_IGLU_DRAP(freq_range)
                #power_spectrum_atRx += ampl_noise
                # Fiber Link passes signal to the DAQ
                power_spectrum_atRx *= amp_response * fiber_link(freq_range, -final_point[2])
                thermal_noise_spectrum *= amp_response * fiber_link(freq_range, -final_point[2])
                ampl_noise *= amp_response * fiber_link(freq_range, -final_point[2])

                # fold signal with receiving antenna response
                WF[iS] = fft.freq2time(power_spectrum_atRx,
                                       sampling_rate_for_IFFT)  # sampling rate is twice the length of original pulse freq band..
                # thermal_noise = fft.freq2time(thermal_noise_spectrum, sampling_rate_for_IFFT)
                # WF[iS] += thermal_noise
            # Superimpose direct and reflected rays
            WF_DnR_superimposed = superimposeWF(WF, dt_DnR)
            # Shift traces in time with respect to trigger time
            WF_superimposed_shifted = shift_trace(WF_DnR_superimposed, dt_channel, sampling_rate_for_IFFT)
            WF_noise = fft.freq2time(thermal_noise_spectrum, sampling_rate_for_IFFT) + fft.freq2time(ampl_noise, sampling_rate_for_IFFT)
            # plt.plot(WF_noise)
            # plt.show()
            Event[ev_id].set_trace( ch_id, WF_superimposed_shifted)


# draw diagnostic plots - amplitude of signal in surface and deep channels - 2 plots, time delay between hit time in
plot_surface_array(array_of_st, cath_site, Event)

ev_id = 1
st_id = 0
drawTraceSurfaceArray(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)

#
# plt.plot(Event[st_id].get_trace(8))
# plt.title("Rx signal without surface amplifier, ch 0 - LPDA 1 m above the snow")
# plt.ylabel('Voltage, [V]')
# plt.xlabel('Time, [ns]')
# plt.tight_layout()
# plt.show()
drawTraceDeepChannels(array_of_st[st_id]._name, 0, Event[ev_id].get_traces(), sampling_rate_for_IFFT,
                      Event[ev_id]._sampling_rate * 1e-9, Event[ev_id]._trace_length)

