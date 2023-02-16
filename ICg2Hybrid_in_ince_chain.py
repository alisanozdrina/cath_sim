from IceCube_gen2_radio.IC_hybrid_station import *
from IceCube_gen2_radio.tools import *

#define ice parameters
n_index = 1
c = constants.c * units.m / units.s
n_firn = 1.3

prop = propagation.get_propagation_module('analytic')

# for the Summit station
# ref_index_model = 'greenland_simple'
# ice = medium.get_ice_model(ref_index_model)
# attenuation_model = 'GL1'

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'SP1'


#define detector layout
Num_of_stations=30
array_of_st = np.empty(Num_of_stations, dtype=IC_hybrid_station)
array_of_st=get_ICH_coordinates(Num_of_stations, separation=1750, grid_type='square', 
                        coordinate_center=np.array([0, 0]), station_type='hybrid')

#define CATH coordinates, antennas, antenna orientation, pulser trace - needs separate class  

cath_site = CATH_site(name='cath 1', coordinates=np.array([0,0,3]))

#define a single event - need event class, would contain traces for each channel and timing information
st_id = 0
Event = np.empty(Num_of_stations, dtype=EventTraces) #for now we only work with a single event, but each station has it's own Event object

for st_id in range(0, Num_of_stations):
	ev_id = st_id
	Event[ev_id] = EventTraces(ev_id,array_of_st[st_id]._name)
	#loop through every station **note maybe start with a single one?

		#loop through surface channels: **note maybe start with a single one?
	#ch_id = 0
	for ch_id in range(0,7):
				#find travel distance: surface Tx between surface channels
		array_of_st[st_id].set_tx_coordinates(cath_site._coordinates) 
		travel_distance =  array_of_st[st_id].getDistance_LPDA()[ch_id] #travel distance - array of 16 elements 
				#find angle for radiation pattern
		rad_pattern = array_of_st[st_id].getRelRadiationAngle_LPDA()
				#calculate antenna radition patter
		zenith = rad_pattern[ch_id ][0]
		azimuth =rad_pattern[ch_id ][1]
		antenna_orientation = cath_site.surface_Tx_orientation
		freq_range = cath_site.get_pulse_fft_freq(0)

		VEL = cath_site.antenna_surface_Tx.get_antenna_response_vectorized(freq_range, zenith, -azimuth, *antenna_orientation)
				#propogate through the air - 1/R factor, save propogation time
		eTheta = VEL['theta'] * (-1j) * cath_site.get_pulse_fft(0) * freq_range * n_index / (c)
		ePhi = VEL['phi'] * (-1j) * cath_site.get_pulse_fft(0)  * freq_range * n_index / (c) 

		eTheta = eTheta/travel_distance
		ePhi = ePhi/travel_distance
				#if the wave passes air/ice boundary - correct on fresnel coefficient 
		if zenith <= 0.5 * np.pi:
		    zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_firn, n_index)
		    t_theta = geo_utl.get_fresnel_t_p(zenith, n_firn, n_index)
		    t_phi = geo_utl.get_fresnel_t_s(zenith, n_firn, n_index)
		else:
		    t_theta = 1.
		    t_phi = 1.
		    zenith_antenna = zenith
				#fold signal with receiving antenna responce
		RxAntenna_type = array_of_st[st_id].antenna_surface_lpda
		RxAntenna_orientation = array_of_st[st_id].getLPDA_orientation()[ch_id ]
		RxAntenna_rotation = array_of_st[st_id].getLPDA_rotation()[ch_id ]


		VEL = RxAntenna_type.get_antenna_response_vectorized(freq_range, zenith_antenna, azimuth, 
		                                                  *RxAntenna_orientation, *RxAntenna_rotation)

		efield_antenna_factor = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
		power_spectrum_atRx = efield_antenna_factor * np.array([eTheta, ePhi]) 
		power_spectrum_atRx = np.sum(power_spectrum_atRx, axis=0)
		#apply amplifier 
		amplifier_s11=array_of_st[st_id].amp_response
		amp_response = amplifier_s11['gain'](freq_range) * amplifier_s11['phase'](freq_range)

		sampling_rate_for_IFFT= 2*np.max(freq_range)
		voltage_trace_atRx = fft.freq2time(power_spectrum_atRx*amp_response, sampling_rate_for_IFFT) #sampling rate is twice the length of original pulse freq band..

		Event[ev_id].set_trace(ch_id, voltage_trace_atRx)



			#save amplitude and hit time

		#check trigger conditions, save trigger info 

		#loop through deep channels: **note maybe start with a single one? 

			#find travel distance: deep Tx and deep channel coordinate difference -- 
			# run analytic ray tracing

			#loop through ray tracing solutions: *start with direct ray
				#save travel distance and time for each solution 
				#find launching and receiving vectors for antenna radiation pattern

				#calculate antenna radition patter

				#propogate through the ice - apply ice attenuation and distance factors

				#fold signal with receiving antenna responce

				#save amplitude and hit time

			#if several solutions are within 500ns time window, superimmpose traces in event object
			#check trigger condition py 

#draw diagnostic plots - amplitude of signal in surface and deep channels - 2 plots, time delay between hit time in 

#drawTraceSurfaceArray(array_of_st[st_id]._name, ev_id, Event[ev_id].get_traces(), sampling_rate_for_IFFT, Event[ev_id]._sampling_rate*1e-9, Event[ev_id]._trace_length)

plot_surface_array(array_of_st,cath_site, Event)