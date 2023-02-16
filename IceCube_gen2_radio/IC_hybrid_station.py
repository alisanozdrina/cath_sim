from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.detector import detector
import datetime
from NuRadioReco.modules import sphericalWaveFitter
from NuRadioReco.modules import channelAddCableDelay
from NuRadioReco.modules.io.rno_g import rnogDataReader
from NuRadioReco.utilities import units
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.framework.channel import Channel
from NuRadioReco.detector import antennapattern
from NuRadioMC.SignalGen import emitter
from NuRadioMC.utilities import medium
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.detector.RNO_G import analog_components
import uproot
from scipy import constants
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import scipy.constants
from NuRadioReco.utilities import units, fft
from radiotools import helper as hp
from NuRadioReco.utilities import geometryUtilities as geo_utl


class IC_hybrid_station: #matching description with hybrid station from gen2 proposal
	#The reference design that we use throughout this document combines shallow and hy-
# brid stations (trigger depth of 150 m) on a square grid (see Figure 12) with a spacing of
# 1.75 km between hybrid stations.

	#antenna models from nuradioreco
	antenna_info = antennapattern.AntennaPatternProvider()
	antenna_surface_lpda = antenna_info.load_antenna_pattern('createLPDA_100MHz_InfFirn')
	antenna_inIce_vpol = antenna_info.load_antenna_pattern('RNOG_vpol_4inch_center_n1.73')
	antenna_inIce_hpol = antenna_info.load_antenna_pattern('RNOG_quadslot_v3_air_rescaled_to_n1.74')

	surface_Tx_orientation = np.array([0,0,np.pi/2,0])
	inIce_Tx_vpol_orientation = np.array([0,0,np.pi/2,0])
	inIce_Tx_hpol_orientation = np.array([0,0,np.pi/2,np.pi/2])

	ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])
	#amplifier s11 from nuradioreco
	det = detector.Detector(json_filename='/Users/alisanozdrina/Documents/phys/exp/rno/soft/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_single_station.json')    # detector file
	det.update(datetime.datetime(2018, 10, 1))
	amp_response = analog_components.load_amp_response('rno_surface')  
	
	def __init__(self, name='', coordinates=np.array([0,0,0])):
		self._name = name
		self._coordinates = coordinates
		self.coordinates_Tx = np.array([])

	def set_tx_coordinates(self, tx_coordinates):
		self.coordinates_Tx = tx_coordinates

	def get_tx_coordinates(self):
		return self.coordinates_Tx

	def getLPDAcoordinate(self):
		ch_coord = np.zeros((7,3))
		center = self._coordinates #x,y,z
		x,y,z=0,1,2
		ch_coord[0] = np.array([center[x]-1, center[y]-1, -3])
		ch_coord[1] = np.array([center[x]-1, center[y]+1, -3])
		ch_coord[2] = np.array([center[x]+1, center[y]-1, -3])
		ch_coord[3] = np.array([center[x]+1, center[y]+1, -3])
		ch_coord[4] = np.array([center[x]-2, center[y]-2, -3])
		ch_coord[5] = np.array([center[x]+2, center[y]-2, -3])
		ch_coord[6] = np.array([center[x], center[y]+2, -3])
		return ch_coord

	def getLPDA_orientation(self):
		ch_ori = np.zeros((7,2))
		ch_ori[0] = np.array([np.pi, 0])
		ch_ori[1] = np.array([np.pi, 0])
		ch_ori[2] = np.array([np.pi, 0])
		ch_ori[3] = np.array([np.pi, 0])
		ch_ori[4] = np.array([0, 0])
		ch_ori[5] = np.array([0, 0])
		ch_ori[6] = np.array([0, 0])
		return ch_ori

	def getLPDA_rotation(self):
		ch_rot = np.zeros((7,2))
		ch_rot[0] = np.array([np.pi/2, np.pi/2])
		ch_rot[1] = np.array([np.pi/2, 0])
		ch_rot[2] = np.array([np.pi/2, np.pi/2])
		ch_rot[3] = np.array([np.pi/2, 0])

		ch_rot[4] = np.array([np.pi/2, -np.pi/2])
		ch_rot[5] = np.array([np.pi/2, -np.pi/2])
		ch_rot[6] = np.array([np.pi/2, np.pi])
		return ch_rot

	def getDistance_LPDA(self):
		
		coordinates_LPDA=self.getLPDAcoordinate()
		coordinates_Tx=self.coordinates_Tx
		ch_distances = np.zeros((7,1))
		for i in range(0,7):
			ch_distances[i] = np.linalg.norm(coordinates_Tx-coordinates_LPDA[i])
		return ch_distances

	def getRelRadiationAngle_LPDA(self):
		coordinates_LPDA=self.getLPDAcoordinate()
		coordinates_Tx=self.coordinates_Tx
		ch_RelRadiationAngle = np.zeros((7,2))
		x,y,z=0,1,2
		for i in range(0,7):
			rad_vec = np.array([coordinates_Tx[x] - coordinates_LPDA[i][x],
								coordinates_Tx[y]- coordinates_LPDA[i][y],
								coordinates_Tx[z] -coordinates_LPDA[i][z]]) # directed to Tx
			ch_RelRadiationAngle[i][0], ch_RelRadiationAngle[i][1]  = hp.cartesian_to_spherical(rad_vec[x],
																								rad_vec[y],
																								rad_vec[z])
		return ch_RelRadiationAngle



class IC_shallow_station: #matching description with hybrid station from gen2 proposal
	#The reference design that we use throughout this document combines shallow and hy-
# brid stations (trigger depth of 150 m) on a square grid (see Figure 12) with a spacing of
# 1.75 km between hybrid stations.

	ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])
		
	def __init__(self, name='', coordinates=np.array([0,0,0]), coordinates_Tx = np.array([0,0,0]), 
				 rx_responce_dummy = 0, rx_max_amplitude = np.zeros(7), rx_hitTime = np.zeros(7)):
		self._name = name
		self._coordinates = coordinates
		self._coordinates_Tx = coordinates_Tx
		self._rx_responce_dummy = rx_responce_dummy
		self._rx_max_amplitude = rx_max_amplitude
		self._rx_hitTime = rx_hitTime

class CATH_site: #matching description from gen2 proposal

	def __init__(self, name='', coordinates=np.array([0,0,0])):

		self._name = name
		self._coordinates = coordinates
		self._coordinates_surface_antenna = np.array([coordinates[0],coordinates[1],+10])
		self._coordinates_deep_antenna_vpol = np.array([coordinates[0],coordinates[1],-299])
		self._coordinates_deep_antenna_hpol = np.array([coordinates[0],coordinates[1],-300])
		self.pulse = [np.array([]) for i in range(3)] #0,1 and 2 are surface, vpol and hpol emitters
		self.pulse_fft = [np.array([]) for i in range(3)]
		self.pulse_fft_freq = [np.array([]) for i in range(3)] 

	antenna_info = antennapattern.AntennaPatternProvider()
	antenna_surface_Tx = antenna_info.load_antenna_pattern('bicone_v8_InfAir')
	antenna_inIce_Tx_vpol = antenna_info.load_antenna_pattern('RNOG_vpol_4inch_center_n1.73')
	antenna_inIce_Tx_hpol = antenna_info.load_antenna_pattern('RNOG_quadslot_v3_air_rescaled_to_n1.74')

	surface_Tx_orientation = np.array([0,0,np.pi/2,0])
	inIce_Tx_vpol_orientation = np.array([0,0,np.pi/2,0])
	inIce_Tx_hpol_orientation = np.array([0,0,np.pi/2,np.pi/2])
	# antennas orientation and rotation chosen similarly to https://github.com/nu-radio/NuRadioMC/blob/develop/NuRadioReco/detector/RNO_G/RNO_season_2021.json

	def set_trace(self, index):
		file = uproot.open('/Users/alisanozdrina/Documents/phys/exp/icecube_gen2/cal_tower/data/idl2.root')
		time_ns = file['waveforms_ffts/Wav_1400'].values()[0] * 1e9
		v_V = file['waveforms_ffts/Wav_1400'].values()[1]
		v_V = v_V*1e4 # increase amplitude 1000 times to make up 60bB attenuation 
		N = len(v_V)
		dt = 0.4 #ns
		sampling_rate = 1/dt #GHz
		freq_axis = np.fft.rfftfreq(N, dt)
		fft_idl = np.abs( fft.time2freq(v_V, sampling_rate) )
		self.pulse[index] = v_V
		self.pulse_fft[index] = fft_idl
		self.pulse_fft_freq[index] = freq_axis

	def get_trace(self, index):
		self.set_trace(index)
		return self.pulse[index]
		
	def get_pulse_fft(self, index):
		self.set_trace(index)
		return self.pulse_fft[index]

	def get_pulse_fft_freq(self, index):
		self.set_trace(index)
		return self.pulse_fft_freq[index]

class EventTraces:

	#channel_mapping: ch0-6 surface array, ch7-16 power string from top to bottom, ch16-19 helper string 1, ch19-23 helper string 2
	#surface array map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])
	def __init__(self, ev_num, station_name):
		self.ev_num = ev_num
		self.station_name = station_name
		self._sampling_rate = 3.2e9 #3.2 GHz similar to RNO-g
		self._trace_length = 2048 #samples - corresponds to 640 ns trace length 
		self._num_of_ch = 24 
		self.traces = [np.array([]) for i in range(self._num_of_ch)] 
		self.fft_traces = [np.array([]) for i in range(self._num_of_ch)] 

	def set_trace(self, index, trace):
		self.traces[index] = trace
		self.fft_traces[index] = np.fft.fft(trace)

	def get_trace(self, index):
		return self.traces[index]

	def get_traces(self):
		return self.traces

	def get_fft_trace(self, index):
		return self.fft_traces[index]

	def isTriggered_surface(self, trigger_level=0.045):
		arrays = self.traces
		num_arrays = len(arrays)
		num_matches = 0
		for i in range(num_arrays):
			for j in range(i+1, num_arrays):
				matches = np.logical_and(np.abs(arrays[i]-x) < x, np.abs(arrays[j]-x) < x)
				idx = np.where(matches)[0]
				for k in range(len(idx)-2):
					if idx[k+2] - idx[k] <= 300:
						num_matches += 1
		return num_matches >= 3

