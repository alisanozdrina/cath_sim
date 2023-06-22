
from NuRadioReco.detector import detector
import datetime
from NuRadioReco.detector import antennapattern
from NuRadioReco.detector.RNO_G import analog_components
import numpy as np
from radiotools import helper as hp

class IC_hybrid_station:  # matching description with hybrid station from gen2 proposal
    # The reference design that we use throughout this document combines shallow and hy-
    # brid stations (trigger depth of 150 m) on a square grid (see Figure 12) with a spacing of
    # 1.75 km between hybrid stations.

    # antenna models from nuradioreco
    antenna_info = antennapattern.AntennaPatternProvider()
    antenna_surface_lpda = antenna_info.load_antenna_pattern('createLPDA_100MHz_InfFirn')
    antenna_inIce_vpol = antenna_info.load_antenna_pattern('RNOG_vpol_4inch_center_n1.73')
    antenna_inIce_hpol = antenna_info.load_antenna_pattern('RNOG_quadslot_v2_n1.74')

    # surface_Tx_orientation = np.array([0,0,np.pi/2,0])
    # inIce_Tx_vpol_orientation = np.array([0,0,np.pi/2,0])
    # inIce_Tx_hpol_orientation = np.array([0,0,np.pi/2,np.pi/2])

    ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])
    # amplifier s11 from nuradioreco
    det = detector.Detector(
        json_filename='/Users/alisanozdrina/Documents/phys/exp/rno/soft/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_single_station.json')  # detector file
    det.update(datetime.datetime(2018, 10, 1))
    amp_response_surface = analog_components.load_amp_response('rno_surface', temp=220)
    amp_response_iglu = analog_components.load_amp_response('iglu')

    def __init__(self, name='', coordinates=np.array([0, 0, 0])):
        self._name = name
        self._coordinates = coordinates
        self.coordinates_Tx = np.array([])

    def set_tx_coordinates(self, tx_coordinates):
        self.coordinates_Tx = tx_coordinates

    def get_tx_coordinates(self):
        return self.coordinates_Tx

    def getAntennaCoordinate(self):
        ch_coord = np.zeros((24, 3))
        center = self._coordinates  # x,y,z
        x, y, z = 0, 1, 2
        ch_coord[0] = np.array([center[x] - 1, center[y] - 1, 1])
        ch_coord[1] = np.array([center[x] - 1, center[y] + 1, -3])
        ch_coord[2] = np.array([center[x] + 1, center[y] - 1, -3])
        ch_coord[3] = np.array([center[x] + 1, center[y] + 1, -3])
        ch_coord[4] = np.array([center[x] - 2, center[y] - 2, -3])
        ch_coord[5] = np.array([center[x] + 2, center[y] - 2, -3])
        ch_coord[6] = np.array([center[x], center[y] + 2, -3])
        # vpols
        # phased array
        ch_coord[7] = np.array([center[x], center[y] + 20, -10])
        ch_coord[8] = np.array([center[x], center[y] + 20, -55])
        ch_coord[9] = np.array([center[x], center[y] + 20, -100])
        ch_coord[10] = np.array([center[x], center[y] + 20, -147])
        ch_coord[11] = np.array([center[x], center[y] + 20, -148])
        ch_coord[12] = np.array([center[x], center[y] + 20, -149])
        ch_coord[13] = np.array([center[x], center[y] + 20, -150])
        # helper string 1
        ch_coord[14] = np.array([center[x] + 17.3, center[y] - 10, -148])
        ch_coord[15] = np.array([center[x] + 17.3, center[y] - 10, -149])
        # helper string 2
        ch_coord[16] = np.array([center[x] - 17.3, center[y] - 10, -148])
        ch_coord[17] = np.array([center[x] - 17.3, center[y] - 10, -149])
        # h pols
        # phased array
        ch_coord[18] = np.array([center[x], center[y] + 20, -145])
        ch_coord[19] = np.array([center[x], center[y] + 20, -146])
        # helper string 1
        ch_coord[20] = np.array([center[x] + 17.3, center[y] - 10, -147])
        ch_coord[21] = np.array([center[x] + 17.3, center[y] - 10, -149])
        # helper string 2
        ch_coord[22] = np.array([center[x] - 17.3, center[y] - 10, -147])
        ch_coord[23] = np.array([center[x] - 17.3, center[y] - 10, -149])
        return ch_coord

    def getnIceVpol_orientation_rotation(self):
        ch_ori = np.zeros((10, 4))
        for i in range(0, 9):
            ch_ori[i] = np.array([0, 0, np.pi / 2, 0])
        return ch_ori

    def getnIceHpol_orientation_rotation(self):
        ch_ori = np.zeros((6, 4))
        for i in range(0, 6):
            ch_ori[i] = np.array([0, 0, np.pi / 2, np.pi / 2])
        return ch_ori

    def getAntenna_orientation(self):
        ch_ori = np.zeros((24, 4))
        ch_ori[0] = np.array([np.pi / 2, np.pi / 2, 0, 0])
        # ch_ori[0] = np.array([np.pi, 0, np.pi/2, np.pi/2])#point straight down
        ch_ori[1] = np.array([np.pi, 0, np.pi / 2, 0])
        ch_ori[2] = np.array([np.pi, 0, np.pi / 2, np.pi])  # point straight down
        ch_ori[3] = np.array([np.pi, 0, np.pi / 2, 0])
        # ch_ori[4] = np.array([0, 0, np.pi/2, -np.pi/2])
        # ch_ori[5] = np.array([0, 0, np.pi/2, -np.pi/2])
        # ch_ori[6] = np.array([0, 0, np.pi/2, -np.pi/2])
        ch_ori[4] = np.array([0, 0, np.pi / 2, -np.pi / 2])
        ch_ori[5] = np.array([0, 0, np.pi / 2, np.pi / 2])
        ch_ori[6] = np.array([0, 0, np.pi / 2, 0])
        # vpols
        # phased array
        ch_ori[7] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[8] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[9] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[10] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[11] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[12] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[13] = np.array([0, 0, np.pi / 2, 0])
        # helper string 1
        ch_ori[14] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[15] = np.array([0, 0, np.pi / 2, 0])
        # helper string 2
        ch_ori[16] = np.array([0, 0, np.pi / 2, 0])
        ch_ori[17] = np.array([0, 0, np.pi / 2, 0])
        # h pols
        # phased array
        ch_ori[18] = np.array([0, 0, np.pi / 2, np.pi / 2])
        ch_ori[19] = np.array([0, 0, np.pi / 2, np.pi / 2])
        # helper string 1
        ch_ori[20] = np.array([0, 0, np.pi / 2, np.pi / 2])
        ch_ori[21] = np.array([0, 0, np.pi / 2, np.pi / 2])
        # helper string 2
        ch_ori[22] = np.array([0, 0, np.pi / 2, np.pi / 2])
        ch_ori[23] = np.array([0, 0, np.pi / 2, np.pi / 2])
        return ch_ori

    def getDistanceToAntenna(self):

        coordinates_antenna = self.getAntennaCoordinate()
        coordinates_Tx = self.coordinates_Tx
        num_of_ch = len(coordinates_antenna)
        ch_distances = np.zeros((num_of_ch, 1))
        for i in range(0, num_of_ch):
            ch_distances[i] = np.linalg.norm(coordinates_Tx - coordinates_antenna[i])
        return ch_distances

    def getRelRadiationAngle(self):
        coordinates_antenna = self.getAntennaCoordinate()
        coordinates_Tx = self.coordinates_Tx
        num_of_ch = len(coordinates_antenna)
        ch_RelRadiationAngle = np.zeros((num_of_ch, 2))
        x, y, z = 0, 1, 2
        for i in range(0, num_of_ch):
            rad_vec = np.array([coordinates_Tx[x] - coordinates_antenna[i][x],
                                coordinates_Tx[y] - coordinates_antenna[i][y],
                                coordinates_Tx[z] - coordinates_antenna[i][z]])  # directed to Tx
            ch_RelRadiationAngle[i][0], ch_RelRadiationAngle[i][1] = hp.cartesian_to_spherical(rad_vec[x],
                                                                                               rad_vec[y],
                                                                                               rad_vec[z])
        return ch_RelRadiationAngle


class IC_shallow_station:  # matching description with hybrid station from gen2 proposal
    # The reference design that we use throughout this document combines shallow and hy-
    # brid stations (trigger depth of 150 m) on a square grid (see Figure 12) with a spacing of
    # 1.75 km between hybrid stations.

    ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])

    def __init__(self, name='', coordinates=np.array([0, 0, 0]), coordinates_Tx=np.array([0, 0, 0]),
                 rx_responce_dummy=0, rx_max_amplitude=np.zeros(7), rx_hitTime=np.zeros(7)):
        self._name = name
        self._coordinates = coordinates
        self._coordinates_Tx = coordinates_Tx
        self._rx_responce_dummy = rx_responce_dummy
        self._rx_max_amplitude = rx_max_amplitude
        self._rx_hitTime = rx_hitTime



