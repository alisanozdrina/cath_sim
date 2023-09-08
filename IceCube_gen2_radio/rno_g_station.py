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


class rno_g_st:  # matching description with hybrid station from gen2 proposal

    up_faced_antenna_ch = np.array([13, 16, 19])
    down_faced_antenna_ch = np.array([12, 14, 15, 17, 18, 20])

    _surf_ch_num = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20])
    _deep_ch_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23])

    num_of_ch = len(_surf_ch_num)

    num_of_ch_deep = 1  # len(_deep_ch_num)

    _ch_map_surface = np.array(['ch12 down', 'ch13 up', 'ch14 down', 'ch15 down', 'ch16 up', 'ch17 down',
                                'ch18 down', 'ch19 up', 'ch20 down'])

    _pathToJson = "/Users/alisanozdrina/Documents/phys/exp/rno/soft/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2021.json"
    _det = detector.Detector(json_filename=_pathToJson)
    _det.update(datetime.datetime(2022, 10, 1))  # taken from the example

    def __init__(self, name='', coordinates=np.array([0, 0, 0]), coordinates_Tx=np.array([0, 0, 0]),
                 rx_max_amplitude=np.zeros(9), rx_hitTime=np.zeros(9)):

        self._name = name
        self._coordinates = coordinates
        self._coordinates_Tx = coordinates_Tx
        self._coordinates_TxInIce = np.array([coordinates_Tx[0], coordinates_Tx[1], coordinates_Tx[2] - 300])
        self._rx_max_amplitude = rx_max_amplitude
        self._rx_hitTime = rx_hitTime

    def set_tx_coordinates(self, tx_coordinates):
        self.coordinates_Tx = tx_coordinates

    def getLPDAcoordinate(self):
        ch_coord = np.zeros((9, 3))
        center = self._coordinates  # x,y,z
        x, y, z = 0, 1, 2

        f = open(self._pathToJson)
        data = json.load(f)

        for i, ch in enumerate(self._surf_ch_num):
            ch_coord[i] = np.array([center[x] + data['channels'][str(ch)]['ant_position_x'],
                                    center[y] + data['channels'][str(ch)]['ant_position_y'],
                                    center[z] + data['channels'][str(ch)]['ant_position_z']])
        return ch_coord

    def getInIceAnt_coordinate(self):

        ch_coord = np.zeros((self.num_of_ch_deep, 3))  # self._deep_ch_num
        center = self._coordinates  # x,y,z
        x, y, z = 0, 1, 2

        f = open(self._pathToJson)
        data = json.load(f)

        for i, ch in enumerate(self._deep_ch_num[:1]):  # self._deep_ch_num
            ch_coord[i] = np.array([center[x] + data['channels'][str(ch)]['ant_position_x'],
                                    center[y] + data['channels'][str(ch)]['ant_position_y'],
                                    center[z] + data['channels'][str(ch)]['ant_position_z']])
        return ch_coord

    def getInIceAnt_orientation(self):

        ch_ori = np.zeros((self.num_of_ch_deep, 4))

        for i, ch in enumerate(self._deep_ch_num[:1]):  # self._deep_ch_num
            ch_ori[i] = np.array(self._det.get_antenna_orientation(11, ch))
        return ch_ori

    def getLPDA_orientation(self):
        ch_ori = np.zeros((9, 4))
        for i, ch in enumerate(self._surf_ch_num):
            ch_ori[i] = np.array(self._det.get_antenna_orientation(11, ch))
        return ch_ori

    def getDistance_LPDA(self):

        coordinates_LPDA = self.getLPDAcoordinate()
        coordinates_Tx = self._coordinates_Tx
        ch_distances = np.zeros((9, 1))
        for i in range(0, 9):
            ch_distances[i] = np.linalg.norm(coordinates_Tx - coordinates_LPDA[i])
        return ch_distances

    def getDistance_InIceAnt(self):

        coordinates_InIceAnt = self.getInIceAnt_coordinate()
        coordinates_Tx = self._coordinates_TxInIce
        ch_distances = np.zeros((self.num_of_ch_deep, 1))
        for i in range(0, self.num_of_ch_deep):
            ch_distances[i] = np.linalg.norm(coordinates_Tx - coordinates_InIceAnt[i])
        return ch_distances

    def getRelRadiationAngle_LPDA(self):
        coordinates_LPDA = self.getLPDAcoordinate()
        coordinates_Tx = self._coordinates_Tx
        ch_RelRadiationAngle = np.zeros((9, 2))
        x, y, z = 0, 1, 2
        for i in range(0, self.num_of_ch):
            rad_vec = np.array([coordinates_Tx[x] - coordinates_LPDA[i][x],
                                coordinates_Tx[y] - coordinates_LPDA[i][y],
                                coordinates_Tx[z] - coordinates_LPDA[i][z]])  # directed to Tx
            ch_RelRadiationAngle[i][0], ch_RelRadiationAngle[i][1] = hp.cartesian_to_spherical(rad_vec[x],
                                                                                               rad_vec[y],
                                                                                               rad_vec[z])
        return ch_RelRadiationAngle

    def isLPDATriggered(self):

        trig_value = 50 * 1e-3
        numHitCh = 0
        hitCh = np.zeros(9)

        for ch in range(0, len(self._rx_max_amplitude)):

            if self._rx_max_amplitude[ch] > trig_value:
                numHitCh += 1
                hitCh[ch] = self._rx_hitTime[ch]

        if (numHitCh >= 3):
            hitCh.sort()
            # at least three hits within 100ns
            trig_condition = abs(hitCh[::-1][2] - hitCh[::-1][0]) < 100
            if trig_condition:
                return True
            else:
                return False
