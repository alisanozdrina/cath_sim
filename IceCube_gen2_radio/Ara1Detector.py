import numpy as np
from NuRadioReco.detector import detector
from NuRadioReco.detector import antennapattern
import json
import numpy as np
from radiotools import helper as hp

class Ara1Detector:
    coordinates_Tx: np.array([])

    _deep_ch_num = np.arange(0,16,1)

    _pathToJson = "/Users/alisanozdrina/Documents/phys/exp/rno/soft/NuRadioMC/NuRadioReco/detector/ARA/ara_detector_db.json"
    _det = detector.Detector(json_filename=_pathToJson)

    def __init__(self, coordinates: object = np.array([0, 0, 0])):
        self._coordinates = coordinates
        self.coordinates_Tx = np.array([])
    
    def set_tx_coordinates(self, tx_coordinates):
        self.coordinates_Tx = tx_coordinates

    def get_tx_coordinates(self):
        return self.coordinates_Tx

    def get_antenna_info(self, ch_id):
        antenna_info = antennapattern.AntennaPatternProvider()

        vpols = antenna_info.load_antenna_pattern('XFDTD_Vpol_CrossFeed_150mmHole_n1.78')
        hpols = antenna_info.load_antenna_pattern('XFDTD_Hpol_150mmHole_n1.78')

        if ch_id < 8:
            return vpols
        else:
            return hpols

    def getAntCoordinate(self):
        ch_coord = np.zeros((len(self._deep_ch_num), 3))  # self._deep_ch_num
        center = self._coordinates  # x,y,z
        x, y, z = 0, 1, 2

        f = open(self._pathToJson)
        data = json.load(f)

        for i, ch in enumerate(self._deep_ch_num[:]):  # self._deep_ch_num
            ch_coord[i] = np.array([center[x] + data['channels'][str(ch)]['ant_position_x'],
                                    center[y] + data['channels'][str(ch)]['ant_position_y'],
                                    center[z] + data['channels'][str(ch)]['ant_position_z']])
        f.close()
        return ch_coord

    def getAntOrientationRotation(self,ch_id=0):

        #ch_ori = np.zeros((len(self._deep_ch_num), 4))
        #
        # f = open(self._pathToJson)
        # data = json.load(f)
        #
        # for i,ch in enumerate(self._deep_ch_num): # self._deep_ch_num
        #     ch_ori[i] = np.array( [data['channels'][str(ch)]['ant_orientation_theta'],
        #                           data['channels'][str(ch)]['ant_orientation_phi'],
        #                           data['channels'][str(ch)]['ant_rotation_theta'],
        #                           data['channels'][str(ch)]['ant_rotation_phi']])
        # f.close()
        # return ch_ori

        if ch_id < 8:
            ch_ori = np.array([0, 0, np.pi / 2, 0])
            return ch_ori
        else:
            ch_ori = np.array([0, 0, np.pi / 2, np.pi / 2])
            return ch_ori



    def getDistance_InIceAnt(self):

        distances = np.zeros((len(self._deep_ch_num)))
        for i in range(0, len(self._deep_ch_num) ):
            distances[i] = np.linalg.norm(self.coordinates_Tx - self.getAntCoordinate()[i])
        return distances

    def getRadiationAngle(self):

        radiationAngle = np.zeros((len(self._deep_ch_num), 2))
        x, y, z = 0, 1, 2
        for i in range(0, len(self._deep_ch_num) ):
            rad_vec = np.array([self.coordinates_Tx[x] - self.getAntCoordinate()[i][x],
                                self.coordinates_Tx[y] - self.getAntCoordinate()[i][y],
                                self.coordinates_Tx[z] - self.getAntCoordinate()[i][z]])  # directed to Tx
            radiationAngle[i][0], radiationAngle[i][1] = hp.cartesian_to_spherical(rad_vec[x], rad_vec[y], rad_vec[z])

        return radiationAngle