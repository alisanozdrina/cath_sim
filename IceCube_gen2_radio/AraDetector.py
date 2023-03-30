import numpy as np
from NuRadioReco.detector import detector
from NuRadioReco.detector import antennapattern
import json
import numpy as np
from radiotools import helper as hp


class AraDetector:
    coordinates_Tx: np.array([])

    _deep_ch_num = np.arange(0, 16, 1)

    _pathToJson = "/Users/alisanozdrina/Documents/phys/exp/rno/soft/NuRadioMC/NuRadioReco/detector/ARA/ara_detector_db.json"
    _det = detector.Detector(json_filename=_pathToJson)

    def __init__(self, coordinates: object = np.array([0, 0, 0]), st_id=2):
        self._coordinates = coordinates
        self.coordinates_Tx = np.array([])
        self._st_id = st_id

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

        # if self._st_id == 1:
        #     ch_numbers_db = np.arange(0, 16, 1)
        if self._st_id == 2:
            ch_coord = np.array([[10.5874, 2.3432, -170.247],
                                [4.85167, -10.3981, -170.347],
                                [-2.58128, 9.37815, -171.589],
                                [-7.84111, -4.05791, -175.377],
                                [10.5873, 2.3428, -189.502],
                                [4.85157, -10.3985, -189.4],
                                [-2.58138, 9.37775, -191.242],
                                [-7.84131, -4.05821, -194.266],
                                [10.5874, 2.3128, -167.492],
                                [4.85167, -10.3981, -167.428],
                                [-2.58128, 9.37825, -168.468],
                                [-7.84111, -4.05781, -172.42],
                                [10.5873, 2.3429, -186.546],
                                [4.85157, -10.3985, -186.115],
                                [-2.58138, 9.37775, -187.522],
                                [-7.84111, -4.05821, -190.981]])
            # ch_coord = np.array([[-2604.46524027,  4008.28584455,  -175.97718961],
            #                     [-2590.82537997,  4011.3191023,   -176.03135318],
            #                     [-2602.21273009,  3993.52761568,  -177.32869767],
            #                     [-2588.28977718,  3997.36210244,  -181.06927118],
            #                     [-2604.404921 ,   4008.30746981,  -195.23294053],
            #                     [-2590.7656895,   4011.34050236,  -195.08509621],
            #                     [-2602.15117188,  3993.54968464,  -196.98246414],
            #                     [-2588.23061746,  3997.3831797,   -199.95900802],
            #                     [-2604.4494657,   4008.30097208,  -173.22198588],
            #                     [-2590.83446654,  4011.31584813,  -173.11223914],
            #                     [-2602.22252557,  3993.52407645,  -174.20757605],
            #                     [-2588.29906214,  3997.35874604,  -178.11215596],
            #                     [-2604.41420285,  4008.30411453,  -192.27682535],
            #                     [-2590.7759154,   4011.33684017,  -191.79996787],
            #                     [-2602.16275189,  3993.5455375,   -193.2623188 ],
            #                     [-2588.24096309,  3997.37967771,  -196.67387987]])

            for i, ch in enumerate( self._deep_ch_num ):  # self._deep_ch_num

                ch_coord[i] = np.array([center[x] + ch_coord[i][x],
                                        center[y] + ch_coord[i][y],
                                        center[z] + ch_coord[i][z]])
            f.close()
        return ch_coord

    def getAntOrientationRotation(self, ch_id=0):

        # ch_ori = np.zeros((len(self._deep_ch_num), 4))
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
        for i in range(0, len(self._deep_ch_num)):
            distances[i] = np.linalg.norm(self.coordinates_Tx - self.getAntCoordinate[i])
        return distances

    def getRadiationAngle(self):

        radiationAngle = np.zeros((len(self._deep_ch_num), 2))
        x, y, z = 0, 1, 2
        for i in range(0, len(self._deep_ch_num)):
            rad_vec = np.array([self.coordinates_Tx[x] - self.getAntCoordinate[i][x],
                                self.coordinates_Tx[y] - self.getAntCoordinate[i][y],
                                self.coordinates_Tx[z] - self.getAntCoordinate[i][z]])  # directed to Tx
            radiationAngle[i][0], radiationAngle[i][1] = hp.cartesian_to_spherical(rad_vec[x], rad_vec[y], rad_vec[z])

        return radiationAngle
