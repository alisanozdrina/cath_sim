from unittest import TestCase
import numpy as np
from Ara1Detector import Ara1Detector
class TestAra1Detector(TestCase):
    def setUp(self):
        self.ara = Ara1Detector()
class TestDetectorInit(TestAra1Detector):
    def test_get_ant_coordinate(self):
        self.assertEqual(self.ara.getAntCoordinate()[0][0], 8.45158 + self.ara._coordinates[0])
        self.assertEqual(len(self.ara.getAntCoordinate()), 16)

    def test_get_ant_angles(self):
        self.assertEqual(self.ara.getAntOrientationRotation()[0][0], 0)

    def test_get_ant_dist(self):
        self.ara.set_tx_coordinates(tx_coordinates=np.array([100,100,-100]))
        distance2antenna = self.ara.getDistance_InIceAnt()[0]
        self.assertAlmostEqual(distance2antenna, 141, 0)

    def test_get_radiation_direction(self):
        self.ara.set_tx_coordinates(tx_coordinates=np.array([100, 4.68337,-76.2137]))
        zenith_angle = self.ara.getRadiationAngle()[0][0]
        azimuth_angle = self.ara.getRadiationAngle()[0][1]
        self.assertAlmostEqual(zenith_angle, np.pi/2, 0)
        self.assertAlmostEqual(azimuth_angle, 0, 0)