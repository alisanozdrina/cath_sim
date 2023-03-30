from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np

from IceCube_gen2_radio.AraDetector import AraDetector
from IceCube_gen2_radio.CATH import CATH
from IceCube_gen2_radio.EventTrace import EventTrace
from IceCube_gen2_radio.tools import *

prop = propagation.get_propagation_module('analytic')

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)
attenuation_model = 'SP1'

# Let us work on the y = 0 plane
#initial_point = np.array( [70, 0, -300] ) * units.m

# This function creates a ray tracing instance refracted index, attenuation model,
# number of frequencies # used for integrating the attenuation and interpolate afterwards,
# and the number of allowed reflections.
rays = prop(ice, attenuation_model,
            n_frequencies_integration=25,
            n_reflections=0)

initial_point = np.array([1062.08323229, -2107.25973729 ,  11.05436689 -1100])

ara2 = AraDetector(coordinates=np.array([0,0,0]), st_id = 2)

for ch_id in [0,4]:
    final_point = ara2.getAntCoordinate()[ch_id]
    #final_point = np.array( [100, 0, -30] ) * units.m

    rays.set_start_and_end_point(initial_point,final_point)
    rays.find_solutions()

    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]

        path = rays.get_path(i_solution)
        # We can calculate the azimuthal angle phi to rotate the
        # 3D path into the 2D plane of the points. This is only
        # necessary if we are not working in the y=0 plane
        launch_vector = rays.get_launch_vector(i_solution)
        phi = np.arctan(launch_vector[1]/launch_vector[0])
        plt.plot(path[:,0]/np.cos(phi), path[:,2], label=str(ch_id) + solution_type)

        # We can also get the 3D receiving vector at the observer position, for instance
        receive_vector = rays.get_receive_vector(i_solution)
        # Or the path length
        path_length = rays.get_path_length(i_solution)
        # And the travel time
        travel_time = rays.get_travel_time(i_solution)

plt.xlabel('horizontal coordinate [m]')
plt.ylabel('vertical coordinate [m]')

plt.legend()
plt.show()

