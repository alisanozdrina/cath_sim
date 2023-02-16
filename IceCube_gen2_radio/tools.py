import numpy as np
import matplotlib.pyplot as plt

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioReco.detector import antennapattern
from NuRadioMC.SignalGen import emitter
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

import scipy.constants
from NuRadioReco.utilities import units, fft
from radiotools import helper as hp
from scipy import constants
import uproot
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.detector.RNO_G import analog_components
from NuRadioReco.detector import detector
from NuRadioReco.utilities import geometryUtilities as geo_utl
import datetime

from IceCube_gen2_radio.IC_hybrid_station import *
import matplotlib.lines

def cable_attenuarion(inputFFT, inputFreqRange):
    cable_length = 300 #m
    # using lmr-600 cable for reference
    # https://www.fairviewmicrowave.com/images/productPDF/LMR-600.pdf data spreadsheet
    freqGHz = np.array([50,150,220,450,900,1500])*1e-3
    att_dBp100m = np.array([1.8,3.2,3.9,5.6,8.2,10.9])

    freqGHz_interpolated = inputFreqRange
    att_dBp100m_interpolated = np.interp(freqGHz_interpolated, freqGHz, att_dBp100m)
    #convert attenuation from db to voltage ratio

    return inputFFT/ (pow(10, att_dBp100m_interpolated/20))
    
def get_ICH_coordinates(Num_of_stations=164, separation=1750, grid_type='square', 
                        coordinate_center=np.array([0, 0]), station_type='hybrid'):
    N_h = Num_of_stations
    #N_sh = 197 # num of IC_hybrid_stations 164 of hybrid and 197 of shallow
    d_h = separation 
    d_sh = separation # separation in m

    a=14
    b=8

    N_X = int(np.ceil(np.sqrt(Num_of_stations)) )
    N_Y = int(np.ceil(np.sqrt(Num_of_stations)) )

    st_array = []

    if ( grid_type=='square' and station_type=='hybrid' ):
        xv_h, yv_h = np.meshgrid(np.arange(N_X), np.arange(N_Y),indexing='xy')
        xv_h = (xv_h-(np.amax(xv_h))/2)*d_h + coordinate_center[0] 
        yv_h = (yv_h-(np.amax(yv_h))/2)*d_h + coordinate_center[1]

        for x in range(N_X):
            for y in range(N_Y):
                #if (  xv_h[0][x]**2/b**2 + yv_h[y][0]**2/a**2 )  <= 1:
                if len(st_array) >= N_h:
                    break
                st_name = str(x) + '.' + str(y)
                #print(x,y, st_num)
                coord_vec = np.array([xv_h[x][y], yv_h[x][y], 0])
                st_array.append(IC_hybrid_station(st_name, coord_vec))


        x = np.zeros(len(st_array))
        y = np.zeros(len(st_array))
        z = np.zeros(len(st_array))
        
        # plt.rcParams["figure.figsize"] = (12,15)
        # plt.gca().set_aspect('equal')
        for i in range(len(st_array)):
            x[i] = st_array[i]._coordinates[0]
            y[i] = st_array[i]._coordinates[1]

            name = str(st_array[i]._name)

            # plt.scatter(x[i], y[i], s=90, marker = 'o', c='b')
            # plt.annotate(name, xy=(x[i], y[i]), 
            #          xytext=(x[i], y[i]-1000))
            
    if (grid_type=='square' and station_type=='shallow'):
        
        xv_sh, yv_sh = np.meshgrid(np.arange(N_X), np.arange(N_Y),indexing='xy')
        xv_sh = (xv_sh-(np.amax(xv_sh))/2)*d_h + coordinate_center[0] + 875
        yv_sh = (yv_sh-(np.amax(yv_sh))/2)*d_h + coordinate_center[1] + 875

        for x in range(N_X):
            for y in range(N_Y):
                #if (  xv_sh[0][x]**2/b**2 + yv_sh[y][0]**2/a**2 )  <= 1:
                if len(st_array) >= N_sh:
                    break
                st_name = str(x) + '.' + str(y)
                coord_vec = np.array([xv_sh[x][y], yv_sh[x][y], 0])
                st_array.append(IC_shallow_station(st_name, coord_vec))


        x = np.zeros(len(st_array))
        y = np.zeros(len(st_array))
        z = np.zeros(len(st_array))

        for i in range(len(st_array)):

            x[i] = st_array[i]._coordinates[0]
            y[i] = st_array[i]._coordinates[1]

            name = str(st_array[i]._name)

            # plt.scatter(x[i], y[i], s=90, marker = 'o', c='g')


    # l = matplotlib.lines.Line2D([np.amin(x)+5000, np.amin(x)], [np.amin(y)-2000, np.amin(y)-2000], c='k')
    # plt.gca().add_line(l)
    # plt.annotate('5 km', xy=(np.amin(x)+3000, np.amin(y)-2000), xytext=(np.amin(x)+2000, np.amin(y)-2500))
    # plt.ylim(np.amin(y)-5000, np.amax(y)+5000)
    # plt.xlim(np.amin(x)-5000, np.amax(x)+5000)
    #plt.plot()
    print( Num_of_stations , station_type + ' stations with' , separation , 'm in a ' + grid_type + ' grid have been generated')
    return st_array

def drawTraceSurfaceArray(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    #https://radio.uchicago.edu/wiki/images/3/3a/Channel-mapping-topview.pdf
    plt.rcParams["figure.figsize"] = (15,8)
    fig, axs = plt.subplots(nrows=3, ncols = 3, sharex=True,  sharey=True)

    #timeToPrint = station.get_station_time().to_datetime().strftime('%Y-%b-%d %H:%M:%S')
    #timeStamp = str(station.get_station_time())
    
    station_id = station_name
    eventID = event_id

    
    fig.suptitle('Station '+ str(station_id) + ' event ' + str(eventID) + ', Surface Array',fontsize = 18)

    #ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])

    power_string_ch = np.array([0,1,2])
    helper_string1_ch = np.array([3,4,5])
    helper_string2_ch = np.array([6])
    
    up_faced_antenna_ch = np.array([4,5,6])
    down_faced_antenna_ch = np.array([0,1,2,3])

    numOfChannels = 7

    duration=num_of_samples*(1/sampling_rate)

    trace_ns = np.zeros((numOfChannels,num_of_samples))
    for i in range(0,numOfChannels):
        trace_ns[i] = np.arange(0, duration, 1/sampling_rate)

        if (len(traceVoltage[i] ) < num_of_samples ) :
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[i] ) !=0):

                duration_old=len(traceVoltage[i])*(1/trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1/trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1/sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[i] = np.interp(t_new, t_old, traceVoltage[i])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[i] )
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[i] = np.append(traceVoltage[i], zeros_array)


    for chanIter in range( 0,len(power_string_ch) ):
        row = 0

        axs[chanIter][row].plot( trace_ns[ power_string_ch[chanIter] ], traceVoltage[  power_string_ch[chanIter] ] )
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center', 
                   transform=axs[chanIter][row].transAxes, fontsize = 15)
        if power_string_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center', 
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes, 
                                    fontsize = 18, c='r')


    for chanIter in range( 0,len(helper_string1_ch) ):
        row = 1
        axs[chanIter][row].plot( trace_ns[ helper_string1_ch[chanIter] ], traceVoltage[  helper_string1_ch[chanIter] ] )
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center', 
                   transform=axs[chanIter][row].transAxes, fontsize = 15)
        if helper_string1_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center', 
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes, 
                                    fontsize = 18, c='r')

    for chanIter in range( 0,len(helper_string2_ch) ):
        row = 2
        axs[chanIter][row].plot( trace_ns[ helper_string2_ch[chanIter] ], traceVoltage[  helper_string2_ch[chanIter] ] )
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center', 
                   transform=axs[chanIter][row].transAxes, fontsize = 15)
        if helper_string2_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center', 
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes, 
                                    fontsize = 18, c='r')


        
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # axs[0][0].set_title('Power string',fontsize = 18)
    # axs[0][1].set_title('Helper string 1',fontsize = 18)
    # axs[0][2].set_title('Helper string 2',fontsize = 18)

    fig.text(0.05, 0.5, 'Amplitude, [V]', ha='center', va='center', rotation='vertical',fontsize = 18)
    fig.text(0.5, 0.04, 'time from arbitary moment, [ns]', ha='center', va='center',fontsize = 18)
    plt.show()

def plot_surface_array(array_of_st,cath_site, Event):

    r_toDraw = 8000 
    Tx_coordinates = cath_site._coordinates
    tx_x = Tx_coordinates[0]
    tx_y = Tx_coordinates[1]
    tx_z = Tx_coordinates[2]

    plt.rcParams["figure.figsize"] = (8,10)

    plt.annotate('Tx', xy=(tx_x+200,tx_y-100), xytext=(tx_x+300,tx_y-800), c='r')
    plt.scatter(tx_x,tx_y, c='r', marker='*', s=180)

    x = np.zeros(len(array_of_st))
    y = np.zeros(len(array_of_st))
    z = np.zeros(len(array_of_st))

    labelFlag=True

    for i in range(0,len(array_of_st)):
        x[i] = array_of_st[i]._coordinates[0]
        y[i] = array_of_st[i]._coordinates[1]
        if( Event[i].isTriggered_surface ):
            z[i] = np.max(Event[i].get_trace(6))
        else:
            z[i] = 0


        name = str(int(array_of_st[i]._name[0])) + '.' + str(int(array_of_st[i]._name[2]))

        if Event[i].isTriggered_surface :


            plt.scatter(x[i], y[i], s=270, marker = 'o', c='w', edgecolor = 'r')

            plt.annotate(name, xy=(x[i], y[i]), xytext=(x[i], y[i]-100))
            if (labelFlag):
                plt.scatter(x[i], y[i], s=270, marker = 'o', c='w', edgecolor = 'r', label = 'above trigger threshold')
                labelFlag=False      
        else:
            plt.scatter(x[i], y[i], s=170, marker = 'o', c='w', edgecolor = 'b')

            plt.annotate(name, xy=(x[i], y[i]), 
                         xytext=(x[i], y[i]-100))

    #draw the grid, fill ice cube Rx station array with coordinates, names, relative angles
    plt.ylim(-r_toDraw-800,r_toDraw+800)
    plt.xlim(-r_toDraw-800,r_toDraw+800)
    N_pos = r_toDraw-100
    plt.annotate('N', xy=(N_pos, N_pos), xytext=(0, N_pos-10),fontsize=15)

    dist = np.linalg.norm(array_of_st[0]._coordinates-array_of_st[1]._coordinates)

    plt.annotate('~'+ str(int(round(dist,-1))) + ' [m]', 
                 xy=(array_of_st[0]._coordinates[0], array_of_st[0]._coordinates[1]-700), 
                 xytext=((array_of_st[1]._coordinates[0]+20, array_of_st[1]._coordinates[1]-850)),
                arrowprops=dict(arrowstyle='<->', color='red'), c='r')

    plt.scatter(x, y, c=z, cmap="binary", s=180, marker = 'o')
    # Adding the colorbar

    plt.colorbar(label="Rx amplitude, [V]", orientation="horizontal",pad = 0.11)

    plt.ylabel('distance, [m]')


    circle1 = plt.Circle((0, 0), 1000, color='r', fill=True, alpha = 0.1, label = 'ray tracing solution exists')
    plt.gca().add_patch(circle1)

    #artist = mpatches.Circle((0.5,0.5),radius=0.5, edgecolor=colorList[i],fill=False,label = #label for the ith circle)
    #plt.add_patch(artist)
                             
    #plt.legend([circle1], 'sfsd')

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


