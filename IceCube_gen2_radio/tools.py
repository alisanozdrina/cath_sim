import numpy as np
import matplotlib.pyplot as plt

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioReco.detector import antennapattern
from NuRadioMC.SignalGen import emitter
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

import scipy.constants
from scipy.signal import butter, lfilter

from NuRadioReco.utilities import units, fft
from radiotools import helper as hp
from scipy import constants
import uproot
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.detector.RNO_G import analog_components
from NuRadioReco.detector import detector
from NuRadioReco.utilities import geometryUtilities as geo_utl
import datetime

import pandas as pd

from IceCube_gen2_radio.IC_hybrid_station import *
import matplotlib.lines


def cable_attenuation(inputFreqRange, cable_length=300):
    # cable_length  [m]
    # using lmr-600 cable for reference
    # https://www.fairviewmicrowave.com/images/productPDF/LMR-600.pdf data spreadsheet
    freqGHz = np.array([50, 150, 220, 450, 900, 1500]) * 1e-3
    att_dBp100m = np.array([1.8, 3.2, 3.9, 5.6, 8.2, 10.9]) * cable_length / 100

    freqGHz_interpolated = inputFreqRange
    att_dBp100m_interpolated = np.interp(freqGHz_interpolated, freqGHz, att_dBp100m)
    att_dBp100m_interpolated = att_dBp100m_interpolated
    # convert attenuation from db to voltage ratio
    # P = P0 / 10^(Lp dB/10 dB)
    # attenuation given per 100 m

    return 1 / ((pow(10, att_dBp100m_interpolated / 10)))


def fiber_link(inputFreqRange, cable_length=300):
    # cable_length  [m]
    # using lmr-600 cable for reference
    # https://www.fairviewmicrowave.com/images/productPDF/LMR-600.pdf data spreadsheet
    distance_factor = cable_length / 1000
    fiber_attenuation = np.ones(len(inputFreqRange)) * 0.4 * distance_factor  # dB

    return 1 / ((pow(10, fiber_attenuation / 10)))


def get_ICH_coordinates(Num_of_stations: object = 164, separation: object = 1750, grid_type: object = 'square',
                        coordinate_center: object = np.array([0, 0]), station_type: object = 'hybrid') -> object:
    N_h = Num_of_stations
    # N_sh = 197 # num of IC_hybrid_stations 164 of hybrid and 197 of shallow
    d_h = separation
    d_sh = separation  # separation in m
    # return array of station objects_

    a = 14
    b = 8

    N_X = int(np.ceil(np.sqrt(Num_of_stations)))
    N_Y = int(np.ceil(np.sqrt(Num_of_stations)))

    st_array = []

    if (grid_type == 'square' and station_type == 'hybrid'):
        xv_h, yv_h = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing='xy')
        xv_h = (xv_h - (np.amax(xv_h)) / 2) * d_h + coordinate_center[0]
        yv_h = (yv_h - (np.amax(yv_h)) / 2) * d_h + coordinate_center[1]

        for x in range(N_X):
            for y in range(N_Y):
                # if (  xv_h[0][x]**2/b**2 + yv_h[y][0]**2/a**2 )  <= 1:
                if len(st_array) >= N_h:
                    break
                st_name = str(x) + '.' + str(y)
                # print(x,y, st_num)
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

    if (grid_type == 'square' and station_type == 'shallow'):

        xv_sh, yv_sh = np.meshgrid(np.arange(N_X), np.arange(N_Y), indexing='xy')
        xv_sh = (xv_sh - (np.amax(xv_sh)) / 2) * d_h + coordinate_center[0] + 875
        yv_sh = (yv_sh - (np.amax(yv_sh)) / 2) * d_h + coordinate_center[1] + 875

        for x in range(N_X):
            for y in range(N_Y):
                # if (  xv_sh[0][x]**2/b**2 + yv_sh[y][0]**2/a**2 )  <= 1:
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
    # plt.plot()
    print(Num_of_stations, station_type + ' stations with', separation,
          'm in a ' + grid_type + ' grid have been generated')
    return st_array


def drawTraceSurfaceArray(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images/3/3a/Channel-mapping-topview.pdf
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

    # timeToPrint = station.get_station_time().to_datetime().strftime('%Y-%b-%d %H:%M:%S')
    # timeStamp = str(station.get_station_time())

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Surface Array', fontsize=18)

    # ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])

    power_string_ch = np.array([0, 1, 2])
    helper_string1_ch = np.array([3, 4, 5])
    helper_string2_ch = np.array([6])

    up_faced_antenna_ch = np.array([4, 5, 6])
    down_faced_antenna_ch = np.array([0, 1, 2, 3])

    numOfChannels = 7

    duration = num_of_samples * (1 / sampling_rate)

    trace_ns = np.zeros((numOfChannels, num_of_samples))
    for i in range(0, numOfChannels):
        trace_ns[i] = np.arange(0, duration, 1 / sampling_rate)

        if (len(traceVoltage[i]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[i]) != 0):
                duration_old = len(traceVoltage[i]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[i] = np.interp(t_new, t_old, traceVoltage[i])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[i])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[i] = np.append(traceVoltage[i], zeros_array)

    for chanIter in range(0, len(power_string_ch)):
        row = 0

        axs[chanIter][row].plot(trace_ns[power_string_ch[chanIter]], traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if power_string_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')

    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter][row].plot(trace_ns[helper_string1_ch[chanIter]], traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if helper_string1_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter][row].plot(trace_ns[helper_string2_ch[chanIter]], traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if helper_string2_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')

    # for ax in fig.get_axes():
    #     ax.label_outer()

    # axs[0][0].set_title('Power string',fontsize = 18)
    # axs[0][1].set_title('Helper string 1',fontsize = 18)
    # axs[0][2].set_title('Helper string 2',fontsize = 18)

    fig.text(0.05, 0.5, 'Amplitude, [V]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.04, 'time from arbitary moment, [ns]', ha='center', va='center', fontsize=18)
    plt.show()


def plot_surface_array(array_of_st, cath_site, Event):
    r_toDraw = 3000
    Tx_coordinates = cath_site._coordinates
    tx_x = Tx_coordinates[0]
    tx_y = Tx_coordinates[1]
    tx_z = Tx_coordinates[2]

    plt.rcParams["figure.figsize"] = (10, 8)

    plt.annotate('Tx', xy=(tx_x + 200, tx_y - 100), xytext=(tx_x + 300, tx_y - 400), c='r')
    plt.scatter(tx_x, tx_y, c='r', marker='*', s=180)

    x = np.zeros(len(array_of_st))
    y = np.zeros(len(array_of_st))
    z = np.zeros(len(array_of_st))

    labelFlag = True

    for i in range(0, len(array_of_st)):
        x[i] = array_of_st[i]._coordinates[0]
        y[i] = array_of_st[i]._coordinates[1]
        z[i] = np.max(Event[i].get_trace(0))

        name = array_of_st[i]._name

        if z[i] > 0.045:

            plt.scatter(x[i], y[i], s=270, marker='o', c='w', edgecolor='r')

            plt.annotate(name, xy=(x[i], y[i]), xytext=(x[i], y[i] - 100))
            if (labelFlag):
                plt.scatter(x[i], y[i], s=270, marker='o', c='w', edgecolor='r', label='above trigger threshold')
                labelFlag = False
        else:
            plt.scatter(x[i], y[i], s=170, marker='o', c='w', edgecolor='b')

            plt.annotate(name, xy=(x[i], y[i]),
                         xytext=(x[i], y[i] - 100))

    # draw the grid, fill ice cube Rx station array with coordinates, names, relative angles
    plt.ylim(-r_toDraw - 800, r_toDraw + 800)
    plt.xlim(-r_toDraw - 800, r_toDraw + 800)
    N_pos = r_toDraw - 100
    plt.annotate('N', xy=(N_pos, N_pos), xytext=(0, N_pos - 10), fontsize=15)

    dist = np.linalg.norm(array_of_st[0]._coordinates - array_of_st[1]._coordinates)

    plt.annotate('~' + str(int(round(dist, -1))) + ' [m]',
                 xy=(array_of_st[0]._coordinates[0], array_of_st[0]._coordinates[1] - 700),
                 xytext=((array_of_st[1]._coordinates[0], array_of_st[1]._coordinates[1] - 750)),
                 arrowprops=dict(arrowstyle='<->', color='red'), c='r')

    # plt.scatter(x, y, c=z, cmap="binary", s=180, marker = 'o')
    # Adding the colorbar

    # plt.colorbar(label="Rx amplitude, [V]", orientation="horizontal",pad = 0.11)
    plt.gca().set_aspect('equal')
    plt.ylabel('distance, [m]')

    circle1 = plt.Circle((tx_x, tx_y), 1300, color='r', fill=True, alpha=0.1, label='ray tracing solution exists')
    plt.gca().add_patch(circle1)

    # artist = matplotlib.patches.Circle((0.5,0.5),radius=0.5, edgecolor=colorList[i],fill=False,label = '')
    # plt.add_patch(artist)

    plt.legend([circle1], 'sfsd')

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def drawTraceDeepChannels(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images/a/aa/Channel-mapping-sideview-v3.pdf
    plt.rcParams["figure.figsize"] = (15, 8)

    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True)

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Deep antennas', fontsize=18)

    # power string  row = 0 ch 7,6,5,4,8,3,2,1,0

    power_string_ch = np.array([7, 8, 9, 18, 19, 10, 11, 12, 13])
    helper_string1_ch = np.array([20, 14, 21, 15])
    helper_string2_ch = np.array([22, 16, 23, 17])

    # h_pols = np.array([,])

    v_pols = np.concatenate((power_string_ch, helper_string1_ch, helper_string2_ch), axis=0)
    numOfChannels = 11

    duration = num_of_samples * (1 / sampling_rate)

    trace_ns = np.zeros((24, num_of_samples))

    for count, ch_id in enumerate(v_pols):
        i = ch_id
        trace_ns[i] = np.arange(0, duration, 1 / sampling_rate)

        if (len(traceVoltage[i]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[i]) != 0):
                duration_old = len(traceVoltage[i]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[i] = np.interp(t_new, t_old, traceVoltage[i])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[i])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[i] = np.append(traceVoltage[i], zeros_array)

    for chanIter in range(0, len(power_string_ch)):
        row = 0
        axs[chanIter][row].plot(trace_ns[power_string_ch[chanIter]], traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)

    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter + 5][row].plot(trace_ns[helper_string1_ch[chanIter]], traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter + 5][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 5][row].transAxes, fontsize=15)

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter + 5][row].plot(trace_ns[helper_string2_ch[chanIter]], traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter + 5][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 5][row].transAxes, fontsize=15)

    # power_string_ch = np.array([7,8,9, 18,19, 10,11,12,13])
    # helper_string1_ch = np.array([20,14,21,15])
    # helper_string2_ch = np.array([22,16,23,17])

    empty_plots = np.array([1, 2, 4, 5, 7, 8, 10, 11, 13, 14])
    for empty_plot in empty_plots:
        fig.delaxes(axs.flatten()[empty_plot])

    for ax in fig.get_axes():
        ax.label_outer()

    axs[0][0].set_title('Power string', fontsize=18)
    axs[5][1].set_title('Helper string 1', fontsize=18)
    axs[5][2].set_title('Helper string 2', fontsize=18)

    axs[3][0].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[3][0].transAxes, fontsize=18, c='r')
    axs[4][0].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[4][0].transAxes, fontsize=18, c='r')
    axs[5][1].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[5][1].transAxes, fontsize=18, c='r')
    axs[5][2].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[5][2].transAxes, fontsize=18, c='r')

    axs[7][1].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[7][1].transAxes, fontsize=18, c='r')
    axs[7][2].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[7][2].transAxes, fontsize=18, c='r')

    fig.text(0.09, 0.5, 'Amplitude, [mV]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.05, 'time from arbitary moment, [ns]', ha='center', va='center', fontsize=18)

    plt.show()


def superimposeWF(WF, t_shift, sampling_rate=3.2):
    # trace_base = np.zeros(2048)
    # for n in range(0, len(WF)):
    #     trace = WF[n]
    #     shift = delta_t[n]
    #     for i in range(0, len(trace)):
    #         it = int(shift+i)
    #         if (it >= len(trace_base)):
    #             #time window is to small, cut the trace
    #             break
    #         else:
    #             trace_base[it] += trace[i]
    # return trace_base
    trace_base = np.zeros(2048)
    shift_inSamples = int(t_shift * sampling_rate)
    for i in range(0, len(WF[0])):
        trace_base[i] += WF[0][i]

    for i in range(0, len(WF[1])):
        it = int(shift_inSamples + i)
        if (it >= len(trace_base)):
            # time window is too small, cut the trace
            break
        else:
            trace_base[it] += WF[1][i]
    return trace_base


def shift_trace(WF, delta_t, sampling_rate):
    WF_shifted = np.zeros(len(WF))
    shift_inSamples = int(delta_t * sampling_rate)
    if shift_inSamples < len(WF):
        for i in range(0, len(WF) - shift_inSamples):
            if i >= 2047:
                print('default time is too big! channel is delayed! ')
            WF_shifted[i + shift_inSamples] = WF[i]
    return WF_shifted


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


#
# [ 184 1225] [ 59.30416222 394.82390608] {'peak_heights': array([107.57064138, 205.74334216])} 0
# [ 33 731] [ 14.38787295 318.71318576] {'peak_heights': array([ 57.30185485, 269.3299978 ])} 1
# [ 452 1317] [177.58542931 517.43365134] {'peak_heights': array([140.80094539, 336.38260643])} 2
# [ 73 868] [ 29.27755644 348.12217795] {'peak_heights': array([ 69.18398449, 249.25956165])} 3
# [ 401 1666] [127.16345817 528.31501575] {'peak_heights': array([226.61028263, 270.86505086])} 4
# [ 133 1392] [ 42.51870148 445.00776288] {'peak_heights': array([156.4980312 , 259.82025763])} 5
# [ 768 1973] [255.90584774 657.42478853] {'peak_heights': array([316.03190869, 278.27886837])} 6
# [ 175 1491] [ 55.94565984 476.65702188] {'peak_heights': array([211.80479952, 391.80404937])} 7
def get_spiceCore_DnR(ch_id):
    dt_DnR = np.array([[ 52.85805763, 385.79935966], [  6.1039461, 309.5572666],
                       [172.8707719,  507.21856028], [ 13.63612218, 342.1062417 ],
                       [ 93.2320616,  513.09345467], [  9.59068454, 435.09738885],
                       [225.58367047, 649.09452004], [ 19.18136909, 465.46788991]])
    return dt_DnR[ch_id]
def find_peak_start(WF, threshold, distance):
    peaks=np.array([],dtype=int)
    shift=0
    for binNum in range(0, len(WF)):
        if binNum+shift >= len(WF):
            break
        else:
            if abs(WF[binNum+shift]) > threshold:
                peaks = np.append(peaks, int(binNum+shift) )
                shift = shift + distance
    return peaks

def get_channel_delay (ch_id):
    delay = np.array([ 5.60000000e-01,  1.39400000e-01, -1.05547000e+02, -2.57590000e+00,
                       -7.34100000e+01, -7.38306000e+01, -1.86317000e+02, -7.65459000e+01,
                       1.28149923e+01,  1.22823466e+01, -9.33794532e+01,  9.56549035e+00,
                       -6.10678246e+01, -6.15181106e+01, -1.67215309e+02, -6.42134005e+01])
    return delay[ch_id]