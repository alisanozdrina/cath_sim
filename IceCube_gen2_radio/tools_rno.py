import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units, fft
def drawTraceSurfaceArrayRNO(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images/3/3a/Channel-mapping-topview.pdf
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Surface Array', fontsize=18)

    power_string_ch = np.array([18, 19, 20])
    helper_string1_ch = np.array([17, 16, 15])
    helper_string2_ch = np.array([12, 13, 14])

    surf_ch_num = np.array([12,13,14,15,16,17,18,19,20])

    up_faced_antenna_ch = np.array([ 14, 19])
    down_faced_antenna_ch = np.array([0, 1, 2, 3])

    numOfChannels = 9

    duration = num_of_samples * (1 / sampling_rate)

    trace_ns = np.arange(0, duration, 1 / sampling_rate)

    for count, ch_id in enumerate(surf_ch_num):


        if (len(traceVoltage[ch_id]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[ch_id]) != 0):
                duration_old = len(traceVoltage[ch_id]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[ch_id] = np.interp(t_new, t_old, traceVoltage[ch_id])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[ch_id])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[ch_id] = np.append(traceVoltage[ch_id], zeros_array)

    for chanIter in range(0, len(power_string_ch)):
        row = 0

        axs[chanIter][row].plot(trace_ns, traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if power_string_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')


    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter][row].plot(trace_ns, traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if helper_string1_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter][row].plot(trace_ns, traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        if helper_string2_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')
    axs[1][2].text(0.9, 0.9, 'Trigger \n Channel', horizontalalignment='center',
                            verticalalignment='center', transform=axs[1][2].transAxes,
                            fontsize=18, c='r')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # axs[0][0].set_title('Power string',fontsize = 18)
    # axs[0][1].set_title('Helper string 1',fontsize = 18)
    # axs[0][2].set_title('Helper string 2',fontsize = 18)

    fig.text(0.05, 0.5, 'Amplitude, [V]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.04, 'time from arbitary moment, [ns]', ha='center', va='center', fontsize=18)
    plt.show()

def drawFFTSurfaceArrayRNO(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images/3/3a/Channel-mapping-topview.pdf
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Surface Array', fontsize=18)

    # ch_map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])

    power_string_ch = np.array([18, 19, 20])
    helper_string1_ch = np.array([17, 16, 15])
    helper_string2_ch = np.array([12, 13, 14])

    surf_ch_num = np.array([12,13,14,15,16,17,18,19,20])

    up_faced_antenna_ch = np.array([ 14, 19])
    down_faced_antenna_ch = np.array([0, 1, 2, 3])

    numOfChannels = 9

    duration = num_of_samples * (1 / sampling_rate)

    trace_ns = np.arange(0, duration, 1 / sampling_rate)

    for count, ch_id in enumerate(surf_ch_num):
        if (len(traceVoltage[ch_id]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[ch_id]) != 0):
                duration_old = len(traceVoltage[ch_id]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[ch_id] = np.interp(t_new, t_old, traceVoltage[ch_id])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[ch_id])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[ch_id] = np.append(traceVoltage[ch_id], zeros_array)

        traceVoltage[ch_id] = 20*np.log10( pow(fft.time2freq(traceVoltage[ch_id], sampling_rate),2)/50 *1e3)

    freq_axis = np.arange(0.05,1, (1-0.05)/len(traceVoltage[12]))

    for chanIter in range(0, len(power_string_ch)):
        row = 0

        axs[chanIter][row].plot(freq_axis, traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.05, 0.8)
        if power_string_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')


    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter][row].plot(freq_axis, traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.05, 0.8)
        if helper_string1_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter][row].plot(freq_axis, traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.05, 0.8)
        if helper_string2_ch[chanIter] in up_faced_antenna_ch:
            axs[chanIter][row].text(0.9, 0.9, 'LPDA Up', horizontalalignment='center',
                                    verticalalignment='center', transform=axs[chanIter][row].transAxes,
                                    fontsize=18, c='r')
    axs[1][2].text(0.9, 0.9, 'Trigger \n Channel', horizontalalignment='center',
                            verticalalignment='center', transform=axs[1][2].transAxes,
                            fontsize=18, c='r')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    # axs[0][0].set_title('Power string',fontsize = 18)
    # axs[0][1].set_title('Helper string 1',fontsize = 18)
    # axs[0][2].set_title('Helper string 2',fontsize = 18)

    fig.text(0.05, 0.5, 'Power, [dBm]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.04, 'Freq, [GHz]', ha='center', va='center', fontsize=18)
    plt.show()


def drawTraceDeepChannelsRNO(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images//aa/Channel-mapping-sideview-v3.pdf
    plt.rcParams["figure.figsize"] = (15, 8)

    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True)

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Deep antennas \n Tx at -200 m depth' , fontsize=18)

    power_string_ch = np.array([7,6,5,4,8,3,2,1,0])
    helper_string1_ch = np.array([11,10,9])
    helper_string2_ch = np.array([21,22,23])

    deep_antennas = np.concatenate((power_string_ch, helper_string1_ch, helper_string2_ch), axis=0)
    duration = num_of_samples * (1 / sampling_rate)
    trace_ns = np.arange(0, duration, 1 / sampling_rate)

    for count, ch_id in enumerate(deep_antennas):
        if (len(traceVoltage[ch_id]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[ch_id]) != 0):
                duration_old = len(traceVoltage[ch_id]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[ch_id] = np.interp(t_new, t_old, traceVoltage[ch_id])

            # set the number of zeros to add to the end

            n = num_of_samples - len(traceVoltage[ch_id])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[ch_id] = np.append(traceVoltage[ch_id], zeros_array)

    for chanIter in range(0, len(power_string_ch)):
        row = 0
        axs[chanIter][row].plot(trace_ns, traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)

    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter + 6][row].plot(trace_ns, traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter + 6][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 6][row].transAxes, fontsize=15)

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter + 6][row].plot(trace_ns, traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter + 6][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 6][row].transAxes, fontsize=15)

    empty_plots = np.array([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17])
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


    axs[6][1].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[6][1].transAxes, fontsize=18, c='r')
    axs[6][2].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[6][2].transAxes, fontsize=18, c='r')

    fig.text(0.09, 0.5, 'Amplitude, [V]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.05, 'time from the arbitrary moment, [ns]', ha='center', va='center', fontsize=18)

    plt.show()


def drawFFTDeepChannelsRNO(station_name, event_id, traceVoltage, trace_sampling_rate, sampling_rate, num_of_samples):
    # https://radio.uchicago.edu/wiki/images//aa/Channel-mapping-sideview-v3.pdf
    plt.rcParams["figure.figsize"] = (15, 8)

    fig, axs = plt.subplots(nrows=9, ncols=3, sharex=True, sharey=True)

    station_id = station_name
    eventID = event_id

    fig.suptitle('Station ' + str(station_id) + ' event ' + str(eventID) + ', Deep antennas \n Tx at -200 m depth' , fontsize=18)

    power_string_ch = np.array([7, 6, 5, 4, 8, 3, 2, 1, 0])
    helper_string1_ch = np.array([11, 10, 9])
    helper_string2_ch = np.array([21, 22, 23])

    deep_antennas = np.concatenate((power_string_ch, helper_string1_ch, helper_string2_ch), axis=0)
    duration = num_of_samples * (1 / sampling_rate)
    trace_ns = np.arange(0, duration, 1 / sampling_rate)

    for count, ch_id in enumerate(deep_antennas):

        if (len(traceVoltage[ch_id]) < num_of_samples):
            if (trace_sampling_rate != sampling_rate and len(traceVoltage[ch_id]) != 0):
                duration_old = len(traceVoltage[ch_id]) * (1 / trace_sampling_rate)
                t_old = np.arange(0, duration_old, 1 / trace_sampling_rate)
                # define the new sampling rate and time points
                sampling_rate_new = sampling_rate

                t_new = np.arange(0, duration_old, 1 / sampling_rate_new)

                # compute the upsampled waveform using linear interpolation
                traceVoltage[ch_id] = np.interp(t_new, t_old, traceVoltage[ch_id])

            # set the number of zeros to add to the end
            n = num_of_samples - len(traceVoltage[ch_id])
            # create a numpy array of n zeros
            zeros_array = np.zeros(n)
            # append the zeros array to the end of the original array
            traceVoltage[ch_id] = np.append(traceVoltage[ch_id], zeros_array)
        traceVoltage[ch_id] = 20*np.log10( pow(fft.time2freq(traceVoltage[ch_id], sampling_rate),2)/50 *1e3)
    freq_axis = np.arange(0.05,1, (1-0.05)/len(traceVoltage[0]))

    for chanIter in range(0, len(power_string_ch)):
        row = 0
        axs[chanIter][row].plot(freq_axis, traceVoltage[power_string_ch[chanIter]])
        channelLabel = 'ch#' + str(power_string_ch[chanIter])
        axs[chanIter][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                transform=axs[chanIter][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.1, 0.8)

    for chanIter in range(0, len(helper_string1_ch)):
        row = 1
        axs[chanIter + 6][row].plot(freq_axis, traceVoltage[helper_string1_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string1_ch[chanIter])
        axs[chanIter + 6][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 6][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.1, 0.8)

    for chanIter in range(0, len(helper_string2_ch)):
        row = 2
        axs[chanIter + 6][row].plot(freq_axis, traceVoltage[helper_string2_ch[chanIter]])
        channelLabel = 'ch#' + str(helper_string2_ch[chanIter])
        axs[chanIter + 6][row].text(0.9, 0.1, channelLabel, horizontalalignment='center', verticalalignment='center',
                                    transform=axs[chanIter + 6][row].transAxes, fontsize=15)
        axs[chanIter][row].set_xlim(0.1, 0.8)

    empty_plots = np.array([1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17])
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


    axs[6][1].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[6][1].transAxes, fontsize=18, c='r')
    axs[6][2].text(0.9, 0.7, 'H pol', horizontalalignment='center', verticalalignment='center',
                   transform=axs[6][2].transAxes, fontsize=18, c='r')
    plt.show()
