import numpy as np
class EventTrace:

    # channel_mapping: ch0-6 surface array, ch7-16 power string from top to bottom, ch16-19 helper string 1, ch19-23 helper string 2
    # surface array map = np.array(['ch1_down', 'ch2_down', 'ch3_down', 'ch4_down', 'ch5_up', 'ch6_up', 'ch7_up'])
    def __init__(self, ev_num, station_name):
        self.ev_num = ev_num
        self.station_name = station_name
        self._sampling_rate = 3.2e9  # 3.2 GHz similar to RNO-g
        self._trace_length = 2048  # samples - corresponds to 640 ns trace length
        self._num_of_ch = 24
        self.traces = [np.array([]) for i in range(self._num_of_ch)]
        self.fft_traces = [np.array([]) for i in range(self._num_of_ch)]
        self.hit_time = np.zeros(self._num_of_ch)

    def set_hit_time(self, index, hit_time):
        # hit_time [ns]
        self.hit_time[index] = hit_time
    def set_trace(self, index, trace, samp_rate=2.5, start_time=0):
        # if start_time!=0:
        # 	dt = 1/samp_rate
        # 	trace = np.concatenate((np.zeros(int(start_time/dt)),trace))
        # 	print(trace)
        # else:
        self.traces[index] = trace
        self.fft_traces[index] = np.fft.fft(trace)

    def get_trace(self, index):
        return self.traces[index]


    def get_traces(self):
        return self.traces

    def get_fft_trace(self, index):
        return self.fft_traces[index]

    def isTriggered_surface(self, x=0.045):
        arrays = self.traces[:7]
        num_arrays = len(arrays)
        count = 0
        for array in arrays:
            if np.max(array) > x:
                count += 1
        return count >= 3
