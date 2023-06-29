import numpy as np
import glob
def generate_thermal_noise(freq_range_input, depth = -150, time_trace_length = 2048):

    # try change delta_f in freq_range_input to increase number of samples...
    # freq_range_input in GHz
    kB = 1.380649 * 10e-23
    Z = 50  # Ohm
    T_ice = 220  # K

    # get s11 of the rno-g vpol antenna, piece of code taken from https://radio.uchicago.edu/wiki/index.php/Science_Antennas

    file_names = glob.glob("/Users/alisanozdrina/Documents/phys/exp/rno/vpol_v2_s11s/*csv*")
    def open_s11(file_name):
        f = open(file_name)
        freqs = []
        s11 = []

        for line in f:
            if ("!" in line or "BEGIN" in line or "END" in line):
                continue
            line_ = line.split(",")
            freqs += [float(line_[0])]
            s11 += [float(line_[1])]

        return np.array(freqs).astype('float'), np.array(s11).astype('float')

    vswr_template = np.zeros(401)
    avg = []

    for file_name in file_names:
        if ("127" in file_name):
            continue

        freqs, s11 = open_s11(file_name)
        if (len(avg) == 0):
            avg = s11
        else:
            avg += s11

    s11 = pow(10, s11 / 20)

    freq_range = freqs[-1] - freqs[1]  # Hz

    P_thermal = np.zeros(len(freqs))

    for count, freq in enumerate(freqs):
        T_noise_antenna = (1 - pow(s11[count], 2)) * T_ice

        P_thermal[count] = kB * freq_range * T_noise_antenna

    R = 50 #Ohm
    V_rms_thermal_noise = np.sqrt(P_thermal*R)
    V_rms_thermal_noise = np.interp(freq_range_input*1e9, freqs, V_rms_thermal_noise)

    ampl_noise = np.zeros(len(freq_range_input))
    rng = np.random.default_rng()
    for f in range(0, len(freq_range_input)):
        ampl_noise[f] = rng.rayleigh(V_rms_thermal_noise[f], 1)

    # add random phase
    noise = np.array(ampl_noise, dtype='complex')
    phases = np.random.rand(len(ampl_noise)) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)

    noise *= phases
    return noise

def get_noise_figure_IGLU_DRAP(freq_range_input, time_trace_length = 2048):
    # freq_range_input in GHz
    file_names_cold = glob.glob("/Users/alisanozdrina/Documents/phys/exp/rno/data/IGLU_2020/IGLU_1/P_cold/*csv*")
    file_names_hot = glob.glob("/Users/alisanozdrina/Documents/phys/exp/rno/data/IGLU_2020/IGLU_1/P_hot/*csv*")
    def open_noise_file(file_name):
        f = open(file_name)
        freqs = []
        P = []

        for line in f:
            if ("!" in line or "Freq(Hz),B Log Mag(dBm)" in line or "BEGIN" in line or "END" in line or line == "\n"):
                continue
            line_ = line.split(",")
            freqs += [float(line_[0])]
            P += [float(line_[1])]

        return np.array(freqs).astype('float'), np.array(P).astype('float')

    avg = []

    for file_name in file_names_cold:
        if ("127" in file_name):
            continue

        freqs, P_cold = open_noise_file(file_name)

        if (len(avg) == 0):
            avg = P_cold
        else:
            avg += P_cold

    avg = []
    for file_name in file_names_hot:
        if ("127" in file_name):
            continue

        freqs, P_hot = open_noise_file(file_name)

        if (len(avg) == 0):
            avg = P_hot
        else:
            avg += P_hot

    Y = P_hot / P_cold
    ENR = 0.35

    NF = abs(ENR / (Y - 1))

    T_amp = 220 #K
    kB = 1.380649 * 10e-23
    freq_range = freqs[-1] - freqs[1]  # Hz

    P_amplif_noise = np.zeros(len(freqs))

    for count, freq in enumerate(freqs):
        T_noise_amp = (NF[count] - 1) * T_amp
        P_amplif_noise[count] = kB * freq_range * T_noise_amp

    R = 50  # Ohm
    V_rms_amp_noise = np.sqrt(P_amplif_noise * R)
    V_rms_amp_noise = np.interp(freq_range_input * 1e9, freqs, V_rms_amp_noise)

    ampl_noise = np.zeros(len(freq_range_input))
    for f in range(0, len(freq_range_input)):
        mu, sigma = 0, V_rms_amp_noise[f]  # mean and standard deviation
        ampl_noise[f] = np.random.normal(mu, sigma, 1)

    noise = np.array(ampl_noise, dtype='complex')
    phases = np.random.rand(len(ampl_noise)) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)

    noise *= phases
    return noise