import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import fft
from scipy import signal
from scipy.ndimage import minimum_filter1d, maximum_filter1d

class STDP_Data_Processing:
    def __init__(self, excel_file:str, peak_voltage=None, reading_voltage=None, offsets=None):
        '''The assumption is that the reading pulses are in channel 1 with the pre-synaptic pulses and Ch2 contains only the post synaptic pulses'''
        self.raw_data = pd.read_excel(excel_file,"Data")
        self.voltage1 = self.raw_data.VMeasCh1.to_numpy()
        self.voltage2 = self.raw_data.VMeasCh2.to_numpy()
        self.current1 = self.raw_data.IMeasCh1.to_numpy()
        self.current2 = self.raw_data.IMeasCh2.to_numpy()
        self.time_series = self.raw_data.TimeOutput.to_numpy()
        self.FFT_voltage1, self.FFT_voltage2 = self.FFT_Voltage()
        self.dt = self.time_series[1] - self.time_series[0] # time between samples

        if peak_voltage == None:
            self.peak_voltage = self.Determine_peak_voltage()
        else:
            self.peak_voltage = peak_voltage
        
        self.reading_wave, self.pre_synaptic_wave, self.index_reading_wave = self.Extract_reading_pulse()
        if reading_voltage == None:
            self.reading_voltage = self.Determine_reading_voltage()
        else:
            self.reading_voltage = reading_voltage

        self.post_synaptic_wave = self.voltage2 # rename for convenience
        

        self.read_current1 = self.current1 * self.index_reading_wave
        self.read_current2 = self.current2 * self.index_reading_wave

        self.collate_trials() # adds the self.trials instance

        self.delta_T = self.trials.delta_T # can approximate the changing Delta_T by using the time between each readin g pulse
        self.Resistances = 1

    # Visualization Functions:
    
    def view_current1(self):
        plt.plot(self.time_series,self.current1)
        plt.title("Current Ch1")
        plt.show()
    def view_current2(self):
        plt.plot(self.time_series,self.current2)
        plt.title("Current Ch2")
        plt.show()
    def view_voltage1(self):
        plt.plot(self.time_series,self.voltage1)
        plt.title("Voltage Ch1")
        plt.show()
    def view_voltage2(self):
        plt.plot(self.time_series,self.voltage2)
        plt.title("Voltage Ch2")
        plt.show()
    def view_voltages(self):
        plt.plot(self.time_series,self.voltage1, label="Voltage 1")
        plt.plot(self.time_series,self.voltage2, label = "Voltage 2")
        plt.title("Voltages")
        plt.legend()
        plt.show()
    def view_reading_pulses(self):
        plt.plot(self.time_series,self.reading_wave, label="Reading Pulse")
        plt.title("Reading Pulse")
        plt.legend()
        plt.show()
    def view_triangle_pulses(self):
        plt.plot(self.time_series,self.triangle_wave, label="Synaptic Pulse")
        plt.title("Synaptic Pulse")
        plt.legend()
        plt.show()

    # Data Wrangling Functions:


    def collate_trials(self):
        # each trial occurs between a reading pulse; so we can separate the time-series into sections based on the reading pulses
        start_count = 0
        trials = []

        for j, ind in enumerate(self.index_reading_wave[1:]):
            i = j+1 # because we sloce the index_reading_wave

            if j != len(self.index_reading_wave)-2:
                if start_count == 0 and ind==1 and self.index_reading_wave[i+1]==0:
                    start_count = 1
                    start_index = i

            if start_count == 1 and ind == 1 and self.index_reading_wave[i-1]==0:
                trials.append((start_index,i)) # add a tuple for the range of indicies 
                start_count = 0

        self.trials = collated_data(self,trials)

        '''plt.plot(self.trials.Trial_11[:,1],self.trials.Trial_11[:,2])
        plt.plot(self.trials.Trial_11[:,1],self.trials.Trial_11[:,3])
        plt.show()'''

    def Determine_peak_voltage(self):
        '''Estimate the crest height of the STDP Pulse from the data'''
        peaks = signal.find_peaks(self.voltage2, height=0.8*np.max(self.voltage2), prominence=0.8*np.max(self.voltage2))[0] # Returns indicies in the array; set the minimal peak height so that we don't detect local maxima in the noise
        return np.mean(self.voltage2[peaks])
    
    def Extract_reading_pulse(self):

        d = np.abs(np.append(0.,self.voltage1[1:] - self.voltage1[:-1]) / self.dt)
        
        switch = 0
        square_est = [self.voltage1[0]]
        switch_history = [0.]
        peaks = signal.find_peaks(d,threshold=200.)[0]

        for i in range(1,len(d)):
            
            if (i in peaks) and (np.abs(self.voltage1[i]) <  0.5*self.peak_voltage) and d[i+1] < 1000. and d[i-1] < 1000.:
                switch = (switch + 1) % 2 # 1-> 0; 0-> 1
            switch_history.append(switch)
            if switch:
                square_est.append(self.voltage1[i])
            else:
                square_est.append(0.)

        square_est = np.array(square_est)
        switch_history = np.array(switch_history)

        triangle_est = self.voltage1 - square_est


        return square_est, triangle_est, switch_history

    def Determine_reading_voltage(self):
        return np.average(self.reading_wave[self.reading_wave>0.])

    def FFT_Voltage(self):
        FFTed_Voltage1 = fft.fft(self.voltage1)
        FFTed_Voltage2 = fft.fft(self.voltage2)

        return FFTed_Voltage1, FFTed_Voltage2

    def Compute_delta_T(self):
        delta_T = []
        start_count = 0

        for j, ind in enumerate(self.index_reading_wave[1:]):
            i = j+1 # because we sloce the index_reading_wave

            if j != len(self.index_reading_wave)-2:
                if start_count == 0 and ind==1 and self.index_reading_wave[i+1]==0:
                    start_count = 1
                    start_time = self.time_series[i]

            if start_count == 1 and ind == 1 and self.index_reading_wave[i-1]==0:
                delta_T.append(self.time_series[i] - start_time)
                start_count = 0
        
        return np.array(delta_T)




    def delta_w_percent(self, R_before:np.ndarray, R_after:np.ndarray):
        G_before = 1/R_before # conductivity before the STDP pulse (previous reading pulse)
        G_after = 1/R_after # conductivity after STDP pulse (current reading pulse)
        R_min = np.min(self.Resistances) # normalization factor
        return (G_after - G_before) * R_min * 100.


class collated_data:
        def __init__(self, outerself:STDP_Data_Processing, trials:tuple[int,int]):
            self.delta_T = [] # this is the distance between peaks of the pre- and post- (post-pre) 
            for i in range(len(trials)):
                self.__dict__["Trial_%i"%i] = np.column_stack([np.arange(trials[i][0],trials[i][1],1),outerself.time_series[trials[i][0]:trials[i][1]], outerself.pre_synaptic_wave[trials[i][0]:trials[i][1]], outerself.post_synaptic_wave[trials[i][0]:trials[i][1]]])
                # first column is the range of indicies, second column is the time series, and so on
                pre_peak = self.__dict__["Trial_%i"%i][np.argmax(self.__dict__["Trial_%i"%i][:,2]),1]
                post_peak = self.__dict__["Trial_%i"%i][np.argmax(self.__dict__["Trial_%i"%i][:,3]),1]
                self.delta_T.append(post_peak - pre_peak)

            self.delta_T = np.array(self.delta_T)

            # Now we want to extract the channel one currents over each reading pulse:
            count = 0
            rolling_sum = 0.
            current_reads = []
            for i in range(len(outerself.index_reading_wave)):
                if np.abs(outerself.read_current1[i]) <= 1e-16:
                    if count > 0:
                        current_reads.append(rolling_sum/count)
                    count = 0
                    rolling_sum = 0.
                else:
                    rolling_sum += outerself.read_current1[i]
                    count += 1
            self.current_reads = np.array(current_reads)


# tests
if __name__ == "__main__":
    directory = "./Data/2026-03-11 STDP Testing/"
    file_name = "F9_STDP.xls"
    file = directory + file_name
    Data = STDP_Data_Processing(file)
    #print(Data.time_series)
    #Data.view_voltages()
    #print(Data.peak_voltage)
    #Data.view_voltage2() # Ch2 is simply triangular pulses
    #print(Data.FFT_voltage2)
    #Data.view_current2()
    #Data.view_current1()
    #plt.plot(Data.time_series, Data.read_current1)
    #plt.plot(Data.time_series, Data.read_current2)
    #plt.show()
    """plt.plot(Data.read_current1)
    plt.plot(Data.read_current2)
    plt.show()"""
    #Data.view_reading_pulses()
    print(Data.delta_T)