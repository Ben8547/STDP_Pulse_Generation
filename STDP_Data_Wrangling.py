import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import fft
from scipy import signal
from scipy.ndimage import minimum_filter1d, maximum_filter1d

class STDP_Data_Processing:
    def __init__(self, excel_file:str, peak_voltage=None, reading_voltage=None, offsets=None):
        '''The assumption is that the reading pulse is in channel 1'''
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
        
        if reading_voltage == None:
            self.reading_wave, self.triangle_wave, self.index_reading_wave = self.Extract_reading_pulse()
            self.reading_voltage = self.Determine_reading_voltage()
        else:
            self.reading_voltage = reading_voltage

        self.read_current1 = self.current1 * self.index_reading_wave
        self.read_current2 = self.current2 * self.index_reading_wave
        self.delta_T = self.Compute_delta_T() # can approximate the changing Delta_T by using the time between each readin g pulse
        self.Resistances = [1]

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

    def Determine_peak_voltage(self):
        '''Estimate the crest height of the STDP Pulse from the data'''
        peaks = signal.find_peaks(self.voltage2, height=0.8*np.max(self.voltage2), prominence=0.8*np.max(self.voltage2))[0] # Returns indicies in the array; set the minimal peak height so that we don't detect local maxima in the noise
        return np.mean(self.voltage2[peaks])

    def Determine_offset(self):
        '''Offset is the time between each pre_pule and equivelently, the time between each post_pulse'''
        # We compute the effset by computing the distance between the peaks of the wave
        peaks = signal.find_peaks(self.voltage2, height=0.8*np.max(self.voltage2), prominence=0.8*np.max(self.voltage2))[0] # Returns indicies in the array; set the minimal peak height so that we don't detect local maxima in the noise
        print(self.voltage2[peaks])
        offset = 1
        return offset
    
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

            if start_count == 1 and ind == 1 and self.index_reading_wave[i-1]==0:
                delta_T.append(self.time_series[i] - start_time)
                start_count = 0
            if j != len(self.index_reading_wave)-2:
                if start_count == 0 and ind==1 and self.index_reading_wave[i+1]==0:
                    start_count = 1
                    start_time = self.time_series[i]
        
        return np.array(delta_T)




    def delta_w_percent(self, R_before:np.ndarray, R_after:np.ndarray):
        G_before = 1/R_before # conductivity before the STDP pulse (previous reading pulse)
        G_after = 1/R_after # conductivity after STDP pulse (current reading pulse)
        R_min = np.min(self.Resistances) # normalization factor
        return (G_after - G_before) * R_min * 100.



# tests
if __name__ == "__main__":
    directory = "./Data/2026-03-11 STDP Testing/"
    file_name = "F7_STDP.xls"
    file = directory + file_name
    Data = STDP_Data_Processing(file)
    #print(Data.time_series)
    #Data.view_voltages()
    #print(Data.peak_voltage)
    #Data.view_voltage2() # Ch2 is simply triangular pulses
    #print(Data.FFT_voltage2)
    #Data.view_current2()
    #Data.view_current1()
    """plt.plot(Data.read_current1)
    plt.plot(Data.read_current2)
    plt.show()"""
    #Data.view_reading_pulses()