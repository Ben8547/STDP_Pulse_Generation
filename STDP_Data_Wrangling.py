import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import fft

class STDP_Data_Processing:
    def __init__(self, excel_file:str):
        self.raw_data = pd.read_excel(excel_file,"Data")
        self.voltage1 = self.raw_data.VMeasCh1.to_numpy()
        self.voltage2 = self.raw_data.VMeasCh2.to_numpy()
        self.current1 = self.raw_data.IMeasCh1.to_numpy()
        self.current2 = self.raw_data.IMeasCh2.to_numpy()
        self.time_series = self.raw_data.TimeOutput.to_numpy()
        self.FFT_voltage1, self.FFT_voltage2 = self.FFT_Voltage()

        self.reading_pulses = self.Locate_Reading_Pulses()
        self.delta_T = 1
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

    # Data Wrangling Functions:

    def FFT_Voltage(self):
        FFTed_Voltage1 = fft.fft(self.voltage1)
        FFTed_Voltage2 = fft.fft(self.voltage2)

        return FFTed_Voltage1, FFTed_Voltage2

    def Locate_Reading_Pulses(self):
        # the idea is to use the FFT of the voltage to deconvolute the two wave forms; thus extracting the square and triangle waves
        reading_pulses = 1
        return reading_pulses

    def Compute_delta_T(self):
        delta_T = 1
        return delta_T




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
    print(Data.time_series)
    Data.view_voltages()
    #Data.view_voltage2() # Ch2 is simply triangular pulses
    print(Data.FFT_voltage2)