from wave_gen import postsynaptic_pulse_spike, presynaptic_pulse_spike, save_synaptic_pulses
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------
# For STDP waveforms
#--------------------------------------------------------------------------

delta_T = -1. # time lag between pre- and post-pulses 
delta_t0_p = 15.
delta_t2_p = 0.
delta_t3_p = 2.
delta_t4_p = 2.
delta_t5_p = 8.

V_start = 0. # start voltage
V_max = 7. # maximum voltage
V_min = -7. #minimum voltage 
offset = 10. # padding time in between pulses
vis = True # if True ==> graph the results
same = True

t0 = -8. # start time
t1 = -2.
t2 = -2.
t3 = 0.
t4 = 0.
t5 = 6. # end time

if same == 0:
    t1_p = t1+delta_T
    t0_p = t1_p-delta_t0_p
    t2_p = t1_p+delta_t2_p
    t3_p = t1_p+delta_t3_p
    t4_p = t1_p+delta_t4_p
    t5_p = t1_p+delta_t5_p

else: # same == 1
    t1_p = t1+delta_T
    t0_p = t0+delta_T
    t2_p = t2+delta_T
    t3_p = t3+delta_T
    t4_p = t4+delta_T
    t5_p = t5+delta_T


times = [t0,t1,t2,t3,t4,t5]
post_times = [t0_p,t1_p,t2_p,t3_p,t4_p,t5_p]

V = presynaptic_pulse_spike(V_max,V_min, times,offset)

V_prime = postsynaptic_pulse_spike(V_max,V_min,post_times, offset+delta_T)

save_synaptic_pulses("synaptic_pulses.csv",V,V_prime)


if vis == 1:
    pre_times = [0.]
    post_times = [0.]
    pre_voltage = [0.]
    post_voltages = [0.]
    for i in range(V.shape[0]):
        pre_times.append(V[i,-1]+pre_times[-1])
        post_times.append(V_prime[i,-1]+post_times[-1])
        pre_voltage.append(V[i,-2])
        post_voltages.append(V_prime[i,-2])

    pre_times[-1] = max(pre_times[-1],post_times[-1])
    post_times[-1] = pre_times[-1]

    plt.plot(pre_times,pre_voltage)
    plt.plot(post_times,post_voltages)
    plt.title('Sample STDP pre and post waveforms')
    plt.legend(['Post','Pre'],loc='upper right')

    plt.grid(True)
    plt.savefig("Figure_1.png")
    plt.show()