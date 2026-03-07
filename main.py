from wave_gen import postsynaptic_pulse_spike, gen_time_array, presynaptic_pulse_spike, save_synaptic_pulses
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
N = 5 # partitions per time interval
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

t, to, tf = gen_time_array(times,post_times,N,offset)

V = presynaptic_pulse_spike(V_max,V_min,V_start,times,offset,t,to,tf,N)

V_prime = postsynaptic_pulse_spike(V_max,V_min,V_start,post_times,t)

save_synaptic_pulses("synaptic_pulses.csv",t,V,V_prime)


if vis == 1:
    plt.plot(t,V)
    plt.plot(t,V_prime)
    plt.title('Sample STDP pre and post waveforms')
    plt.legend(['Post','Pre'],loc='upper right')

    plt.grid(True)
    plt.show()