import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# clear all
# clear 

# % %--------------------------------------------------------------------------
# % % For triangle waveforms
# % % -------------------------------------------------------------------------
# % N = 20;
# % d = 0.01;
# % % p1 = [0:0.1:1];
# % % p2 = [0.9:-0.1:0];
# % 
# % V = [];
# % 
# % for i = 1:1:N
# %     if mod(i,2) == 1
# %         p1 = [0:0.1:0.9];
# %         V = cat(2,V,p1);
# %     
# %     else
# %         p2 = [1:-0.1:0.1];
# %         V = cat(2,V,p2);
# %     end
# % end
# % 
# % 
# % if mod(N,2) == 1
# %     
# %     V = [V,1];
# % else
# %     V = [V,0];
# % end
# % 
# % [r,c] = size(V);
# % 
# % t = [0:d:(c-1)*d];
# % 
# % p = [t',V'];
# % 
# % % plot(p(:,1),p(:,2))
# % % hold on

# % %--------------------------------------------------------------------------
# % % For Square waves
# % %--------------------------------------------------------------------------
# % N = 5; % Number of square pulses
# % N_points = 20; % Number of points in each section of the square wave
# % delta_t = 0.01; % Time spacing
# % t_w = 0.01; % How much time the pulse is on
# % t_o = 0.01; % How much time the pulse is off (t_w + t_o is one period)
# % 
# % V = [];
# % 
# % for i=1:1:5
# %     [r,c] = size(0:(t_w)/(N_points-1):t_w);
# %     V = cat(2,V,ones(1,c));
# %     
# %     [r,c] = size(0:(t_o)/(N_points-1):t_o);
# %     V = cat(2,V,zeros(1,c));
# % end
# % V = [0,V];
# % 
# % t = [0:delta_t:(size(V,2)-1)*delta_t];
# % p = [t',V'];
# % 
# % plot(p(:,1),p(:,2))

#--------------------------------------------------------------------------
# For STDP waveforms
#--------------------------------------------------------------------------

delta_T = -1 # time lag between pre- and post-pulses 
V_start = 0 # start voltage
V_max = 7 # maximum voltage
V_min = -7 #minimum voltage 
N = 5 # partitions per time interval
offset = 10
vis = 1 # if 1 - graph the results
same = 1

t0 = -8 # start time
t1 = -2
t2 = -2
t3 = 0
t4 = 0
t5 = 6 # end time

if same == 0:
    delta_t0_p = 15
    delta_t2_p = 0
    delta_t3_p = 2
    delta_t4_p = 2
    delta_t5_p = 8

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

#-------------------------------------
# Create the time array
#-------------------------------------

t_pre = np.linspace(t0,t5,5*N)
t_post = np.linspace(t0_p,t5_p,5*N)

t_o = min([t0,t0_p]) # lowest time
t_f = max([t5,t5_p])  # highest time
endpoint = max([abs(t_o),abs(t_f)])
t_before = np.linspace(t_o-offset,t_o,N+15)
t_after = np.linspace(t_f,t_f+offset,N+15)

temp = np.concatenate((t_before,t_pre,t_post,t_after))
temp = np.sort(temp)

for n in range(len(temp)):
    if n == 0:
        t = np.concatenate((t,[temp[n]])) if len(t)>0 else np.array([temp[n]])
    else:
        if temp[n-1] == temp[n]:
            continue
        else:
            t = np.concatenate((t,[temp[n]]))
#-------------------------------------
# Define a presynaptic pulse spike
#-------------------------------------

p = []
V = []

m1 = (V_max-V_start)/(t1-t0)
b1 = V_max - m1*t1
m2 = (V_min-V_max)/(t3-t2)
b2 = V_min - m2*t3
m3 = (V_start-V_min)/(t5-t4)
b3 = V_start - m3*t5

def f(x):
    return m1*x+b1
def g(x):
    return m2*x+b2
def h(x):
    return m3*x+b3

for n in range(len(t)):
    if (t[n]>=t0) and (t[n]<t1):
        V = np.append(V,f(t[n])) if len(V)>0 else np.array([float(f.subs(x,t[n]))])
    elif (t[n]>=t1) and (t[n]<t2):
        V = np.concatenate((V,[V_max]))
    elif (t[n]>=t2) and (t[n]<t3):
        V = np.concatenate((V,[float(g.subs(x,t[n]))]))
    elif (t[n]>=t3) and (t[n]<t4):
        V = np.concatenate((V,[V_min]))
    elif (t[n]>=t4) and (t[n]<t5):
        V = np.concatenate((V,[float(h.subs(x,t[n]))]))
    else:
        if (t[n]>=t_o-2*(t_o-(t_o-offset))/(N-1)) and (t[n]<t_o):
            V = np.concatenate((V,[0]))
        elif (t[n]>t_f) and (t[n]<=t_f+2*((t_f+offset)-t_f)/(N-1)):
            V = np.concatenate((V,[0]))
        else:
            V = np.concatenate((V,[V_start]))

p = np.column_stack((t,V))

#-------------------------------------
# Define a postsynaptic pulse spike
#-------------------------------------

def postsynaptic_pulse_spike(vis):

    p_prime = []
    V_prime = []

    m1_p = (V_max-V_start)/(t1_p-t0_p)
    b1_p = V_max - m1_p*t1_p
    m2_p = (V_min-V_max)/(t3_p-t2_p)
    b2_p = V_min - m2_p*t3_p
    m3_p = (V_start-V_min)/(t5_p-t4_p)
    b3_p = V_start - m3_p*t5_p

    f_p = m1_p*x+b1_p
    g_p = m2_p*x+b2_p
    h_p = m3_p*x+b3_p

    for n in range(len(t)):
        if (t[n]>=t0_p) and (t[n]<t1_p):
            V_prime = np.concatenate((V_prime,[float(f_p.subs(x,t[n]))])) if len(V_prime)>0 else np.array([float(f_p.subs(x,t[n]))])
        elif (t[n]>=t1_p) and (t[n]<t2_p):
            V_prime = np.concatenate((V_prime,[V_max]))
        elif (t[n]>=t2_p) and (t[n]<t3_p):
            V_prime = np.concatenate((V_prime,[float(g_p.subs(x,t[n]))]))
        elif (t[n]>=t3_p) and (t[n]<t4_p):
            V_prime = np.concatenate((V_prime,[V_min]))
        elif (t[n]>=t4_p) and (t[n]<t5_p):
            V_prime = np.concatenate((V_prime,[float(h_p.subs(x,t[n]))]))
        else:
            V_prime = np.concatenate((V_prime,[V_start]))
    p_prime = np.column_stack((t,V_prime))

    #-------------------------------------
    # Get output points as start channel voltage, end channel voltage, segment
    # time
    result_pre = []
    for i in range(1,len(p)):
        seg_time = (p[i,0]-p[i-1,0])*10**(-6)
        start_V = p[i-1,1]
        end_V = p[i,1]
        
        row = np.array([start_V,end_V,seg_time])
        result_pre = np.vstack((result_pre,row)) if len(result_pre)>0 else np.array([row])

    result_post = []
    for i in range(1,len(p_prime)):
        seg_time = (p_prime[i,0]-p_prime[i-1,0])*10**(-6)
        start_V = p_prime[i-1,1]
        end_V = p_prime[i,1]
        
        row = np.array([start_V,end_V,seg_time])
        result_post = np.vstack((result_post,row)) if len(result_post)>0 else np.array([row])

    result_diff = []
    for i in range(1,len(p)):
        seg_time = (p[i,0]-p[i-1,0])*10**(-6)
        start_V = p_prime[i-1,1]-p[i-1,1]
        end_V = p_prime[i,1]-p[i,1]
        
        row = np.array([start_V,end_V,seg_time])
        result_diff = np.vstack((result_diff,row)) if len(result_diff)>0 else np.array([row])

    #-------------------------------------
    if vis == 1:
        plt.plot(p_prime[:,0],p_prime[:,1])
        plt.plot(p[:,0],p[:,1])
        plt.title('Sample STDP pre and post waveforms')
        plt.legend(['Post','Pre'],loc='upper right')

        plt.grid(True)
        plt.show()