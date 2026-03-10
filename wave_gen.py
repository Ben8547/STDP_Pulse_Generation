import numpy as np

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



#-------------------------------------
# Define a presynaptic pulse spike
#-------------------------------------

def presynaptic_pulse_spike(V_max:float, V_min:float, discerete_times:list, offset:float) -> np.ndarray:

    t0, t1, t2, t3, t4, t5 = discerete_times

    V = np.array([
        [0., 0., offset],
        [0., V_max, t1-t0],
        [V_max, V_min, t3-t2],
        [V_min,0.,t5-t4],
        [0.,0.,offset]
    ])
        
    return V

#-------------------------------------
# Define a postsynaptic pulse spike
#-------------------------------------

def postsynaptic_pulse_spike(V_max:float, V_min:float, post_times: list, offset:float) -> np.ndarray:

    t0_p, t1_p, t2_p, t3_p, t4_p, t5_p = post_times

    V_prime = np.array([
        [0., 0., offset],
        [0., V_max, t1_p-t0_p],
        [V_max, V_min, t3_p-t2_p],
        [V_min,0.,t5_p-t4_p],
        [0.,0.,offset]
    ])
        
    return V_prime

    #-------------------------------------
    # Get output points as start channel voltage, end channel voltage, segment
    # time


def save_synaptic_pulses(file_path:str,pre_voltages:list,post_voltages:list) -> None: 
    headers = ["pre-voltage (V)", "post-voltage(V)","rise time (us)"]
    header_str = ','.join(headers)
    np.savetxt("pre-"+file_path,pre_voltages,fmt="%16f",delimiter=",",header=header_str)
    np.savetxt("post-"+file_path,post_voltages,fmt="%16f",delimiter=",",header=header_str)