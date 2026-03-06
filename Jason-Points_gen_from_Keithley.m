clear all
clear 

% %--------------------------------------------------------------------------
% % For triangle waveforms
% % -------------------------------------------------------------------------
% N = 20;
% d = 0.01;
% % p1 = [0:0.1:1];
% % p2 = [0.9:-0.1:0];
% 
% V = [];
% 
% for i = 1:1:N
%     if mod(i,2) == 1
%         p1 = [0:0.1:0.9];
%         V = cat(2,V,p1);
%     
%     else
%         p2 = [1:-0.1:0.1];
%         V = cat(2,V,p2);
%     end
% end
% 
% 
% if mod(N,2) == 1
%     
%     V = [V,1];
% else
%     V = [V,0];
% end
% 
% [r,c] = size(V);
% 
% t = [0:d:(c-1)*d];
% 
% p = [t',V'];
% 
% % plot(p(:,1),p(:,2))
% % hold on
% %--------------------------------------------------------------------------
% % For Square waves
% %--------------------------------------------------------------------------
% N = 5; % Number of square pulses
% N_points = 20; % Number of points in each section of the square wave
% delta_t = 0.01; % Time spacing
% t_w = 0.01; % How much time the pulse is on
% t_o = 0.01; % How much time the pulse is off (t_w + t_o is one period)
% 
% V = [];
% 
% for i=1:1:5
%     [r,c] = size(0:(t_w)/(N_points-1):t_w);
%     V = cat(2,V,ones(1,c));
%     
%     [r,c] = size(0:(t_o)/(N_points-1):t_o);
%     V = cat(2,V,zeros(1,c));
% end
% V = [0,V];
% 
% t = [0:delta_t:(size(V,2)-1)*delta_t];
% p = [t',V'];
% 
% plot(p(:,1),p(:,2))

%--------------------------------------------------------------------------
% For STDP waveforms
%--------------------------------------------------------------------------

delta_T = -1; % Time separating the pre- and postsynaptic spikes in us
V_start = 0; % Voltage starting amplitude 
V_max = 7; % Max value of the spike
V_min = -7; % Min value of the spike
N = 5; % Number of points for each linspace segment
offset = 10; % Time before the spike rise
vis = 1; % 0 for no plot shown and 1 for plot shown
same = 1; % 0 for different post and pre shape and 1 for same shape

t0 = -8; % Time at which rising begins in us
t1 = -2; % Time of the first peak
t2 = -2; % Time to end first peak plateau
t3 = 0; % Time of the second peak
t4 = 0; % Time to end second peak plateau
t5 = 6; % Time to return to starting position 

if same == 0
    delta_t0_p = 15; % Duration from t0_p to t1_p
    delta_t2_p = 0; % Duration from t1_p to t2_p
    delta_t3_p = 2; % Duration from t1_p to t3_p
    delta_t4_p = 2; % Duration from t1_p to t4_p
    delta_t5_p = 8; % Duration from t1_p to t5_p

    t1_p = t1+delta_T; % Time at which post first reaches its max value
    t0_p = t1_p-delta_t0_p;
    t2_p = t1_p+delta_t2_p;
    t3_p = t1_p+delta_t3_p;
    t4_p = t1_p+delta_t4_p;
    t5_p = t1_p+delta_t5_p;
else
    t1_p = t1+delta_T; % Time at which post first reaches its max value
    t0_p = t0+delta_T;
    t2_p = t2+delta_T;
    t3_p = t3+delta_T;
    t4_p = t4+delta_T;
    t5_p = t5+delta_T;
end

%-------------------------------------
% Create the time array
t = []; % Time points array
t_pre = [];
t_post = [];
t_pre = cat(2,t_pre,linspace(t0,t1,N),linspace(t1,t2,N),linspace(t2,t3,N),linspace(t3,t4,N),linspace(t4,t5,N));
t_post = cat(2,t_post,linspace(t0_p,t1_p,N),linspace(t1_p,t2_p,N),linspace(t2_p,t3_p,N),linspace(t3_p,t4_p,N),linspace(t4_p,t5_p,N));

t_o = min([t0,t0_p]);
t_f = max([t5,t5_p]);
endpoint = max([abs(t_o),abs(t_f)]);
t_before = linspace(t_o-offset,t_o,N+15); 
t_after = linspace(t_f,t_f+offset,N+15);

temp = cat(2,t_before,t_pre,t_post,t_after); % Combines t_pre and t_post into one 
temp = sort(temp); % Sort the combined times

for n = 1:size(temp,2)
    if n == 1
        t = cat(2,t,temp(n));
    else
        if temp(n-1) == temp(n)
            continue
        else
            t = cat(2,t,temp(n));
        end
    end
end
%-------------------------------------
% Define a presynaptic pulse spike
p = []; % Points array
V = []; % All Voltage points array

m1 = (V_max-V_start)/(t1-t0); % Calculate slope of first line
b1 = V_max - m1*t1; % Find b of first line 
m2 = (V_min-V_max)/(t3-t2); % Calculate slope of second line
b2 = V_min - m2*t3; % Find b of second line
m3 = (V_start-V_min)/(t5-t4); % Calculate slope of third line 
b3 = V_start - m3*t5; % Find b of third line

% Define Piecewise function without using piecewise
% P.a = @(x) m1*x+b1; % Rising line to first peak
% P.b = @(x) V_max; % Plateau
% P.c = @(x) m2*x+b2; % Falling line to second peak
% P.d = @(x) V_min; % Plateau
% P.e = @(x) m3*x+b3; % Rising line to V_start

syms x f(x) g(x) h(x)
f(x) = m1*x+b1; 
g(x) = m2*x+b2;
h(x) = m3*x+b3;

for n = 1:size(t,2)
    if (t(n)>=t0) && (t(n)<t1)
        V = cat(2,V,double(f(t(n))));
    elseif (t(n)>=t1) && (t(n)<t2)
        V = cat(2,V,V_max);
    elseif (t(n)>=t2) && (t(n)<t3)
        V = cat(2,V,double(g(t(n))));
    elseif (t(n)>=t3) && (t(n)<t4)
        V = cat(2,V,V_min);
    elseif (t(n)>=t4) && (t(n)<t5)
        V = cat(2,V,double(h(t(n))));
    else
        if (t(n)>=t_o-2*(t_o-(t_o-offset))/(N-1)) && (t(n)<t_o)
            V = cat(2,V,[0]);
        elseif (t(n)>t_f) && (t(n)<=t_f+2*((t_f+offset)-t_f)/(N-1))
            V = cat(2,V,[0]);
        else
            V = cat(2,V,V_start);
        end
    end
end

% for n = 1:size(t,2)
%     if (t(n)>=t0) && (t(n)<t1)
%         V = cat(2,V,P.a(t(n)));
%     elseif (t(n)>=t1) && (t(n)<t2)
%         V = cat(2,V,P.b(t(n)));
%     elseif (t(n)>=t2) && (t(n)<t3)
%         V = cat(2,V,P.c(t(n)));
%     elseif (t(n)>=t3) && (t(n)<t4)
%         V = cat(2,V,P.d(t(n)));
%     elseif (t(n)>=t4) && (t(n)<t5)
%         V = cat(2,V,P.e(t(n)));
%     else
%         V = cat(2,V,V_start);
%     end
% end
p = [t',V'];

%-------------------------------------
% Define a postsynaptic pulse spike
p_prime = []; % Points array
V_prime = []; % All Voltage points array

m1_p = (V_max-V_start)/(t1_p-t0_p); % Calculate slope of first line
b1_p = V_max - m1_p*t1_p; % Find b of first line 
m2_p = (V_min-V_max)/(t3_p-t2_p); % Calculate slope of second line
b2_p = V_min - m2_p*t3_p; % Find b of second line
m3_p = (V_start-V_min)/(t5_p-t4_p); % Calculate slope of third line 
b3_p = V_start - m3_p*t5_p; % Find b of third line

% Define piecewise function
% P_prime.a = @(x) m1_p*x+b1_p;
% P_prime.b = @(x) V_max;
% P_prime.c = @(x) m2_p*x+b2_p;
% P_prime.d = @(x) V_min;
% P_prime.e = @(x) m3_p*x+b3_p;

syms x f_p(x) g_p(x) h_p(x)
f_p(x) = m1_p*x+b1_p;
g_p(x) = m2_p*x+b2_p;
h_p(x) = m3_p*x+b3_p;

for n = 1:size(t,2)
    if (t(n)>=t0_p) && (t(n)<t1_p)
        V_prime = cat(2,V_prime,double(f_p(t(n))));
    elseif (t(n)>=t1_p) && (t(n)<t2_p)
        V_prime = cat(2,V_prime,V_max);
    elseif (t(n)>=t2_p) && (t(n)<t3_p)
        V_prime = cat(2,V_prime,double(g_p(t(n))));
    elseif (t(n)>=t3_p) && (t(n)<t4_p)
        V_prime = cat(2,V_prime,V_min);
    elseif (t(n)>=t4_p) && (t(n)<t5_p)
        V_prime = cat(2,V_prime,double(h_p(t(n))));
    else
        V_prime = cat(2,V_prime,V_start);
    end
end
p_prime = [t',V_prime'];

%-------------------------------------
% Get output points as start channel voltage, end channel voltage, segment
% time
result_pre = [];
for i = 2:size(p,1)
    seg_time = (p(i,1)-p(i-1,1))*10^(-6);
    start_V = p(i-1,2);
    end_V = p(i,2);
    
    result_pre = cat(1,result_pre,[start_V,end_V,seg_time]);
end

result_post = [];
for i = 2:size(p_prime,1)
    seg_time = (p_prime(i,1)-p_prime(i-1,1))*10^(-6);
    start_V = p_prime(i-1,2);
    end_V = p_prime(i,2);
    
    result_post = cat(1,result_post,[start_V,end_V,seg_time]);
end

result_diff = [];
for i = 2:size(p,1)
    seg_time = (p(i,1)-p(i-1,1))*10^(-6);
    start_V = p_prime(i-1,2)-p(i-1,2);
    end_V = p_prime(i,2)-p(i,2);
    
    result_diff = cat(1,result_diff,[start_V,end_V,seg_time]);
end
%-------------------------------------
if vis == 1
    plot(p_prime(:,1),p_prime(:,2))
    hold on
    plot(p(:,1),p(:,2))
    title('Sample STDP pre and post waveforms');
    legend('Post','Pre','Location','northeast');

    % fplot(g(x));
    % hold on
    % fplot(f(x));
    % % fplot(g(x)-f(x));
    grid on;
end






