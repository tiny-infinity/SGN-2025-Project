from sPGG import run_simulation_homo, run_simulation_hybrid, run_simulation_localized
import numpy as np
import matplotlib.pyplot as plt

init_matrix = np.zeros((100,100))
init_matrix[:32,:32] = 1
init_matrix[60:92,60:92] = 2



sim_params_0 = {'N':100,'synergy_factor_1':1,'synergy_factor_2':1,'noise':10,'composition':None}
sim_params_1 = {'N':100,'synergy_factor_1':5,'synergy_factor_2': 5,'noise':2.0,'composition':[0.8,0.1,0.1],'cost_good_1':1.5,'cost_good_2':1,'init_arrangement':init_matrix,'sim_type':'frequency','num_frames':500,'time_series':False}

test = run_simulation_localized(sim_params=sim_params_1,file_name='local_comp_homo')


time_arr = np.linspace(0,len(test),len(test))

test = test.T

np.save('local_comp_homo.npy',test)

plt.figure(figsize=(6,6))
plt.title("Local goods; 8:1:1 initial composition, different initial costs")
plt.plot(time_arr,test[0],label='coop',color="#004488")
plt.plot(time_arr,test[1],label='ch1',color="#DDAA33")
plt.plot(time_arr,test[2],label='ch2',color="#BB5566")
plt.ylabel("Frequencies")
plt.xlabel('Time')
plt.legend()
plt.show()
