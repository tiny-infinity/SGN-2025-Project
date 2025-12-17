from sPGG import run_simulation_homo, run_simulation_hybrid, run_simulation_localized
import numpy as np

init_matrix = np.zeros((100,100))
init_matrix[:33,:] = 0
init_matrix[33:37,33:37] = 1
init_matrix[40:55,40:55] = 2


sim_params_0 = {'N':100,'synergy_factor_1':1,'synergy_factor_2':1,'noise':10,'composition':None}
sim_params_1 = {'N':100,'synergy_factor_1':1,'synergy_factor_2':5,'noise':5.0,'composition':[0.9,0.05,0.05],'cost_good_1':1.1,'cost_good_2':1,'init_arrangement':init_matrix,'sim_type':'poop','num_frames':500}

run_simulation_homo(sim_params=sim_params_1,file_name='local_test')