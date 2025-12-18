import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from matplotlib import colors
import matplotlib.animation as animation
from tqdm import tqdm
import time
import matplotlib.patches as mpatches
class Agent(object):

    def __init__(self,pos,strategy,c1,c2):
        self._pos = pos # of form (i,j)
        self._strategy = strategy # coop, ch1 or ch2
        self._payoff = 0.0 #baseline fitness (?)


        if self._strategy == 'coop':  #cooperator produces both goods
            self._good_1 = c1   
            self._good_2 = c2
            self._cval = 0

        elif self._strategy == 'ch1': #cheater 1 produces only good 2 
            self._good_1 = 0
            self._good_2 = c2
            self._cval = 1

        elif self._strategy == 'ch2': #cheater 2 produces only good 1
            self._good_1 = c1
            self._good_2 = 0
            self._cval = 2

        else:
            raise ValueError('Not a valid strategy')

    def get_strategy(self):
        return self._strategy

    def get_payoff(self):
        return self._payoff

    def reset_payoff(self):
        self._payoff = 0.0

    def get_color(self):
        return self._cval

    def get_pos(self):
        return self._pos

    def __str__(self):
        return "Position : " + f"{self._pos}" + ";" + "Strategy : " + f"{self._strategy}"


class Population(object):

    def __init__(self,size): 
        self._size = size
        self._agents = []
        self._agent_map={}
        self._neighbours = {} # maintaining cache of neighbours of each agent
        
    def get_agents(self):
        return self._agents

    def add_agent(self,agent):
        pos = agent.get_pos()
        if pos in self._agent_map:
            raise ValueError("Already in the population")
        else:
            self._agents.append(agent)
            self._agent_map[pos] = agent #position mapping

    def initialize_population(self,init_comp,c1,c2): #init_comp must be of form [freq_coop, freq_ch1, freq_ch2] such that entries sum up to 1

        strategies = ['coop','ch1','ch2']
        if init_comp:
            probabilities = init_comp #initial population composition specified in terms of frequencies

        else:
            probabilities = [1/3,1/3,1/3]
        
        positions = []
        for i in range(self._size):
            for j in range(self._size):
                positions.append((i,j))

        for pos in positions:
            strat = np.random.choice(strategies,size=1,p=probabilities)
            self.add_agent(Agent(pos,strat[0],c1,c2))

    def build_neighbour_cache(self):
        for agent in self._agents:
            self._neighbours[agent] = self.get_moore_neighbours(agent)

    def initialize_from_matrix(self, strategy_matrix, c1, c2): 
        """
        Providing a position matrix consisting of 0,1,2 (COOP,CH1,CH2) respectively to initalize population with a 
        defined spatial structure
        """
        if strategy_matrix.shape != (self._size, self._size):
            raise ValueError("Matrix shape does not match population size")

        mapping = {0: 'coop', 1: 'ch1', 2: 'ch2'}

        for i in range(self._size):
            for j in range(self._size):
                strat = mapping[strategy_matrix[i, j]]
                self.add_agent(Agent((i, j), strat,c1,c2))

    def get_moore_neighbours(self,agent):
        """
        Returns the 8 neighbours of an agent
        NOTE : Boundary conditions here are periodic
        """
        
        x,y = agent.get_pos()
        L = self._size
        neighbour_positions = []

        for dx in [-1,0,1]:
             for dy in [-1,0,1]:
                if dx==0 and dy ==0:
                      continue
                nx = (x+dx)%L
                ny = (y+dy)%L
                neighbour_positions.append(self._agent_map[(nx,ny)])

        return neighbour_positions

    def get_focal_group(self,agent):
        return [agent]+self._neighbours[agent]
    
    def global_good1_benefit(self,r1):
        N = len(self._agents)
        total_c1 = sum(a._good_1 for a in self._agents)
        return r1*total_c1/N

    def global_good2_benefit(self,r2):
        N = len(self._agents)
        total_c2 = sum(a._good_2 for a in self._agents)
        return r2*total_c2/N

    def local_good1_payoff(self,group,r1):
        total_c1 = sum(a._good_1 for a in group)
        return r1*total_c1/len(group)

    def local_good2_payoff(self,group,r2):
        total_c2 = sum(a._good_2 for a in group)
        return r2*total_c2/len(group)
    
# Accumulating payoffs when T_reproduction =/= T_games

    def accumulate_payoffs_localized(self,r1,r2):
        for agent in self._agents:
            agent._payoff += self.local_total_payoff(agent,r1,r2)

    def accumulate_payoffs_hybrid(self,r1,r2):
        global_benefit = self.global_good2_benefit(r2)
        for agent in self._agents:
            agent._payoff += self.hybrid_total_payoff(agent,r1,r2,global_benefit)

    def accumulate_payoffs_homo(self,r1,r2):

        gb1 = self.global_good1_benefit(r1)
        gb2 = self.global_good2_benefit(r2)
        
        for agent in self._agents:
            agent._payoff += self.homo_total_payoff(agent,gb1,gb2)


# Total payoff of a single agent when playing with either local or global goods
    def hybrid_total_payoff(self,agent,r1,r2,global_benefit_2):
        local_payoff_1 = 0.0
        focal_agents = [agent] + self._neighbours[agent]
        
        for focal in focal_agents:
            group = self.get_focal_group(focal)

            local_payoff_1+=self.local_good1_payoff(group,r1) #good_1 -> local
        
        return ((local_payoff_1/len(focal_agents)) + global_benefit_2 - (agent._good_1 + agent._good_2) )

    def homo_total_payoff(self,agent,global_benefit_1,global_benefit_2):
        
        return global_benefit_1 + global_benefit_2 - (agent._good_1 + agent._good_2)


    def local_total_payoff(self,agent,r1,r2): #r's are the synergy factor
        payoff = 0.0
        focal_agents = [agent] + self._neighbours[agent]

        for focal in focal_agents:
            group = self.get_focal_group(focal)
            payoff += (self.local_good1_payoff(group,r1) + self.local_good2_payoff(group,r2))
            
        return (payoff/len(focal_agents)) - (agent._good_1 + agent._good_2)
    
    def copy_strategy(self, agent, model, c1, c2):
        # Directly copy the string to avoid logic errors
        new_strat = model.get_strategy()
        agent._strategy = new_strat
    
    # Manually map production values and colors
        if new_strat == 'coop':
            agent._good_1, agent._good_2, agent._cval = c1, c2, 0
        elif new_strat == 'ch1':
            agent._good_1, agent._good_2, agent._cval = 0, c2, 1
        elif new_strat == 'ch2':
            agent._good_1, agent._good_2, agent._cval = c1, 0, 2
    
#MC updates for games + reproductive events
# done based on Fermi update rule

    def hybrid_monte_carlo_update(self,r1,r2,K,n_interactions, c1, c2,w=0.5):

        for _ in range(n_interactions):
            self.accumulate_payoffs_hybrid(r1,r2)

        agents = np.random.permutation(self._agents)
        for agent in agents:
            if np.random.rand()<w:
                neighbours = self._neighbours[agent]
                model = np.random.choice(neighbours)

                Pi = agent._payoff
                Pj = model._payoff 

                prob = 1.0 / (1.0 + np.exp((Pi-Pj)/K))

                if np.random.rand() < prob:
                    self.copy_strategy(agent,model,c1,c2)

        for agent in self._agents:
            agent.reset_payoff()

    def homo_monte_carlo_update(self,r1,r2,K,n_interactions,c1,c2,w=0.5):
        
        for _ in range(n_interactions):
            self.accumulate_payoffs_homo(r1,r2)

        agents = np.random.permutation(self._agents)
        for agent in agents:
            if np.random.rand()<w:
                neighbours = self._neighbours[agent]
                model = np.random.choice(neighbours)

                Pi = agent._payoff
                Pj = model._payoff 
            

                prob = 1.0 / (1.0 + np.exp((Pi-Pj)/K))

                if np.random.rand() < prob:
                    self.copy_strategy(agent,model,c1,c2)

        for agent in self._agents:
            agent.reset_payoff()

    def localized_monte_carlo_update(self,r1,r2,K,n_interactions,c1,c2,w=0.5):

        for _ in range(n_interactions):
            self.accumulate_payoffs_localized(r1,r2)

        agents = np.random.permutation(self._agents)

        for agent in agents:
            if np.random.rand()<w:
                neighbours = self._neighbours[agent]
                model = np.random.choice(neighbours)

                Pi = agent._payoff
                Pj = model._payoff

                prob = 1.0/(1.0 + np.exp((Pi-Pj)/K))

                if np.random.rand() < prob:
                    self.copy_strategy(agent,model,c1,c2)

        for agent in self._agents:
            agent.reset_payoff()

    def pop_composition(self):
        """
        Returns Frequencies of Strategies
        
        """
        coop,ch1,ch2 = 0,0,0
        N = (self._size)**2
        for agent in self._agents:
            if agent.get_strategy() == 'coop':
                coop+=1
            elif agent.get_strategy() == 'ch1':
                ch1+=1
            elif agent.get_strategy() == 'ch2':
                ch2+=1

        return [coop/N, ch1/N, ch2/N]


    def lattice(self,show=False):
        """
        Returns spatial distribution of strategies
        Can visualize if needed
        """
        lattice = np.zeros((self._size,self._size))
        agents = self._agents
        for agent in agents:
            pos = agent.get_pos()
            x,y=pos[0],pos[1]
            lattice[x][y] = agent.get_color()

        if show:
            fig,ax = plt.subplots(figsize=(6,6))
            cmap  = colors.ListedColormap(['#004488','#DDAA33','#BB5566'])
            ax.imshow(lattice,cmap=cmap,origin='upper')
            plt.title("Population")
            plt.show()

        return lattice



def run_simulation_homo(sim_params,file_name):
    """
    Runs simulation for set number of frames/generations
    Saves the animation as  file_name.mp4
    """
    pop_size = sim_params['N']
    r1 = sim_params['synergy_factor_1']
    r2 = sim_params['synergy_factor_2']
    K = sim_params['noise']
    init_comp = sim_params['composition']
    c1 = sim_params['cost_good_1']
    c2 = sim_params['cost_good_2']
    init_matrix = sim_params['init_arrangement']
    sim_type = sim_params['sim_type']
    num_frames = sim_params['num_frames']

    
    
    print("Initializing Population...")
    population = Population(pop_size)
    if sim_type == 'frequency':
        population.initialize_population(init_comp=init_comp,c1=c1,c2=c2)

    else:
        population.initialize_from_matrix(init_matrix,c1=c1,c2=c2)
    print("Building Neighbour Cache...")
    population.build_neighbour_cache()

    init_frame = population.lattice()
    
    print("Running simulations...")
    N = pop_size
    freq_series=[]

    lattice_snapshot = population.lattice()
    unique, counts = np.unique(lattice_snapshot, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    print("-" * 30)
    print("DIAGNOSTIC CHECK (Before Animation):")
    print(f"Lattice Color Counts (Should see 0.0 for Coops): {count_dict}")
    
    freqs = population.pop_composition()
    print(f"Strategy Frequencies (Coop, Ch1, Ch2): {freqs}")
    
    if freqs[0] > 0 and 0.0 not in count_dict:
        print("CRITICAL WARNING: Cooperators exist in data, but have wrong color code!")
    elif 0.0 in count_dict:
        print(f"CONFIRMED: There are {count_dict[0.0]} Blue pixels ready to plot.")
    print("-" * 30)
    # ------------------------

    print("Animating...")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title(f"Two global goods ; r1={r1}, r2={r2} ; c1/c2 = {c1/c2}")

    colors_list = ['#004488','#DDAA33','#BB5566']
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    img = ax.imshow(
    population.lattice(),
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        population.homo_monte_carlo_update(r1, r2, K, 5, c1, c2)
        lattice = population.lattice()
        freq_series.append(population.pop_composition())
        img.set_data(population.lattice())
        return (img,)
    
    ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=False )
    
    ani.save(f"{file_name}.mp4", writer="ffmpeg", fps=30)
    plt.show()
    return np.array(freq_series)

def run_simulation_hybrid(sim_params,file_name):
    """
    Runs simulation for set number of frames/generations
    Saves the animation as  file_name.mp4
    """
    pop_size = sim_params['N']
    r1 = sim_params['synergy_factor_1']
    r2 = sim_params['synergy_factor_2']
    K = sim_params['noise']
    init_comp = sim_params['composition']
    c1 = sim_params['cost_good_1']
    c2 = sim_params['cost_good_2']
    init_matrix = sim_params['init_arrangement']
    sim_type = sim_params['sim_type']
    num_frames = sim_params['num_frames']

    
    
    print("Initializing Population...")
    population = Population(pop_size)
    if sim_type == 'frequency':
        population.initialize_population(init_comp=init_comp,c1=c1,c2=c2)

    else:
        population.initialize_from_matrix(init_matrix,c1=c1,c2=c2)
    print("Building Neighbour Cache...")
    population.build_neighbour_cache()

    init_frame = population.lattice()
    
    print("Running simulations...")
    N = pop_size
    freq_series=[]

    lattice_snapshot = population.lattice()
    unique, counts = np.unique(lattice_snapshot, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    print("-" * 30)
    print("DIAGNOSTIC CHECK (Before Animation):")
    print(f"Lattice Color Counts (Should see 0.0 for Coops): {count_dict}")
    
    freqs = population.pop_composition()
    print(f"Strategy Frequencies (Coop, Ch1, Ch2): {freqs}")
    
    if freqs[0] > 0 and 0.0 not in count_dict:
        print("CRITICAL WARNING: Cooperators exist in data, but have wrong color code!")
    elif 0.0 in count_dict:
        print(f"CONFIRMED: There are {count_dict[0.0]} Blue pixels ready to plot.")
    print("-" * 30)
    # ------------------------

    print("Animating...")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title(f"Mixed type global goods ; r1(local)={r1}, r2(global)={r2} ; c1/c2 = {c1/c2}")

    colors_list = ['#004488','#DDAA33','#BB5566']
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    img = ax.imshow(
    population.lattice(),
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        population.hybrid_monte_carlo_update(r1, r2, K, 5, c1, c2)
        lattice = population.lattice()
        freq_series.append(population.pop_composition())
        img.set_data(population.lattice())
        return (img,)
    
    ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=False )
    
    ani.save(f"{file_name}.mp4", writer="ffmpeg", fps=30)
    plt.show()
    return np.array(freq_series)

def run_simulation_localized(sim_params,file_name):
    """
    Runs simulation for set number of frames/generations
    Saves the animation as  file_name.mp4
    """
    pop_size = sim_params['N']
    r1 = sim_params['synergy_factor_1']
    r2 = sim_params['synergy_factor_2']
    K = sim_params['noise']
    init_comp = sim_params['composition']
    c1 = sim_params['cost_good_1']
    c2 = sim_params['cost_good_2']
    init_matrix = sim_params['init_arrangement']
    sim_type = sim_params['sim_type']
    num_frames = sim_params['num_frames']

    
    
    print("Initializing Population...")
    population = Population(pop_size)
    if sim_type == 'frequency':
        population.initialize_population(init_comp=init_comp,c1=c1,c2=c2)

    else:
        population.initialize_from_matrix(init_matrix,c1=c1,c2=c2)
    print("Building Neighbour Cache...")
    population.build_neighbour_cache()

    init_frame = population.lattice()
    
    print("Running simulations...")
    N = pop_size
    freq_series=[]

    lattice_snapshot = population.lattice()
    unique, counts = np.unique(lattice_snapshot, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    print("-" * 30)
    print("DIAGNOSTIC CHECK (Before Animation):")
    print(f"Lattice Color Counts (Should see 0.0 for Coops): {count_dict}")
    
    freqs = population.pop_composition()
    print(f"Strategy Frequencies (Coop, Ch1, Ch2): {freqs}")
    
    if freqs[0] > 0 and 0.0 not in count_dict:
        print("CRITICAL WARNING: Cooperators exist in data, but have wrong color code!")
    elif 0.0 in count_dict:
        print(f"CONFIRMED: There are {count_dict[0.0]} Blue pixels ready to plot.")
    print("-" * 30)
    # ------------------------

    print("Animating...")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title(f"Two Local goods ; r1={r1}, r2={r2} ; c1/c2 = {c1/c2}")

    colors_list = ['#004488','#DDAA33','#BB5566']
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    img = ax.imshow(
    population.lattice(),
    cmap=cmap,
    norm=norm,
    interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        population.localized_monte_carlo_update(r1, r2, K, 5, c1, c2)
        lattice = population.lattice()
        freq_series.append(population.pop_composition())
        img.set_data(population.lattice())
        return (img,)
    
    ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=False )
    
    ani.save(f"{file_name}.mp4", writer="ffmpeg", fps=30)
    plt.show()
    return np.array(freq_series)




