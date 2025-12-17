import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from matplotlib import colors
import matplotlib.animation as animation


class Agent(object):

    def __init__(self,pos,strategy,c1,c2):
        self._pos = pos # of form (i,j)
        self._strategy = strategy # coop, ch1 or ch2
        self._payoff = 5.0 #baseline fitness (?)


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
            agent._payoff += self.total_payoff(agent,r1,r2)

    def accumulate_payoffs_hybrid(self,r1,r2):
        global_benefit = self.global_good2_benefit(r2)
        for agent in self._agents:
            agent._payoff += self.hybrid_total_payoff(agent,r1,r2,global_benefit)

    def accumulate_payoffs_mixed(self,r1,r2):
        global_benefit = self.global_good1_benefit(r1) + self.global_good2_benefit(r1)
        for agent in self._agents:
            agent._payoff += self.homo_total_payoff(agent,r1,r2,global_benefit)


# Total payoff of a single agent when playing with either local or global goods
    def hybrid_total_payoff(self,agent,r1,r2,global_benefit):
        payoff = 0.0
        focal_agents = [agent] + self._neighbours[agent]

        for focal in focal_agents:
            group = self.get_focal_group(focal)
            payoff+=self.local_good1_payoff(group,r1)

        payoff -= agent._good_1 
        payoff -= agent._good_2 
        payoff += global_benefit

        return payoff/len(focal_agents)

    def homo_total_payoff(self,agent,r1,r2,global_benefit):
        payoff = 0.0
        focal_agents = [agent] + self._neighbours[agent]

        payoff -= (agent._good_1 + agent._good_2)
        payoff += global_benefit

        return payoff/len(focal_agents)

    def total_payoff(self,agent,r1,r2): #r's are the synergy factor
        payoff = 0.0
        focal_agents = [agent] + self._neighbours[agent]

        for focal in focal_agents:
            group = self.get_focal_group(focal)
            payoff+=self.local_good1_payoff(group,r1) + self.local_good2_payoff(group,r2)

        payoff -= (agent._good_1 + agent._good_2)

        return payoff/len(focal_agents)


#MC updates for games + reproductive events
# done based on Fermi update rule
    def hybrid_monte_carlo_update(self,r1,r2,K,n_interactions, c1, c2):

        for _ in range(n_interactions):
            self.accumulate_payoffs_hybrid(r1,r2)

        agents = np.random.permutation(self._agents)
        for agent in agents:
            neighbours = self._neighbours[agent]
            model = np.random.choice(neighbours)

            Pi = agent._payoff
            Pj = model._payoff 

            prob = 1.0 / (1.0 + np.exp((Pi-Pj)/K))

            if np.random.rand() < prob:
                agent.__init__(agent.get_pos(), model.get_strategy(), c1, c2)

        for agent in self._agents:
            agent.reset_payoff()

    def homo_monte_carlo_update(self,r1,r2,K,n_interactions,c1,c2):
        
        for _ in range(n_interactions):
            self.accumulate_payoffs_mixed(r1,r2)

        agents = np.random.permutation(self._agents)
        for agent in agents:
            neighbours = self._neighbours[agent]
            model = np.random.choice(neighbours)

            Pi = agent._payoff
            Pj = model._payoff 

            prob = 1.0 / (1.0 + np.exp((Pi-Pj)/K))

            if np.random.rand() < prob:
                agent.__init__(agent.get_pos(), model.get_strategy(), c1, c2)

        for agent in self._agents:
            agent.reset_payoff()

    def localized_monte_carlo_update(self,r1,r2,K,n_interactions,c1,c2):

        for _ in range(n_interactions):
            self.accumulate_payoffs_mixed(r1,r2)

        agents = np.random.permutation(self._agents)

        for agent in agents:
            neighbours = self._neighbours[agent]
            model = np.random.choice(neighbours)

            Pi = agent._payoff
            Pj = model._payoff

            prob = 1.0/(1.0 + np.exp((Pi-Pj)/K))

            if np.random.rand() < prob:
                agent.__init__(agent.get_pos(), model.get_strategy(), c1, c2)

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

        return (coop/N, ch1/N, ch2/N)


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
            cmap  = colors.ListedColormap(['blue','green','pink'])
            ax.imshow(lattice,cmap=cmap,origin='upper')
            plt.title("Population")
            plt.show()

        return lattice


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

    fig,ax=plt.subplots(figsize=(12,6))
    cmap  = colors.ListedColormap(['blue','green','pink'])

    population = Population(pop_size)
    population.initialize_population(init_comp)
    population.build_neighbour_cache()

    init_frame = population.lattice()
    img = ax.imshow(init_frame,cmap=cmap)

    def update(frame):
        population.hybrid_monte_carlo_update(r1,r2,K,10,c1,c2)
        new_data = population.lattice()
        img.set_data(new_data)
        return [img]
    
    ani = animation.FuncAnimation(fig,update,frames=300,interval=100,blit=True)
    ani.save(f'{file_name}.mp4',writer='ffmpeg',fps=30)
    plt.show()
   

sim_params_0 = {'N':100,'synergy_factor_1':1,'synergy_factor_2':1,'noise':10,'composition':None}
sim_params_1 = {'N':100,'synergy_factor_1':5,'synergy_factor_2':2,'noise':1.0,'composition':None}

if __name__=="main":
    run_simulation_hybrid(sim_params_1,"standard_1")


