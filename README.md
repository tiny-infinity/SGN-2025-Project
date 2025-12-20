Project : Studying the Spatial Dynamics of a Dual Public Goods Game
Members : Govind R Nair (IISER Pune), Meghana Sangle (IISER Mohali), Sidharth Raman (IISER Bhopal)

### On the Project
Presentation can be found at [here](https://docs.google.com/presentation/d/1K_j9Tua2dNFYAFXXLDS3wloYGc0FBdb4GVRL0thYv_k/edit?usp=sharing)

Here we have used Agent-based modelling to study the spatial dynamics of cooperators and cheaters (of two different types/for two different goods) on a lattice.
There are two public goods - with costs $c_1$ and $c_2$. The three strategies involved in our games are:
- Wild Type/Generalist : Produces and bears the cost for both goods
- Cheater/Specialist 1 : Produces only good 1
- Cheater/Specialist 2 : Produces only good 2

#### global Public Goods Game
In the global case, goods produced by an individual can be diffused to any individual anywhere on the lattice. This represents homogenous, well-mixed conditions. Benefits obtained by an individual are independent of its position and neighbours. In an iteration, the payoff of agent i is  given by:

$$
P_{i}^{(global)}=(\dfrac{r_{1}\sum_{k=1}^{N}c_{1,k}}{N})+(\dfrac{r_{2}\sum_{k=1}^{N}c_{2,k}}{N})-(c_{1,i}+c_{2,i})
$$

where $r_1$ and $r_2$ are the synergy factors of the goods.

#### local Public Goods Game
In the local case, goods produced by an individual can be shared only with its neighbours. In our program, the neighbours of an agent are the agents in its' Moore Neighbourhood (1 on each cardinal direction + 4 diagonal elements = 8). This alters the dynamics since the benefit of an individual is a function of its neighbourhood and position on the lattice. Payoffs given by:

$$
P_{i}^{(local)}={\dfrac{1}{M}\sum_{j \in G_{i} \cup {i}}(\dfrac{r_{1}\sum_{k \in G_{j}}c_{1,k}}{M} + \dfrac{r_{2}\sum_{k \in G_{j}}c_{2,k}}{M})}-(c_{1,i}+c_{2,i})
$$


#### Model Specifications
- Boundary Conditions : For convenience, we have used periodic BCs, which means that this model is simulated on a torus. 
- Generation Time : 5 rounds of games are played before a reproduction event.
- Strategy Updates : Strategy of an agent is updated by a randomly picking a neighbour and computing an update probability using the Fermi Update Rule.

### How to Use

`sPGG.py` contains the main program modules. `run_simulation_homo`, `run_simulation_localized` and `run_simulation_hybrid` are the execution functions to which you provide a paramater set `sim_params` in the format of a dictionary. 

#### Parameters:
- `N` : Population given by N * N
- `cost_good_1`,`cost_good_2` : costs of goods 1 and 2 respectively
- `r1`, `r2` : synergy factors of goods 1 and 2 respectively
- `K` : Noise parameter for Fermi Update
- `num_frames` : Number of generations to run
- `sim_type` : 'frequency' if the only initial conditions you wish to specify are the population composition. Entering anything else will mean you want to specify spatial coordinates
- `composition` : [$f_{wt}$,$f_{ch1}$,$f_{ch2}$]. Entries should sum to 1. If `None` then all strategies are assumed to be equally common. 
- `init_matrix` : Initial population with the positions specified by strategy IDs (0-WT, 1-ch1, 2-ch2)
- `save_movie` : Boolean
- `show_anim` : Boolean




