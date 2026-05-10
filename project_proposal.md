# Mid-Semester Project Proposal: Vision-Based Continuous Drone Navigation in Wind Environments

**Authors:**
* Vishal Reddy K (vishal.kondakindi@iiitb.ac.in)
* Harsha Vardhan (Harsha.Vardhan@iiitb.ac.in)
* Vishal Sriram (Vishal.Sriram@iiitb.ac.in)
* Lohith Pasumarthi (Lohith.Pasumarthi@iiitb.ac.in)
*Note: The problem statement is adapted from the student projects section of Stanford University's CS224R course.*

## Proposed Idea

**Real-World Problem Identification**
Unmanned Aerial Vehicles (UAVs) or drones are critically useful for real-world tasks where humans cannot safely or quickly go. Specific use cases for autonomous indoor navigation include: delivering emergency medical supplies (like AEDs) down apartment corridors before paramedics arrive, navigating collapsed mine shafts and damaged buildings for search and rescue operations, and flying through the narrow aisles of massive warehouses to take automated inventory. However, deploying them in these real-world scenarios is difficult because they have to reliably fly through narrow indoor corridors while handling unpredictable winds, such as drafts from HVAC systems, open windows, or the drone's own propeller wash bouncing off the walls.

The main problem is that most existing vision-based Reinforcement Learning (RL) methods simplify the task by using discrete actions (moving step-by-step) or by assuming there is no wind during training. Because of this, the drones fly in a jerky manner and fail when tested on real hardware in windy conditions. Building upon established foundational methods, our goal is to train a single RL model entirely in simulation that uses continuous actions to smoothly fly through clutter and handle wind speeds up to $5$ m/s.

**RL Framework Formulation (POMDP)**
Since the drone makes decisions using camera images instead of knowing the exact environment map, we model this task as a Partially Observable Markov Decision Process (POMDP). The problem is defined by the following elements $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \gamma)$:
*   **Observation Space ($\mathcal{O}$):** At time step $t$, the agent receives an observation $o_t = [ I_t, s^{low}_t ]$. Here, $I_t \in \mathbb{R}^{64 \times 64 \times 3}$ is a resized RGB image from a front-facing camera, and $s^{low}_t$ is a vector containing the drone's physical data like position, speed, and yaw angle.
*   **Action Space ($\mathcal{A}$):** The agent outputs a continuous action $a_t = (v_x, v_y, v_z) \in \mathbb{R}^3$. These are the target speeds in the x, y, and z directions, which the simulator's built-in controller tries to follow.
*   **Transition Function ($\mathcal{T}$):** We use the Flightmare simulator. It calculates the realistic drone physics and adds random generated wind fields to simulate real-world disturbances.
*   **Reward Function ($\mathcal{R}$):** Getting the reward function right is a big challenge in continuous control tasks. We use a combined reward function: $R_t = r_{\text{distance}} + r_{\text{goal}} + r_{\text{time}} + r_{\text{timeout}} + r_{\text{collision}}$. The drone gets positive rewards for moving closer to the goal, but gets penalties if it crashes, takes too much time, or applies sudden huge forces.

**Proposed RL Techniques and Evaluation**
To solve this complex continuous control problem, we will test different RL algorithms starting from the basics:
1.  **Q-Learning (Baseline):** First, we will frame the problem using basic value-based methods like Q-Learning. Since Q-Learning is meant for discrete actions and our drone uses continuous speeds, we will explore methods like Deep Deterministic Policy Gradient (DDPG). This will act as our basic starting point to show the difficulties of simple methods in high-dimensional flight control.
2.  **Proximal Policy Optimization (PPO):** Our main proposed method is PPO, an advanced actor-critic algorithm. Moving beyond standard baseline implementations, we will implement a lightweight Vision Transformer (ViT) architecture to process the camera images. The self-attention mechanism of the ViT will help better capture the global spatial relationships of the indoor obstacles. The actor network will output the continuous actions, and a separate transformer-based critic network will estimate the value. 

We will test our custom PPO implementation and compare it against the standard baselines in the Flightmare simulator. We will evaluate overall performance based on average episode reward, training stability, and how smoothly the physical drone flies. We will also study the impact of important RL training techniques like advantage normalization and reward scaling.

**Environment Setup**
The experimental environment will be built using the **Flightmare** simulator, an open-source, photo-realistic quadcopter simulator specifically designed for reinforcement learning. Flightmare utilizes Unity for its rendering engine and integrates with the Robot Operating System (ROS) to provide accurate physical dynamics. We will develop a custom Python wrapper conforming to the standard OpenAI Gym interface. This wrapper will process the down-sampled continuous visual feeds, manage the continuous action outputs, and inject procedural Ornstein-Uhlenbeck wind vectors as external forces during the simulation steps to test the policy's robustness. Training the PPO algorithm within this environment will likely leverage established RL frameworks such as Stable Baselines3.

## Literature Survey
Early research in vision-based drone flight used simplified settings. Munoz et al. (2019) used a Double DQN framework combining depth images in the AirSim simulator. However, their use of discrete actions and lack of wind modeling made it hard to implement in the real world. Moving towards continuous control, Hodge et al. (2021) trained a PPO agent using curriculum learning. But, they assumed the drone had perfect control and ignored wind disturbances, leading to failure in real-world conditions. 

Recent researchers have tried to solve these remaining issues. Wu et al. (2024) handled wind disturbances by using multi-objective RL in a simulated urban environment. However, their reward design was heavily dependent on one specific map, which means the drone cannot easily adapt to new locations. Some other researchers break down the problem; Choi et al. (2023) separated the flight into different sub-tasks, while Elrod et al. (2025) used graph neural networks to control multiple drones together. Saran and Zakhor (2023) provided a very useful paper for us, as they showed that it is possible to train a vision-only deep RL model in a simulator and successfully transfer it to a real physical drone. 

In conclusion, our proposed project aims to solve the sim-to-real gap for agile drone flight under windy conditions. We do this by combining the realistic procedural wind features of the Flightmare simulator with an end-to-end, continuous-action vision-based PPO controller inspired by state-of-the-art continuous control methodologies.
