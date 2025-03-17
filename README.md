---

# DiffTaichi Jumping Robot

This repository contains a DiffTaichi simulation project for a human-skeleton robot that jumps using open-loop control. The project builds upon previous labs by evolving the robot's shape and topology and then introducing time‑based actuation to generate rhythmic jumping behavior.  

## Overview

In this project, we simulate a 2D human-skeleton robot constructed from rigid boxes (representing the torso, head, and legs) connected by springs. The system’s dynamics are simulated using DiffTaichi—a high-performance differentiable programming framework.  

### Key Features

- **Procedural Skeleton Generation:**  
  The robot is built in a modular way using a function (`build_human_skeleton()`) that creates the geometric structure and spring-based joints.
  
- **Open-Loop Control:**  
  Actuation is provided via a simple sinusoidal function that modulates the spring rest lengths (and thus the forces) over time. This open‑loop control strategy creates a rhythmic jumping pattern.

- **Differentiable Simulation & Optimization:**  
  A gradient‑based optimization loop adjusts parameters (e.g., knee spring stiffness) to improve performance. The loss function consists of multiple terms including head height, torso stability, velocity, and geometric regularization.

- **Visualization & Training Plot:**  
  The simulation outputs animation frames, and a training plot is generated using Matplotlib to show the evolution of the loss and the optimized stiffness values.

## Setup & Requirements

### Software

- **Python 3.10+**
- **DiffTaichi** (version 1.7.3 or later)
- **NumPy**
- **Matplotlib**

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/shayan-shabani/difftaichi-jumping-robot.git
   cd difftaichi-jumping-robot
   ```

2. **Install Dependencies:**

   It is recommended to use a virtual environment. Then install the required packages:
   
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install taichi numpy matplotlib
   ```

## How to Run

The main simulation script is `rigid_body.py`.

### Running the Simulation

To run the jumping simulation:
  
```bash
python rigid_body.py 3 jump
```

Here, the first argument (3) is the robot ID (choose from available configurations in `robot_config.py`), and the second argument is the command (`jump` to run the optimization/training loop with open‑loop control).

After training, the final simulation output is saved under a folder named `rigid_body/final_jump/`. A training progress plot (`training_progress.png`) is also generated.

## Code Structure

- **rigid_body.py:**  
  Contains the main simulation code. Key sections include:
  
  - **Field Setup & Initialization:**  
    Defines state variables such as positions (`x`), velocities (`v`), rotations, and spring properties.
  
  - **Physics Kernels:**  
    Implements collision handling (`collide`), spring force application (`apply_spring_force`), and state advancement (`advance_no_toi`).
  
  - **Loss & Optimization:**  
    Multiple kernels compute the loss based on head height, torso stability, and more. The `optimize_jump()` function runs the optimization loop using gradient descent.
  
  - **Visualization:**  
    Uses Taichi’s GUI and Matplotlib to visualize simulation states and training progress.

- **robot_config.py:**  
  (Not shown here) Contains functions for building the robot’s geometry and topology, notably `build_human_skeleton()`, which defines the positions of the torso, head, and legs, as well as the connections (springs) between them.

## Experimentation & Results

- **Jumping Behavior:**  
  The robot exhibits a stable jumping pattern driven by the open‑loop (sinusoidal) actuation of its knee springs. Occasional floating with extended legs is observed, but overall, the behavior is significantly more stable than previous iterations.

- **Training Curve:**  
  The loss function oscillates periodically due to the nature of the sinusoidal control signal. While the loss values alternate between two levels, they remain bounded, and the optimization converges by adjusting the knee spring stiffness within the set limits.  

- **Final Observations:**  
  The open‑loop approach (α(t)=sin(ωt)) proved effective. The robot meets the project’s goal by repeatedly performing a jump without uncontrollable behavior. Further improvements might include averaging the loss over several time steps or incorporating additional damping.

## Discussion

- **Controllable Parameters:**  
  The knee spring stiffness is the primary parameter optimized using gradient descent. This meets the requirement of having at least one controllable parameter.

- **Open‑Loop Control:**  
  Our use of a sinusoidal actuation function meets the minimum requirement for open‑loop control.  
  *Future work may explore closed‑loop control using gradient descent on control signals or reinforcement learning.*

- **Stability & Challenges:**  
  While the overall jumping pattern is stable, occasional anomalies (e.g., leg floating) persist. The oscillatory loss is an expected outcome of periodic control. Future experiments could smooth the loss function or add additional penalties.

## Future Directions

- **Closed‑Loop Control:**  
  Incorporate feedback to adapt control signals based on real-time observations.
- **Enhanced Loss Functions:**  
  Consider averaging the loss over multiple phases of the jump to achieve a smoother training curve.
- **Parameter Tuning:**  
  Further refine the damping, friction, and actuation parameters to mitigate unwanted mid-air behavior.
- **Extended Simulation:**  
  Expand the topology or add more degrees of freedom (e.g., arms) for more complex locomotion.

## Acknowledgments

This project was developed as part of the COMP_SCI 302 course. Special thanks to the DiffTaichi community and resources for making differentiable simulation accessible and efficient.

---
