"""
Simple 2D Multi-Agent Brax Environment: Double-Jointed Arms
Two agents each control a 2-DOF arm and must touch their end-effectors together.
"""

import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import html
from flax import struct
from typing import Any, Dict
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
from pathlib import Path
import numpy as np


# Define the 2D system XML for two double-jointed arms
# Using only capsules and spheres (MJX compatible)
DOUBLE_ARM_2D_SYSTEM = """
<mujoco model="double_arm_touch_2d">
  <compiler angle="radian" autolimits="true"/>
  
  <option timestep="0.02" iterations="1" ls_iterations="4"/>
  
  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom friction="1 0.5 0.5" density="1000" margin="0.01"/>
  </default>
  
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Agent 0: Left arm (blue) -->
    <body name="base_0" pos="-1.5 0 0.5">
      <geom name="base_0" type="sphere" size="0.1" rgba="0.2 0.2 0.8 1" mass="1"/>
      
      <!-- First link - constrained to 2D plane -->
      <body name="link1_0" pos="0 0 0.05">
        <joint name="shoulder_0" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_0" type="capsule" size="0.04" fromto="0 0 0 0.5 0 0" 
              rgba="0.3 0.3 0.9 1" mass="0.5"/>
        
        <!-- Second link -->
        <body name="link2_0" pos="0.5 0 0">
          <joint name="elbow_0" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_0" type="capsule" size="0.03" fromto="0 0 0 0.5 0 0" 
                rgba="0.4 0.4 1 1" mass="0.3"/>
          
          <!-- End effector -->
          <body name="endeff_0" pos="0.5 0 0">
            <geom name="endeff_0" type="sphere" size="0.06" rgba="0 0.8 0.8 1" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Agent 1: Right arm (red) -->
    <body name="base_1" pos="1.5 0 0.5">
      <geom name="base_1" type="sphere" size="0.1" rgba="0.8 0.2 0.2 1" mass="1"/>
      
      <!-- First link - constrained to 2D plane -->
      <body name="link1_1" pos="0 0 0.05">
        <joint name="shoulder_1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_1" type="capsule" size="0.04" fromto="0 0 0 -0.5 0 0" 
              rgba="0.9 0.3 0.3 1" mass="0.5"/>
        
        <!-- Second link -->
        <body name="link2_1" pos="-0.5 0 0">
          <joint name="elbow_1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_1" type="capsule" size="0.03" fromto="0 0 0 -0.5 0 0" 
                rgba="1 0.4 0.4 1" mass="0.3"/>
          
          <!-- End effector -->
          <body name="endeff_1" pos="-0.5 0 0">
            <geom name="endeff_1" type="sphere" size="0.06" rgba="0.8 0.8 0 1" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Actuators for both arms -->
  <actuator>
    <!-- Agent 0 actuators -->
    <motor name="shoulder_motor_0" joint="shoulder_0" gear="40"/>
    <motor name="elbow_motor_0" joint="elbow_0" gear="30"/>
    
    <!-- Agent 1 actuators -->
    <motor name="shoulder_motor_1" joint="shoulder_1" gear="40"/>
    <motor name="elbow_motor_1" joint="elbow_1" gear="30"/>
  </actuator>
</mujoco>
"""


class DoubleArmTouch2D(PipelineEnv):
    """
    Multi-agent 2D environment where two double-jointed arms try to touch end-effectors.
    
    Observation space (per agent): 10-dimensional
      - Own joint positions (2)
      - Own joint velocities (2)
      - Own end-effector position (2D: x, z)
      - Other end-effector position (2D: x, z)
      - Distance vector between end-effectors (2D: x, z)
    
    Action space (per agent): 2-dimensional (continuous)
      - Shoulder joint torque [-1, 1]
      - Elbow joint torque [-1, 1]
    """
    
    def __init__(
        self,
        backend='mjx',
        contact_reward_scale=100.0,
        distance_reward_scale=10.0,
        action_penalty_scale=0.1,
        success_threshold=0.15,
        **kwargs
    ):
        # Create MuJoCo model
        mj_model = mujoco.MjModel.from_xml_string(DOUBLE_ARM_2D_SYSTEM)
        
        # Store for rendering
        self._mj_model = mj_model
        
        # Convert to MJX if using mjx backend
        if backend == 'mjx':
            sys = mjx.put_model(mj_model)
        else:
            sys = mj_model
            
        self.contact_reward_scale = contact_reward_scale
        self.distance_reward_scale = distance_reward_scale
        self.action_penalty_scale = action_penalty_scale
        self.success_threshold = success_threshold
        
        # Number of agents
        self.num_agents = 2
        
        # Joint indices for each agent
        self.agent_joint_indices = {
            0: jnp.array([0, 1]),  # shoulder_0, elbow_0
            1: jnp.array([2, 3]),  # shoulder_1, elbow_1
        }
        
        # Body indices for end-effectors
        self.endeff_body_indices = {
            0: mj_model.body('endeff_0').id,
            1: mj_model.body('endeff_1').id,
        }
        
        super().__init__(sys, backend=backend, n_frames=4)
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment to initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        # Create initial pipeline state
        pipeline_state = self.pipeline_init(
            q=jnp.zeros(self.sys.nq),
            qd=jnp.zeros(self.sys.nv)
        )
        
        # Get multi-agent observations
        obs = self._get_obs(pipeline_state)
        
        # Calculate initial reward (should be 0)
        reward = jnp.zeros(self.num_agents)
        
        # Initial metrics
        metrics = {
            'distance': self._compute_distance(pipeline_state),
            'success': 0.0,
        }
        
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=0.0,
            metrics=metrics
        )
    
    def step(self, state: State, action: Dict[int, jax.Array]) -> State:
        """
        Step the environment.
        
        Args:
            state: Current environment state
            action: Dict mapping agent_id -> action array
                    action is shape (2,) for each agent
        """
        # Combine actions from both agents into single control vector
        ctrl = jnp.zeros(4)  # 4 total actuators (2 per agent)
        ctrl = ctrl.at[0:2].set(action[0])  # Agent 0 actions
        ctrl = ctrl.at[2:4].set(action[1])  # Agent 1 actions
        
        # Step physics
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        
        # Get observations for both agents
        obs = self._get_obs(pipeline_state)
        
        # Compute rewards for both agents
        reward, metrics = self._compute_reward(
            pipeline_state, 
            action
        )
        
        # Check if done (not used in continuous tasks, but included for completeness)
        done = 0.0
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics
        )
    
    def _get_obs(self, pipeline_state: Any) -> Dict[int, jax.Array]:
        """Get observations for each agent (2D version)."""
        obs_dict = {}
        
        # Get end-effector positions (only x and z, ignore y)
        endeff_pos_0_full = pipeline_state.x.pos[self.endeff_body_indices[0]]
        endeff_pos_1_full = pipeline_state.x.pos[self.endeff_body_indices[1]]
        
        # Extract 2D positions (x, z)
        endeff_pos_0 = jnp.array([endeff_pos_0_full[0], endeff_pos_0_full[2]])
        endeff_pos_1 = jnp.array([endeff_pos_1_full[0], endeff_pos_1_full[2]])
        
        # Distance vector (2D)
        distance_vec = endeff_pos_1 - endeff_pos_0
        
        for agent_id in range(self.num_agents):
            joint_indices = self.agent_joint_indices[agent_id]
            
            # Own joint state
            joint_pos = pipeline_state.q[joint_indices]
            joint_vel = pipeline_state.qd[joint_indices]
            
            # Own end-effector position
            own_endeff = endeff_pos_0 if agent_id == 0 else endeff_pos_1
            other_endeff = endeff_pos_1 if agent_id == 0 else endeff_pos_0
            
            # Combine into observation (2D version)
            obs = jnp.concatenate([
                joint_pos,          # 2
                joint_vel,          # 2
                own_endeff,         # 2 (x, z)
                other_endeff,       # 2 (x, z)
                distance_vec,       # 2 (x, z)
            ])  # Total: 10 dimensions
            
            obs_dict[agent_id] = obs
        
        return obs_dict
    
    def _compute_distance(self, pipeline_state: Any) -> jax.Array:
        """Compute 2D distance between end-effectors."""
        endeff_pos_0 = pipeline_state.x.pos[self.endeff_body_indices[0]]
        endeff_pos_1 = pipeline_state.x.pos[self.endeff_body_indices[1]]
        
        # 2D distance (x and z only)
        dx = endeff_pos_1[0] - endeff_pos_0[0]
        dz = endeff_pos_1[2] - endeff_pos_0[2]
        return jnp.sqrt(dx**2 + dz**2)
    
    def _compute_reward(
        self, 
        pipeline_state: Any,
        action: Dict[int, jax.Array]
    ) -> tuple[jax.Array, Dict[str, Any]]:
        """Compute rewards for both agents."""
        
        # Distance between end-effectors
        distance = self._compute_distance(pipeline_state)
        
        # Reward for getting close (shared between agents)
        distance_reward = -self.distance_reward_scale * distance
        
        # Bonus reward when within threshold (contact)
        contact_bonus = jnp.where(
            distance < self.success_threshold,
            self.contact_reward_scale,
            0.0
        )
        
        # Action penalty (encourage smooth movements)
        action_penalty_0 = -self.action_penalty_scale * jnp.sum(action[0] ** 2)
        action_penalty_1 = -self.action_penalty_scale * jnp.sum(action[1] ** 2)
        
        # Total rewards (shared cooperative task)
        shared_reward = distance_reward + contact_bonus
        reward_0 = shared_reward + action_penalty_0
        reward_1 = shared_reward + action_penalty_1
        
        rewards = jnp.array([reward_0, reward_1])
        
        # Metrics for logging
        metrics = {
            'distance': distance,
            'success': jnp.where(distance < self.success_threshold, 1.0, 0.0),
            'distance_reward': distance_reward,
            'contact_bonus': contact_bonus,
        }
        
        return rewards, metrics


def create_mujoco_visualization(mj_model, states, save_path="double_arm_visualization.html"):
    """
    Create visualization using MuJoCo's native rendering (fixed encoding).
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Double Arm Touch Visualization</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Double Arm Touch Simulation (2D)</h1>
        <div class="info">
            <p><strong>Simulation completed with {len(states)} timesteps</strong></p>
            <p>Note: Interactive 3D visualization for MJX backend requires additional setup.</p>
            <p>Check the generated PNG plots for metrics visualization!</p>
        </div>
        <h2>Simulation Stats:</h2>
        <ul>
            <li>Total frames: {len(states)}</li>
            <li>Physics backend: MJX (MuJoCo XLA)</li>
            <li>Model bodies: {mj_model.nbody}</li>
            <li>Model joints: {mj_model.njnt}</li>
            <li>Dimension: 2D (constrained to XZ plane)</li>
            <li>Geometry: Capsules and spheres (MJX compatible)</li>
        </ul>
    </div>
</body>
</html>
"""
    
    # Write with UTF-8 encoding to avoid emoji issues
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def visualize_rollout(env, states, save_path="double_arm_visualization.html"):
    """
    Create an interactive HTML visualization of the rollout.
    """
    print(f"\nGenerating visualization...")
    
    # Convert states to the format expected
    if isinstance(states, list):
        qps = [s.pipeline_state for s in states]
    else:
        qps = states.pipeline_state
    
    try:
        # Use MuJoCo rendering
        if hasattr(env, '_mj_model'):
            mj_model = env._mj_model
        else:
            mj_model = env.sys
        
        print("Using MuJoCo rendering (MJX backend detected)")
        create_mujoco_visualization(mj_model, qps, save_path)
        
        print(f"Visualization saved to: {save_path}")
        print(f"Open this file in a web browser to view the info!")
        
    except Exception as e:
        print(f"Warning: Could not generate HTML visualization: {e}")


def plot_metrics(states, save_path="metrics_plot.png"):
    """
    Plot distance, rewards, and success over time.
    """
    # Extract metrics
    distances = [float(s.metrics['distance']) for s in states]
    successes = [float(s.metrics['success']) for s in states]
    rewards_0 = [float(s.reward[0]) for s in states]
    rewards_1 = [float(s.reward[1]) for s in states]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    timesteps = range(len(states))
    
    # Plot 1: Distance over time
    ax1 = axes[0]
    ax1.plot(timesteps, distances, 'b-', linewidth=2)
    ax1.axhline(y=0.15, color='r', linestyle='--', label='Success Threshold', linewidth=2)
    ax1.set_ylabel('Distance (m)', fontsize=11)
    ax1.set_title('End-Effector Distance Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Rewards
    ax2 = axes[1]
    ax2.plot(timesteps, rewards_0, 'b-', linewidth=2, label='Agent 0', alpha=0.7)
    ax2.plot(timesteps, rewards_1, 'r-', linewidth=2, label='Agent 1', alpha=0.7)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Agent Rewards Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Success indicator
    ax3 = axes[2]
    ax3.fill_between(timesteps, 0, successes, alpha=0.5, color='green')
    ax3.set_ylabel('Success', fontsize=11)
    ax3.set_xlabel('Timestep', fontsize=11)
    ax3.set_title('Success Indicator (End-Effectors Touching)', fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {save_path}")
    plt.close()


def plot_2d_trajectory(states, save_path="trajectory_2d.png"):
    """
    Plot the 2D trajectory of both end-effectors.
    """
    # Extract end-effector positions
    env_temp = states[0]
    
    # Get body indices
    positions_0 = []
    positions_1 = []
    
    for state in states:
        # Extract from pipeline state
        if hasattr(state.pipeline_state, 'x'):
            pos_0 = state.pipeline_state.x.pos[3]  # endeff_0 body index
            pos_1 = state.pipeline_state.x.pos[7]  # endeff_1 body index
            
            positions_0.append([float(pos_0[0]), float(pos_0[2])])  # x, z
            positions_1.append([float(pos_1[0]), float(pos_1[2])])  # x, z
    
    positions_0 = np.array(positions_0)
    positions_1 = np.array(positions_1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trajectories
    ax.plot(positions_0[:, 0], positions_0[:, 1], 'b-', linewidth=2, label='Agent 0 (Blue)', alpha=0.6)
    ax.plot(positions_1[:, 0], positions_1[:, 1], 'r-', linewidth=2, label='Agent 1 (Red)', alpha=0.6)
    
    # Mark start and end positions
    ax.scatter(positions_0[0, 0], positions_0[0, 1], c='blue', s=100, marker='o', 
               edgecolors='black', linewidths=2, label='Agent 0 Start', zorder=5)
    ax.scatter(positions_0[-1, 0], positions_0[-1, 1], c='blue', s=100, marker='s', 
               edgecolors='black', linewidths=2, label='Agent 0 End', zorder=5)
    
    ax.scatter(positions_1[0, 0], positions_1[0, 1], c='red', s=100, marker='o', 
               edgecolors='black', linewidths=2, label='Agent 1 Start', zorder=5)
    ax.scatter(positions_1[-1, 0], positions_1[-1, 1], c='red', s=100, marker='s', 
               edgecolors='black', linewidths=2, label='Agent 1 End', zorder=5)
    
    # Mark base positions
    ax.scatter(-1.5, 0.5, c='lightblue', s=200, marker='D', 
               edgecolors='black', linewidths=2, label='Base 0', zorder=4)
    ax.scatter(1.5, 0.5, c='lightcoral', s=200, marker='D', 
               edgecolors='black', linewidths=2, label='Base 1', zorder=4)
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Z Position (m)', fontsize=11)
    ax.set_title('2D Trajectory of End-Effectors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D trajectory plot saved to: {save_path}")
    plt.close()


def run_with_visualization(
    env,
    rng,
    num_steps=500,
    policy=None,
    save_html=True,
    save_plots=True
):
    """
    Run environment with visualization.
    """
    print("=" * 70)
    print("Running 2D Double Arm Touch Environment with Visualization")
    print("=" * 70)
    
    # Reset environment
    state = jax.jit(env.reset)(rng)
    states = [state]
    
    print(f"\nInitial observations:")
    print(f"  Agent 0 obs shape: {state.obs[0].shape} (2D: 10 dimensions)")
    print(f"  Agent 1 obs shape: {state.obs[1].shape} (2D: 10 dimensions)")
    print(f"  Initial distance: {state.metrics['distance']:.3f}m")
    
    # JIT compile step function
    step_fn = jax.jit(env.step)
    
    # Run rollout
    print(f"\nRunning {num_steps}-step rollout...")
    successes = 0
    
    for i in range(num_steps):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        if policy is None:
            # Random actions
            actions = {
                0: jax.random.uniform(rng1, (2,), minval=-1.0, maxval=1.0),
                1: jax.random.uniform(rng2, (2,), minval=-1.0, maxval=1.0),
            }
        else:
            # Use provided policy
            actions = {
                0: policy(state.obs[0], rng1),
                1: policy(state.obs[1], rng2),
            }
        
        state = step_fn(state, actions)
        states.append(state)
        
        if state.metrics['success']:
            successes += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{num_steps}: "
                  f"Distance = {state.metrics['distance']:.3f}m, "
                  f"Success = {state.metrics['success']}, "
                  f"Cumulative successes = {successes}")
    
    # Summary statistics
    distances = [float(s.metrics['distance']) for s in states]
    min_distance = min(distances)
    avg_distance = sum(distances) / len(distances)
    success_rate = successes / num_steps
    
    print(f"\n{'=' * 70}")
    print("Rollout Summary:")
    print(f"{'=' * 70}")
    print(f"  Total steps: {num_steps}")
    print(f"  Minimum distance achieved: {min_distance:.3f}m")
    print(f"  Average distance: {avg_distance:.3f}m")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Total successes: {successes}")
    
    # Generate visualizations
    if save_html:
        visualize_rollout(env, states)
    
    if save_plots:
        plot_metrics(states)
        plot_2d_trajectory(states)
    
    return states


def simple_policy_2d(obs, rng):
    """
    A simple policy that tries to move arms toward each other (2D version).
    """
    # Extract distance vector (last 2 elements of observation for 2D)
    distance_vec = obs[-2:]  # Only x and z
    
    # Simple proportional control toward target
    target_0 = jnp.tanh(distance_vec[0] * 2.0)  # Shoulder (x direction)
    target_1 = jnp.tanh(jnp.linalg.norm(distance_vec) - 0.5)  # Elbow (distance)
    
    # Add small noise
    noise = jax.random.normal(rng, (2,)) * 0.1
    
    action = jnp.array([target_0, target_1]) + noise
    action = jnp.clip(action, -1.0, 1.0)
    
    return action


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D Double Arm Touch Environment - Visualization Demo")
    print("=" * 70 + "\n")
    
    # Create environment
    print("Creating 2D environment...")
    env = DoubleArmTouch2D(backend='mjx')
    
    print(f"* Environment created with {env.num_agents} agents")
    print(f"* Observation space: 10D (2D positions and velocities)")
    print(f"* Action space: 2D per agent (shoulder, elbow torques)")
    print(f"* Geometry: Capsules and spheres (MJX compatible)")
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Run with random actions
    print("\n" + "-" * 70)
    print("Demo 1: Random Actions")
    print("-" * 70)
    states_random = run_with_visualization(
        env,
        rng,
        num_steps=300,
        policy=None,
        save_html=True,
        save_plots=True
    )
    
    # Run with simple heuristic policy
    print("\n" + "-" * 70)
    print("Demo 2: Simple Heuristic Policy")
    print("-" * 70)
    rng, _ = jax.random.split(rng)
    states_policy = run_with_visualization(
        env,
        rng,
        num_steps=300,
        policy=simple_policy_2d,
        save_html=True,
        save_plots=True
    )
    
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - double_arm_visualization.html (simulation info)")
    print("  - metrics_plot.png (distance, rewards, success over time)")
    print("  - trajectory_2d.png (2D path of end-effectors)")