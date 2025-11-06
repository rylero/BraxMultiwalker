"""
Brax Multi-Agent MultiWalker Environment
Based on PettingZoo's SISL MultiWalker environment

Multiple bipedal walkers must cooperate to carry a package to the right.
The package is placed on top of the walkers and they must coordinate their
movements to transport it without dropping it.
"""

import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import PipelineEnv, State
from typing import Any, Dict, Tuple
import mujoco
from mujoco import mjx
import matplotlib.pyplot as plt
import numpy as np


# MuJoCo XML for MultiWalker environment
def generate_multiwalker_xml(n_walkers=3, terrain_length=200):
    """Generate XML for n bipedal walkers with a package on top."""
    
    # Calculate spacing between walkers
    walker_spacing = 1.0
    package_length = (n_walkers - 1) * walker_spacing + 1.0
    
    xml = f"""
<mujoco model="multiwalker">
  <compiler angle="radian" autolimits="true"/>
  
  <option timestep="0.01" iterations="50" ls_iterations="10" gravity="0 0 -9.81"/>
  
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom friction="0.8 0.1 0.1" density="1000" margin="0.01" condim="3"/>
  </default>
  
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="{terrain_length} 5 0.1" rgba="0.7 0.7 0.7 1"/>
    
"""
    
    # Add each walker
    for i in range(n_walkers):
        x_pos = -10 + i * walker_spacing
        walker_color_r = 0.3 + (i / n_walkers) * 0.5
        walker_color_b = 0.8 - (i / n_walkers) * 0.5
        
        xml += f"""
    <!-- Walker {i} -->
    <body name="walker_{i}_torso" pos="{x_pos} 0 1.3">
      <joint name="walker_{i}_root_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="walker_{i}_root_z" type="slide" axis="0 0 1" limited="false"/>
      <joint name="walker_{i}_root_y" type="hinge" axis="0 1 0" limited="false"/>
      
      <!-- Torso -->
      <geom name="walker_{i}_torso" type="capsule" size="0.05" fromto="0 0 0 0 0 0.5" 
            rgba="{walker_color_r} 0.3 {walker_color_b} 1" mass="5"/>
      
      <!-- Head -->
      <body name="walker_{i}_head" pos="0 0 0.55">
        <geom name="walker_{i}_head" type="sphere" size="0.1" 
              rgba="{walker_color_r} 0.3 {walker_color_b} 1" mass="1"/>
      </body>
      
      <!-- Left leg -->
      <body name="walker_{i}_left_thigh" pos="0 0 0">
        <joint name="walker_{i}_left_hip" type="hinge" axis="0 1 0" range="-2.0 1.0"/>
        <geom name="walker_{i}_left_thigh" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.45"
              rgba="{walker_color_r} 0.4 {walker_color_b} 1" mass="1"/>
        
        <!-- Left shin -->
        <body name="walker_{i}_left_shin" pos="0 0 -0.45">
          <joint name="walker_{i}_left_knee" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
          <geom name="walker_{i}_left_shin" type="capsule" size="0.035" fromto="0 0 0 0 0 -0.5"
                rgba="{walker_color_r} 0.5 {walker_color_b} 1" mass="0.8"/>
          
          <!-- Left foot -->
          <body name="walker_{i}_left_foot" pos="0 0 -0.5">
            <geom name="walker_{i}_left_foot" type="capsule" size="0.03" fromto="0 0 0 0.15 0 0"
                  rgba="{walker_color_r} 0.6 {walker_color_b} 1" mass="0.5"/>
          </body>
        </body>
      </body>
      
      <!-- Right leg -->
      <body name="walker_{i}_right_thigh" pos="0 0 0">
        <joint name="walker_{i}_right_hip" type="hinge" axis="0 1 0" range="-2.0 1.0"/>
        <geom name="walker_{i}_right_thigh" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.45"
              rgba="{walker_color_r} 0.4 {walker_color_b} 1" mass="1"/>
        
        <!-- Right shin -->
        <body name="walker_{i}_right_shin" pos="0 0 -0.45">
          <joint name="walker_{i}_right_knee" type="hinge" axis="0 1 0" range="-1.0 2.0"/>
          <geom name="walker_{i}_right_shin" type="capsule" size="0.035" fromto="0 0 0 0 0 -0.5"
                rgba="{walker_color_r} 0.5 {walker_color_b} 1" mass="0.8"/>
          
          <!-- Right foot -->
          <body name="walker_{i}_right_foot" pos="0 0 -0.5">
            <geom name="walker_{i}_right_foot" type="capsule" size="0.03" fromto="0 0 0 0.15 0 0"
                  rgba="{walker_color_r} 0.6 {walker_color_b} 1" mass="0.5"/>
          </body>
        </body>
      </body>
    </body>
    
"""
    
    # Add package on top of walkers
    package_x = -10 + (n_walkers - 1) * walker_spacing / 2
    xml += f"""
    <!-- Package (carried by walkers) -->
    <body name="package" pos="{package_x} 0 2.0">
      <joint name="package_x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="package_z" type="slide" axis="0 0 1" limited="false"/>
      <joint name="package_rot" type="hinge" axis="0 1 0" limited="false"/>
      <geom name="package" type="box" size="{package_length/2} 0.15 0.1" 
            rgba="0.8 0.6 0.2 1" mass="3"/>
    </body>
  </worldbody>
  
  <actuator>
"""
    
    # Add actuators for each walker
    for i in range(n_walkers):
        xml += f"""
    <!-- Walker {i} actuators -->
    <motor name="walker_{i}_left_hip_motor" joint="walker_{i}_left_hip" gear="80" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="walker_{i}_left_knee_motor" joint="walker_{i}_left_knee" gear="80" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="walker_{i}_right_hip_motor" joint="walker_{i}_right_hip" gear="80" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="walker_{i}_right_knee_motor" joint="walker_{i}_right_knee" gear="80" ctrllimited="true" ctrlrange="-1 1"/>
"""
    
    xml += """
  </actuator>
</mujoco>
"""
    
    return xml


class MultiWalker(PipelineEnv):
    """
    Multi-agent cooperative environment where bipedal walkers carry a package.
    
    Based on PettingZoo's MultiWalker environment from SISL.
    
    Args:
        n_walkers: Number of bipedal walkers (default: 3)
        forward_reward: Scaling factor for forward progress reward (default: 1.0)
        fall_reward: Penalty when a walker falls (default: -10.0)
        terminate_reward: Penalty for dropping the package (default: -100.0)
        shared_reward: Whether to share rewards among all agents (default: True)
        terminate_on_fall: Whether environment terminates if any walker falls (default: True)
        max_steps: Maximum episode length (default: 500)
    """
    
    def __init__(
        self,
        n_walkers: int = 3,
        forward_reward: float = 1.0,
        fall_reward: float = -10.0,
        terminate_reward: float = -100.0,
        shared_reward: bool = True,
        terminate_on_fall: bool = True,
        max_steps: int = 500,
        backend: str = 'mjx',
        **kwargs
    ):
        # Generate XML for the environment
        xml_string = generate_multiwalker_xml(n_walkers=n_walkers)
        
        # Create MuJoCo model
        mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self._mj_model = mj_model
        
        # Convert to MJX
        if backend == 'mjx':
            sys = mjx.put_model(mj_model)
        else:
            sys = mj_model
        
        self.n_walkers = n_walkers
        self.forward_reward = forward_reward
        self.fall_reward = fall_reward
        self.terminate_reward = terminate_reward
        self.shared_reward = shared_reward
        self.terminate_on_fall = terminate_on_fall
        self.max_steps = max_steps
        
        # Get body indices
        self.walker_torso_ids = [mj_model.body(f'walker_{i}_torso').id for i in range(n_walkers)]
        self.walker_head_ids = [mj_model.body(f'walker_{i}_head').id for i in range(n_walkers)]
        self.package_id = mj_model.body('package').id
        
        # Get joint indices for each walker (4 actuators per walker: left_hip, left_knee, right_hip, right_knee)
        self.walker_joint_indices = {}
        for i in range(n_walkers):
            start_idx = i * 4
            self.walker_joint_indices[i] = jnp.array([start_idx, start_idx + 1, start_idx + 2, start_idx + 3])
        
        # Observation size: 31 per walker (like PettingZoo)
        # - 8: own joint angles and velocities (4 joints * 2)
        # - 6: own torso and head positions (3 each)
        # - 3: package position
        # - 2: package velocity
        # - 6: left neighbor info (if exists)
        # - 6: right neighbor info (if exists)
        self.obs_size = 31
        
        super().__init__(sys, backend=backend, n_frames=4)
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment to initial state."""
        
        # Initialize with small random perturbations
        q = jnp.zeros(self.sys.nq)
        qd = jnp.zeros(self.sys.nv)
        
        # Add small random noise to initial state
        rng, rng_q, rng_qd = jax.random.split(rng, 3)
        q = q + jax.random.normal(rng_q, q.shape) * 0.01
        qd = qd + jax.random.normal(rng_qd, qd.shape) * 0.01
        
        pipeline_state = self.pipeline_init(q, qd)
        
        # Get initial observations
        obs = self._get_obs(pipeline_state)
        
        # Initial rewards and metrics
        reward = jnp.zeros(self.n_walkers)
        
        metrics = {
            'package_pos': self._get_package_x(pipeline_state),
            'package_height': self._get_package_z(pipeline_state),
            'walkers_alive': jnp.ones(self.n_walkers),
            'step': 0,
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
            action: Dict mapping walker_id -> action array of shape (4,)
                    [left_hip, left_knee, right_hip, right_knee]
        """
        # Combine actions from all walkers
        ctrl = jnp.zeros(self.n_walkers * 4)
        for i in range(self.n_walkers):
            ctrl = ctrl.at[self.walker_joint_indices[i]].set(action[i])
        
        # Step physics
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        
        # Get new observations
        obs = self._get_obs(pipeline_state)
        
        # Compute rewards and check termination
        reward, done, metrics = self._compute_reward_and_done(
            state.pipeline_state,
            pipeline_state,
            state.metrics
        )
        
        # Update step counter
        step = state.metrics['step'] + 1
        metrics['step'] = step
        
        # Check max steps
        done = jnp.where(step >= self.max_steps, 1.0, done)
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics
        )
    
    def _get_obs(self, pipeline_state: Any) -> Dict[int, jax.Array]:
        """Get observations for each walker."""
        obs_dict = {}
        
        # Get package state
        package_pos = pipeline_state.x.pos[self.package_id]
        package_vel = pipeline_state.xd.vel[self.package_id]
        
        for i in range(self.n_walkers):
            # Own joint state (4 joints * 2 = 8 values)
            joint_indices = self.walker_joint_indices[i]
            # Get the corresponding q and qd indices for this walker's actuated joints
            # Each walker has 3 root joints (x, z, rot) + 4 actuated joints
            q_start = i * 7 + 3  # Skip root joints
            joint_pos = pipeline_state.q[q_start:q_start + 4]
            joint_vel = pipeline_state.qd[q_start:q_start + 4]
            
            # Own torso and head positions (6 values)
            torso_pos = pipeline_state.x.pos[self.walker_torso_ids[i]]
            head_pos = pipeline_state.x.pos[self.walker_head_ids[i]]
            
            # Package state (5 values: x, y, z position + x, z velocity)
            package_info = jnp.concatenate([
                package_pos,  # 3
                jnp.array([package_vel[0], package_vel[2]])  # 2 (x and z velocity)
            ])
            
            # Left neighbor (6 values: relative x, z, angle + velocities)
            if i > 0:
                left_torso = pipeline_state.x.pos[self.walker_torso_ids[i - 1]]
                left_vel = pipeline_state.xd.vel[self.walker_torso_ids[i - 1]]
                left_neighbor = jnp.concatenate([
                    jnp.array([left_torso[0] - torso_pos[0], left_torso[2] - torso_pos[2]]),
                    jnp.array([0.0]),  # angle difference (simplified)
                    jnp.array([left_vel[0], left_vel[2], 0.0])
                ])
            else:
                left_neighbor = jnp.zeros(6)
            
            # Right neighbor (6 values)
            if i < self.n_walkers - 1:
                right_torso = pipeline_state.x.pos[self.walker_torso_ids[i + 1]]
                right_vel = pipeline_state.xd.vel[self.walker_torso_ids[i + 1]]
                right_neighbor = jnp.concatenate([
                    jnp.array([right_torso[0] - torso_pos[0], right_torso[2] - torso_pos[2]]),
                    jnp.array([0.0]),  # angle difference (simplified)
                    jnp.array([right_vel[0], right_vel[2], 0.0])
                ])
            else:
                right_neighbor = jnp.zeros(6)
            
            # Combine observation (total: 8 + 6 + 5 + 6 + 6 = 31)
            obs = jnp.concatenate([
                joint_pos,       # 4
                joint_vel,       # 4
                torso_pos,       # 3
                head_pos,        # 3
                package_info,    # 5
                left_neighbor,   # 6
                right_neighbor,  # 6
            ])
            
            obs_dict[i] = obs
        
        return obs_dict
    
    def _get_package_x(self, pipeline_state: Any) -> jax.Array:
        """Get package x position."""
        return pipeline_state.x.pos[self.package_id, 0]
    
    def _get_package_z(self, pipeline_state: Any) -> jax.Array:
        """Get package z (height) position."""
        return pipeline_state.x.pos[self.package_id, 2]
    
    def _check_walker_fallen(self, pipeline_state: Any, walker_id: int) -> jax.Array:
        """Check if a walker has fallen."""
        torso_z = pipeline_state.x.pos[self.walker_torso_ids[walker_id], 2]
        # Walker is considered fallen if torso is below 0.5m
        return torso_z < 0.5
    
    def _compute_reward_and_done(
        self,
        prev_state: Any,
        curr_state: Any,
        prev_metrics: Dict[str, Any]
    ) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
        """Compute rewards and check for termination."""
        
        # Forward progress reward
        prev_package_x = self._get_package_x(prev_state)
        curr_package_x = self._get_package_x(curr_state)
        forward_progress = (curr_package_x - prev_package_x) * self.forward_reward
        
        # Check package height (fallen if below 1.0m)
        package_z = self._get_package_z(curr_state)
        package_fallen = package_z < 1.0
        
        # Check which walkers have fallen
        walkers_fallen = jnp.array([
            self._check_walker_fallen(curr_state, i) 
            for i in range(self.n_walkers)
        ])
        
        # Individual rewards
        rewards = jnp.ones(self.n_walkers) * forward_progress
        
        # Apply fall penalties
        rewards = jnp.where(
            walkers_fallen,
            rewards + self.fall_reward,
            rewards
        )
        
        # Check termination conditions
        done = 0.0
        
        # Package fallen
        if package_fallen:
            rewards = jnp.ones(self.n_walkers) * self.terminate_reward
            done = 1.0
        
        # Walker(s) fallen with terminate_on_fall
        if self.terminate_on_fall and jnp.any(walkers_fallen):
            rewards = jnp.ones(self.n_walkers) * self.terminate_reward
            done = 1.0
        
        # Shared reward mode
        if self.shared_reward:
            mean_reward = jnp.mean(rewards)
            rewards = jnp.ones(self.n_walkers) * mean_reward
        
        # Update metrics
        metrics = {
            'package_pos': curr_package_x,
            'package_height': package_z,
            'walkers_alive': 1.0 - walkers_fallen.astype(jnp.float32),
            'forward_progress': forward_progress,
            'package_fallen': package_fallen.astype(jnp.float32),
        }
        
        return rewards, done, metrics


def plot_multiwalker_metrics(states, save_path="multiwalker_metrics.png"):
    """Plot MultiWalker training metrics."""
    
    # Extract metrics
    package_positions = [float(s.metrics['package_pos']) for s in states]
    package_heights = [float(s.metrics['package_height']) for s in states]
    rewards = [float(s.reward[0]) for s in states]  # Shared reward
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    timesteps = range(len(states))
    
    # Plot 1: Package position
    ax1 = axes[0]
    ax1.plot(timesteps, package_positions, 'b-', linewidth=2)
    ax1.set_ylabel('Package X Position (m)', fontsize=11)
    ax1.set_title('Package Forward Progress', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Package height
    ax2 = axes[1]
    ax2.plot(timesteps, package_heights, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Fallen Threshold', linewidth=2)
    ax2.set_ylabel('Package Height (m)', fontsize=11)
    ax2.set_title('Package Height (Stability)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Rewards
    ax3 = axes[2]
    ax3.plot(timesteps, rewards, 'purple', linewidth=2, alpha=0.7)
    ax3.set_ylabel('Reward', fontsize=11)
    ax3.set_xlabel('Timestep', fontsize=11)
    ax3.set_title('Team Reward', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {save_path}")
    plt.close()


def run_multiwalker_demo(
    n_walkers=3,
    num_steps=500,
    policy=None
):
    """Run MultiWalker environment demo."""
    
    print("=" * 70)
    print(f"MultiWalker Environment Demo ({n_walkers} walkers)")
    print("=" * 70)
    
    # Create environment
    print(f"\nCreating environment with {n_walkers} walkers...")
    env = MultiWalker(
        n_walkers=n_walkers,
        forward_reward=1.0,
        fall_reward=-10.0,
        terminate_reward=-100.0,
        shared_reward=True,
        terminate_on_fall=True,
        max_steps=num_steps,
        backend='mjx'
    )
    
    print(f"* {n_walkers} bipedal walkers")
    print(f"* Action space: 4D per walker (hip and knee torques)")
    print(f"* Observation space: 31D per walker")
    print(f"* Task: Cooperatively carry package forward")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    state = jax.jit(env.reset)(rng)
    states = [state]
    
    print(f"\nInitial state:")
    print(f"  Package position: {state.metrics['package_pos']:.3f}m")
    print(f"  Package height: {state.metrics['package_height']:.3f}m")
    print(f"  Observation shape: {state.obs[0].shape}")
    
    # JIT compile
    step_fn = jax.jit(env.step)
    
    # Run rollout
    print(f"\nRunning {num_steps}-step rollout...")
    
    for i in range(num_steps):
        rng = jax.random.split(rng)[0]
        
        if policy is None:
            # Random actions (will likely fail)
            actions = {}
            for walker_id in range(n_walkers):
                rng, rng_act = jax.random.split(rng)
                actions[walker_id] = jax.random.uniform(rng_act, (4,), minval=-1.0, maxval=1.0)
        else:
            # Use provided policy
            actions = {}
            for walker_id in range(n_walkers):
                rng, rng_act = jax.random.split(rng)
                actions[walker_id] = policy(state.obs[walker_id], rng_act)
        
        state = step_fn(state, actions)
        states.append(state)
        
        # Check if done
        if state.done:
            print(f"\n  Episode ended at step {i + 1}")
            if state.metrics.get('package_fallen', 0.0):
                print("  Reason: Package fell")
            else:
                print("  Reason: Walker(s) fell")
            break
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}: "
                  f"Package X = {state.metrics['package_pos']:.3f}m, "
                  f"Height = {state.metrics['package_height']:.3f}m, "
                  f"Reward = {state.reward[0]:.2f}")
    
    # Summary
    final_pos = states[-1].metrics['package_pos']
    initial_pos = states[0].metrics['package_pos']
    distance_traveled = final_pos - initial_pos
    
    print(f"\n{'=' * 70}")
    print("Episode Summary:")
    print(f"{'=' * 70}")
    print(f"  Steps completed: {len(states) - 1}")
    print(f"  Initial package position: {initial_pos:.3f}m")
    print(f"  Final package position: {final_pos:.3f}m")
    print(f"  Distance traveled: {distance_traveled:.3f}m")
    print(f"  Final package height: {states[-1].metrics['package_height']:.3f}m")
    
    # Generate plots
    plot_multiwalker_metrics(states)
    
    return states


# Simple coordinated policy (heuristic)
def coordinated_policy(obs, rng):
    """
    A simple coordinated walking policy.
    Uses sinusoidal patterns for leg movements.
    """
    # Extract phase from observation (use step count or estimate from velocity)
    phase = jax.random.uniform(rng, minval=0.0, maxval=2 * jnp.pi)
    
    # Coordinated walking pattern
    left_hip = jnp.sin(phase) * 0.5
    left_knee = jnp.maximum(jnp.sin(phase + jnp.pi/4), 0.0) * 0.8
    right_hip = jnp.sin(phase + jnp.pi) * 0.5
    right_knee = jnp.maximum(jnp.sin(phase + jnp.pi + jnp.pi/4), 0.0) * 0.8
    
    action = jnp.array([left_hip, left_knee, right_hip, right_knee])
    action = jnp.clip(action, -1.0, 1.0)
    
    return action


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Brax MultiWalker Environment")
    print("Based on PettingZoo SISL MultiWalker")
    print("=" * 70 + "\n")
    
    # Demo with 3 walkers (default)
    print("Demo: Random Actions")
    states = run_multiwalker_demo(
        n_walkers=3,
        num_steps=500,
        policy=None  # Random
    )
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - multiwalker_metrics.png")
    print("\nNote: Random actions will likely cause the walkers to fall.")
    print("Train with RL algorithms (PPO, IPPO, MAPPO) for coordinated behavior!")