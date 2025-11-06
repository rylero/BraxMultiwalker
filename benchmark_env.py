"""
Benchmark script comparing Brax physics backends: MJX, Spring, Positional, Generalized
Tests on the DoubleArmTouch multi-agent environment
"""

import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

# Import the environment from previous code
# Assuming it's saved as double_arm_env.py
from double_arm_env import DoubleArmTouch, DOUBLE_ARM_SYSTEM


class BackendBenchmark:
    """Benchmark different Brax physics backends."""
    
    def __init__(self, num_envs_list=[1, 10, 100, 1000, 2048], num_steps=1000):
        self.num_envs_list = num_envs_list
        self.num_steps = num_steps
        self.backends = ['mjx', 'spring', 'positional', 'generalized']
        self.results = []
        
    def create_env(self, backend: str):
        """Create environment with specified backend."""
        try:
            env = DoubleArmTouch(backend=backend)
            return env
        except Exception as e:
            print(f"Error creating {backend} environment: {e}")
            return None
    
    def benchmark_single_env(
        self, 
        env, 
        backend: str, 
        num_envs: int = 1
    ) -> Tuple[float, float]:
        """
        Benchmark a single backend configuration.
        
        Returns:
            (steps_per_second, compilation_time)
        """
        print(f"\nBenchmarking {backend} with {num_envs} parallel environments...")
        
        # Create batched reset and step functions
        def batched_reset(rng):
            rngs = jax.random.split(rng, num_envs)
            return jax.vmap(env.reset)(rngs)
        
        def batched_step(states, actions):
            return jax.vmap(env.step)(states, actions)
        
        # JIT compile
        print(f"  Compiling...")
        start_compile = time.time()
        
        jit_reset = jax.jit(batched_reset)
        jit_step = jax.jit(batched_step)
        
        # Warm-up compilation with actual execution
        rng = jax.random.PRNGKey(0)
        states = jit_reset(rng)
        
        # Create random actions
        action_rng = jax.random.split(rng, num_envs * 2)
        actions = {
            0: jax.random.uniform(action_rng[:num_envs], (num_envs, 2), minval=-1.0, maxval=1.0),
            1: jax.random.uniform(action_rng[num_envs:], (num_envs, 2), minval=-1.0, maxval=1.0),
        }
        
        states = jit_step(states, actions)
        states.obs[0].block_until_ready()  # Wait for compilation
        
        compile_time = time.time() - start_compile
        print(f"  Compilation time: {compile_time:.3f}s")
        
        # Benchmark execution
        print(f"  Running {self.num_steps} steps...")
        start_time = time.time()
        
        for step in range(self.num_steps):
            # Generate random actions
            step_rng = jax.random.PRNGKey(step)
            action_rng = jax.random.split(step_rng, num_envs * 2)
            actions = {
                0: jax.random.uniform(action_rng[:num_envs], (num_envs, 2), minval=-1.0, maxval=1.0),
                1: jax.random.uniform(action_rng[num_envs:], (num_envs, 2), minval=-1.0, maxval=1.0),
            }
            
            states = jit_step(states, actions)
        
        # Wait for all computations to finish
        states.obs[0].block_until_ready()
        
        elapsed_time = time.time() - start_time
        total_steps = self.num_steps * num_envs
        steps_per_second = total_steps / elapsed_time
        
        print(f"  Time: {elapsed_time:.3f}s")
        print(f"  Steps/sec: {steps_per_second:,.0f}")
        
        return steps_per_second, compile_time
    
    def run_benchmark(self):
        """Run full benchmark across all backends and environment counts."""
        print("=" * 70)
        print("Brax Physics Backend Benchmark")
        print("=" * 70)
        
        for backend in self.backends:
            print(f"\n{'=' * 70}")
            print(f"Testing backend: {backend.upper()}")
            print(f"{'=' * 70}")
            
            env = self.create_env(backend)
            if env is None:
                print(f"Skipping {backend} due to initialization error")
                continue
            
            for num_envs in self.num_envs_list:
                try:
                    steps_per_sec, compile_time = self.benchmark_single_env(
                        env, backend, num_envs
                    )
                    
                    self.results.append({
                        'backend': backend,
                        'num_envs': num_envs,
                        'steps_per_second': steps_per_sec,
                        'compile_time': compile_time,
                        'time_per_step_ms': 1000.0 / (steps_per_sec / num_envs)
                    })
                    
                except Exception as e:
                    print(f"Error benchmarking {backend} with {num_envs} envs: {e}")
                    continue
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display benchmark results."""
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        # Overall statistics by backend
        print("\nAverage Steps/Second by Backend (across all environment counts):")
        backend_avg = df.groupby('backend')['steps_per_second'].mean().sort_values(ascending=False)
        for backend, speed in backend_avg.items():
            print(f"  {backend:12s}: {speed:>12,.0f} steps/sec")
        
        print("\nAverage Compilation Time by Backend:")
        compile_avg = df.groupby('backend')['compile_time'].mean().sort_values()
        for backend, ctime in compile_avg.items():
            print(f"  {backend:12s}: {ctime:>8.3f}s")
        
        # Best performance for each environment count
        print("\nBest Backend by Number of Parallel Environments:")
        for num_envs in self.num_envs_list:
            subset = df[df['num_envs'] == num_envs]
            if len(subset) > 0:
                best = subset.loc[subset['steps_per_second'].idxmax()]
                print(f"  {num_envs:4d} envs: {best['backend']:12s} "
                      f"({best['steps_per_second']:>12,.0f} steps/sec)")
        
        # Detailed table
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
        
        return df
    
    def plot_results(self, df: pd.DataFrame, save_path='backend_benchmark.png'):
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Steps/sec vs Number of Environments
        ax1 = axes[0, 0]
        for backend in self.backends:
            backend_data = df[df['backend'] == backend]
            if len(backend_data) > 0:
                ax1.plot(
                    backend_data['num_envs'], 
                    backend_data['steps_per_second'],
                    marker='o', 
                    label=backend.upper(),
                    linewidth=2
                )
        
        ax1.set_xlabel('Number of Parallel Environments', fontsize=11)
        ax1.set_ylabel('Steps per Second', fontsize=11)
        ax1.set_title('Throughput vs Parallelization', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Time per step (ms)
        ax2 = axes[0, 1]
        for backend in self.backends:
            backend_data = df[df['backend'] == backend]
            if len(backend_data) > 0:
                ax2.plot(
                    backend_data['num_envs'],
                    backend_data['time_per_step_ms'],
                    marker='s',
                    label=backend.upper(),
                    linewidth=2
                )
        
        ax2.set_xlabel('Number of Parallel Environments', fontsize=11)
        ax2.set_ylabel('Time per Step (ms)', fontsize=11)
        ax2.set_title('Latency per Environment Step', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Plot 3: Speedup relative to single environment
        ax3 = axes[1, 0]
        for backend in self.backends:
            backend_data = df[df['backend'] == backend].copy()
            if len(backend_data) > 0 and backend_data['num_envs'].min() == 1:
                baseline = backend_data[backend_data['num_envs'] == 1]['steps_per_second'].values[0]
                backend_data['speedup'] = backend_data['steps_per_second'] / baseline
                
                ax3.plot(
                    backend_data['num_envs'],
                    backend_data['speedup'],
                    marker='^',
                    label=backend.upper(),
                    linewidth=2
                )
        
        # Ideal linear speedup reference
        ideal_x = [1, max(self.num_envs_list)]
        ideal_y = ideal_x
        ax3.plot(ideal_x, ideal_y, 'k--', alpha=0.5, label='Ideal Linear', linewidth=1)
        
        ax3.set_xlabel('Number of Parallel Environments', fontsize=11)
        ax3.set_ylabel('Speedup Factor', fontsize=11)
        ax3.set_title('Parallel Scaling Efficiency', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Plot 4: Compilation time comparison
        ax4 = axes[1, 1]
        compile_data = df.groupby('backend')['compile_time'].mean().sort_values()
        bars = ax4.barh(
            range(len(compile_data)), 
            compile_data.values,
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(compile_data)]
        )
        ax4.set_yticks(range(len(compile_data)))
        ax4.set_yticklabels([b.upper() for b in compile_data.index])
        ax4.set_xlabel('Compilation Time (seconds)', fontsize=11)
        ax4.set_title('Average JIT Compilation Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, compile_data.values)):
            ax4.text(value, i, f' {value:.3f}s', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.show()


def run_quick_comparison():
    """Run a quick comparison with fewer environment counts."""
    print("Running QUICK comparison (fewer environment counts)...\n")
    benchmark = BackendBenchmark(
        num_envs_list=[1, 100, 1000],
        num_steps=500
    )
    results = benchmark.run_benchmark()
    df = benchmark.analyze_results()
    
    if df is not None and len(df) > 0:
        benchmark.plot_results(df, save_path='quick_backend_benchmark.png')
    
    return df


def run_full_comparison():
    """Run comprehensive comparison with all environment counts."""
    print("Running FULL comparison (all environment counts)...\n")
    benchmark = BackendBenchmark(
        num_envs_list=[1, 10, 100, 1000, 2048],
        num_steps=1000
    )
    results = benchmark.run_benchmark()
    df = benchmark.analyze_results()
    
    if df is not None and len(df) > 0:
        benchmark.plot_results(df, save_path='full_backend_benchmark.png')
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Brax physics backends')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full'],
        default='quick',
        help='Benchmark mode: quick (faster) or full (comprehensive)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 70}")
    print(f"Brax Backend Benchmark - {args.mode.upper()} mode")
    print(f"{'=' * 70}\n")
    
    # Check JAX/GPU availability
    print("System Information:")
    print(f"  JAX version: {jax.__version__}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Backend: {jax.default_backend()}")
    
    # Run appropriate benchmark
    if args.mode == 'quick':
        df = run_quick_comparison()
    else:
        df = run_full_comparison()
    
    # Save results to CSV
    if df is not None and len(df) > 0:
        csv_path = f'{args.mode}_benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
