import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

# Parameters for pendulum
k = 0.0  # Damping
phi = 0.66667  # Driving force frequency
A = 0.0  # Driving force amplitude

# Pendulum's equations of motion
def f_linear(theta, omega, t):
    return -theta - k * omega + A * np.cos(phi * t)

def f_nonlinear(theta, omega, t):
    return -np.sin(theta) - k * omega + A * np.cos(phi * t)

# Initialize variables
initial_theta_values = [np.pi / 10, np.pi / 2, np.pi / 1.5, np.pi - 0.0001]
dt = 0.01
nsteps = 10000

plot_dir = 'method_comparison_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Function for simulation
def simulate_pendulum(f_motion, initial_theta, plot_dir, ax):
    time_vals = np.zeros(nsteps)
    theta_vals = {'euler': np.zeros(nsteps), 'trap': np.zeros(nsteps), 'rk4': np.zeros(nsteps)}
    omega_vals = {'euler': 0.0, 'trap': 0.0, 'rk4': 0.0}

    theta_vals['euler'][0] = theta_vals['trap'][0] = theta_vals['rk4'][0] = initial_theta
    time_vals[0] = 0.0

    # Simulation loop
    for i in range(1, nsteps):
        t = i * dt
        time_vals[i] = t

        # Euler's Method
        theta_vals['euler'][i] = theta_vals['euler'][i-1] + dt * omega_vals['euler']
        omega_vals['euler'] += dt * f_motion(theta_vals['euler'][i-1], omega_vals['euler'], t)
        
        # Trapezoidal Rule
        k1_theta, k1_omega = dt * omega_vals['trap'], dt * f_motion(theta_vals['trap'][i-1], omega_vals['trap'], t)
        k2_theta = dt * (omega_vals['trap'] + k1_omega)
        k2_omega = dt * f_motion(theta_vals['trap'][i-1] + k1_theta, omega_vals['trap'] + k1_omega, t + dt)
        theta_vals['trap'][i] = theta_vals['trap'][i-1] + (k1_theta + k2_theta) / 2
        omega_vals['trap'] += (k1_omega + k2_omega) / 2

        # Fourth-Order Runge-Kutta Method
        k1 = dt * omega_vals['rk4']
        l1 = dt * f_motion(theta_vals['rk4'][i-1], omega_vals['rk4'], t)
        k2 = dt * (omega_vals['rk4'] + l1 / 2)
        l2 = dt * f_motion(theta_vals['rk4'][i-1] + k1 / 2, omega_vals['rk4'] + l1 / 2, t + dt / 2)
        k3 = dt * (omega_vals['rk4'] + l2 / 2)
        l3 = dt * f_motion(theta_vals['rk4'][i-1] + k2 / 2, omega_vals['rk4'] + l2 / 2, t + dt / 2)
        k4 = dt * (omega_vals['rk4'] + l3)
        l4 = dt * f_motion(theta_vals['rk4'][i-1] + k3, omega_vals['rk4'] + l3, t + dt)
        theta_vals['rk4'][i] = theta_vals['rk4'][i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        omega_vals['rk4'] += (l1 + 2 * l2 + 2 * l3 + l4) / 6

    # Plotting
    for method in theta_vals:
        ax.plot(time_vals, theta_vals[method], label=method.capitalize(), alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Theta [rad]')
    ax.legend()
    ax.set_title(f'{plot_dir.capitalize()} Pendulum (Initial Theta: {initial_theta})')


for initial_theta in initial_theta_values:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Simulate and plot linear pendulum
    simulate_pendulum(f_linear, initial_theta, 'Linear', axs[0])
    axs[0].set_title('Linear Pendulum')

    # Simulate and plot nonlinear pendulum
    simulate_pendulum(f_nonlinear, initial_theta, 'Nonlinear', axs[1])
    axs[1].set_title('Nonlinear Pendulum')

    fig.suptitle(f'Pendulum Simulations (Initial Theta: {initial_theta:.5f})')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{plot_dir}/{initial_theta}_comparison_plot.pdf")
    plt.savefig(f"{plot_dir}/{initial_theta}_comparison_plot.png")
    plt.show()
