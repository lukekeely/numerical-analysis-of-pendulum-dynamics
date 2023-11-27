import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

# Check and create a directory for plots
plot_dir = 'damped_nonlinear_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Pendulum's equation of motion
def pendulum_motion(theta, omega, t, k=0.5, phi=0.66667, A=0.00):
    return -np.sin(theta) - k * omega + A * np.cos(phi * t)

# Function to solve and plot the pendulum motion using the trapezoidal rule
def solve_and_plot_pendulum(nsteps, initial_conditions, motion_func, plot_dir):
    theta, omega, t, dt = initial_conditions
    thetas, omegas, times = np.zeros(nsteps), np.zeros(nsteps), np.zeros(nsteps)

    for step in range(nsteps):
        thetas[step], omegas[step] = theta, omega
        times[step] = t

        # Trapezoidal rule
        k1_theta, k1_omega = dt * omega, dt * motion_func(theta, omega, t)
        k2_theta = dt * (omega + k1_omega)
        k2_omega = dt * motion_func(theta + k1_theta, omega + k1_omega, t + dt)
        
        theta += (k1_theta + k2_theta) / 2
        omega += (k1_omega + k2_omega) / 2
        t += dt

    # Plotting 
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, thetas, label='Theta (Angle)')
    plt.xlabel('Time')
    plt.ylabel('Theta (Angle)')
    plt.title('Theta vs Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(times, omegas, label='Omega (Angular Velocity)', color='r')
    plt.xlabel('Time')
    plt.ylabel('Omega (Angular Velocity)')
    plt.title('Omega vs Time')
    plt.legend()
    
    plt.suptitle(f"Damped Nonlinear Pendulum Simulation (Initial Theta: {initial_conditions[0]})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{plot_dir}/pendulum_{initial_conditions[0]}.pdf")
    plt.savefig(f"{plot_dir}/pendulum_{initial_conditions[0]}.png")
    plt.show()

# Initial conditions
initial_conditions_list = [
    #theta, omega, t, dt
    (3.0, 0.0, 0.0, 0.01),
]

for initial_conditions in initial_conditions_list:
    solve_and_plot_pendulum(5000, initial_conditions, pendulum_motion, plot_dir)
