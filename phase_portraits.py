import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

# Check and create a directory for plots
plot_dir = 'phase_portraits'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters
A_values = [0.90, 1.07, 1.35, 1.47, 1.5]
k = 0.5
phi = 0.66667
transient = 5000  # Number of iterations to skip as transient
nsteps = 10000  # Total number of steps
dt = 0.01  # Time step

# Pendulum's equation of motion
def f_damped_driven(theta, omega, t, A):
    return -np.sin(theta) - k * omega + A * np.cos(phi * t)

for A in A_values:
    theta = 0.2
    omega = 0.0
    t = 0.0
    iteration_number = 0

    theta_values = []
    omega_values = []
    last_theta = theta

    for step in range(nsteps):
        iteration_number += 1

        # Trapezoid rule
        k1a = dt * omega
        k1b = dt * f_damped_driven(theta, omega, t, A)
        k2a = dt * (omega + k1b)
        k2b = dt * f_damped_driven(theta + k1a, omega + k1b, t + dt, A)

        theta += (k1a + k2a) / 2
        omega += (k1b + k2b) / 2
        t += dt

        # Adjust theta to stop jumps in the phase portrait
        while np.abs(theta - last_theta) > np.pi:
            theta -= 2 * np.pi * np.sign(theta - last_theta)
        
        last_theta = theta  # Update last_theta for the next iteration

        if iteration_number > transient:
            theta_values.append(theta)
            omega_values.append(omega)

    # Plot the phase portrait
    plt.figure(figsize=(4, 4))
    plt.scatter(theta_values, omega_values, s=2, c='b')
    plt.xlabel('Theta')
    plt.ylabel('Omega')
    plt.title(f'Phase Portrait for A = {A}')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/PhasePortrait_A_{A}.pdf")
    plt.savefig(f"{plot_dir}/PhasePortrait_A_{A}.png")
    plt.show()
