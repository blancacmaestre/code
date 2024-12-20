import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate example data for input and recovered values
np.random.seed(42)
parameters = [f"Param {i+1}" for i in range(10)]
input_values = np.random.uniform(0, 10, size=10)
recovered_values = input_values + np.random.normal(0, 1, size=10)  # Slightly noisy recovery

# Set up the color palette
palette = sns.color_palette("husl", 10)  # Husl palette for vibrant, cute colors

# Create the plot
plt.figure(figsize=(10, 6))
for i, (param, input_val, recovered_val) in enumerate(zip(parameters, input_values, recovered_values)):
    plt.plot([i, i], [input_val, recovered_val], color=palette[i], linestyle='--', marker='o', label=f"{param} (Recovered)")
    plt.scatter(i, input_val, color=palette[i], label=f"{param} (Input)")

# Customizing the plot
plt.xticks(range(len(parameters)), parameters, rotation=45, ha="right")
plt.ylabel("Value")
plt.title("Input vs Recovered Values for Parameters")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Legend")
plt.tight_layout()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Example data: 10 parameters, constant within specific radial ranges
radii = np.linspace(0, 10, 100)  # Radii from 0 to 10, with 100 points
parameters = [f"Param {i+1}" for i in range(10)]

# Define input and recovered values as constant within ranges
np.random.seed(42)
input_values = np.random.uniform(5, 15, size=10)  # Input values for each parameter
recovered_values = input_values + np.random.normal(0, 1, size=10)  # Slightly noisy recovery

# Generate step-like data for each parameter
step_radii = [radii[i * 10:(i + 1) * 10] for i in range(10)]  # Divide radius range into 10 chunks

# Set up the color palette using a matplotlib colormap
colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use tab10 for distinct colors

# Create the plot
plt.figure(figsize=(10, 6))
for i, (param, input_val, recovered_val, step_radius) in enumerate(zip(parameters, input_values, recovered_values, step_radii)):
    plt.plot(step_radius, [input_val] * len(step_radius), label=f"{param} (Input)", color=colors[i], linestyle='-')
    plt.plot(step_radius, [recovered_val] * len(step_radius), label=f"{param} (Recovered)", color=colors[i], linestyle='--')

# Customizing the plot
plt.xlabel("Radius")
plt.ylabel("Value")
plt.title("Input vs Recovered Values for Parameters Across Radius")
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Legend")
plt.tight_layout()

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Example data setup
radii = np.linspace(0, 100, 100)  # Radial distances
parameters = [f"Param {i+1}" for i in range(5)]  # Five parameters for demonstration

# Generate mock values
np.random.seed(42)
input_values = np.random.uniform(50, 70, size=5)  # Input values for each parameter
recovered_values = input_values + np.random.normal(0, 2, size=5)  # Recovered values with some noise
errors_low = np.random.uniform(1, 3, size=5)  # Lower uncertainty
errors_high = np.random.uniform(1, 3, size=5)  # Upper uncertainty

# Initialize subplots
fig, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

# Plot each parameter
for i, param in enumerate(parameters):
    # Plot input value as a constant line
    ax[i].plot(radii, [input_values[i]] * len(radii), label=f"{param} (Input)", linestyle="-", color="blue")
    
    # Plot recovered value as a dashed line
    ax[i].plot(radii, [recovered_values[i]] * len(radii), label=f"{param} (Recovered)", linestyle="--", color="orange")
    
    # Add uncertainty as a shaded region
    ax[i].fill_between(
        radii,
        [recovered_values[i] - errors_low[i]] * len(radii),
        [recovered_values[i] + errors_high[i]] * len(radii),
        color="orange",
        alpha=0.2,
        label=f"{param} (Uncertainty)"
    )
    
    # Customize subplot
    ax[i].set_ylabel(f"{param} Value")
    ax[i].set_ylim(
        recovered_values[i] - errors_low[i] - 5,
        recovered_values[i] + errors_high[i] + 5
    )
    ax[i].legend(loc="upper right")
    ax[i].grid()

# Add common labels and adjust layout
ax[-1].set_xlabel("Radius")
fig.suptitle("Parameter Input vs Recovered Values Across Radius", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()

