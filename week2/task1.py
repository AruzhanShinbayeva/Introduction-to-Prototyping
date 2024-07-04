import mujoco
import numpy as np
import csv
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model_path = 'W1_task1.xml'
model = mujoco.MjModel.from_xml_path(filename=model_path)
data = mujoco.MjData(model)

# Get Joints
joints = []
joint_limits = []
num_samples = 10
for i in range(model.njnt):
    joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    joint_limits.append(model.jnt_range[i])

# Simulate a step to initialize
mujoco.mj_step(model, data)
configs = itertools.product(*[np.linspace(limit[0], limit[1], num=num_samples) for limit in joint_limits])

# Collect results
experiment_data = []
for config in configs:
    data.qpos[:] = config
    mujoco.mj_forward(model, data)  # Update simulation
    mujoco.mj_inverse(model, data)  # Compute inverse dynamics
    torque_values = data.qfrc_inverse
    lst = list(config)
    lst.extend(list(torque_values))
    experiment_data.append(lst)

# Save results
csv_file = 'outcomes.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = joints + [f'torque_{joint}' for joint in joints]
    writer.writerow(header)
    writer.writerows(experiment_data)

data = np.genfromtxt(csv_file, delimiter=',', names=True)

torque_data = {joint: data[f'torque_{joint}'] for joint in joints}

# Show results
plt.figure(figsize=(12, 8))
sns.violinplot(data=list(torque_data.values()))
plt.xticks(range(len(joints)), joints)
plt.xlabel('Joint')
plt.ylabel('Torque')
plt.title('Torque Distribution Across Joints')
plt.show()
