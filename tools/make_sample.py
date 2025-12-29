import numpy as np

# -------------------------
# Parameters
# -------------------------
np.random.seed(0)

# ground plane
num_ground = 5000
xg = np.random.uniform(-30, 30, num_ground)
yg = np.random.uniform(-30, 30, num_ground)
zg = np.random.normal(loc=0.0, scale=0.02, size=num_ground)  # flat ground

ground = np.stack([xg, yg, zg], axis=1)

# obstacles
num_obj = 800
xo = np.random.uniform(-10, 10, num_obj)
yo = np.random.uniform(-10, 10, num_obj)
zo = np.random.uniform(0.5, 2.0, num_obj)  # above ground

objects = np.stack([xo, yo, zo], axis=1)

points = np.concatenate([ground, objects], axis=0)

np.save("sample.npy", points)
print("saved sample.npy:", points.shape)
