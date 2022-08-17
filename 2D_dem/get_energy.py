import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
from force import get_force_list,get_force_of_wall,get_v_list,get_v_of_wall
from sdf import phy_seed
from jax import random
import numpy as np
import time

with open('test_z.npy', 'rb') as f:
    z = jnp.load(f)
T = jnp.power(z,2)
T = jnp.sum(T,axis = 1).reshape(-1)
with open('T.npy', 'wb') as f:
    jnp.save(f,T)
with open('test_y.npy', 'rb') as f:
    y = jnp.load(f)

v_list = []
for i in range(100000):
    if (i%1000 ==0):
        print(i)

    y_t = y[i].reshape(2,-1)
    v = get_v_list(y_t)
    v_list.append(v)

with open('V.npy', 'wb') as f:
    jnp.save(f,v_list)