import jax.numpy as jnp
import numpy as np
i32 = jnp.int32
f32 = jnp.float32
from force import get_force_list,get_force_of_wall,get_v_list,get_v_of_wall
with open('test_y.npy', 'rb') as f:
    y = np.load(f)
with open('test_z.npy', 'rb') as f:
    z = np.load(f)

y = y.reshape(1000000,3)
T = (y*y).reshape(10000,300)
T = np.sum(T,axis = 1)

z = z.reshape(10000,100,3)
V = np.zeros((10000),dtype = f32)
for i in range(10000):
    z_t = z[i].reshape(100,3)
    V[i]= get_v_list(z_t)
V_wall = get_v_of_wall(z.reshape(1000000,3)) 
V_wall = np.sum(V_wall.reshape(10000,100),axis = 1)
with open('T_list.npy', 'wb') as f:
    jnp.save(f,T)

f.close()
with open('V_list.npy','wb') as f:
    jnp.save(f,V)
    
f.close()
with open('V_w_list.npy','wb') as f:
    jnp.save(f,V_wall)
    
f.close()