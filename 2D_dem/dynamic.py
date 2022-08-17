import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
from force import get_force_list,get_force_of_wall,get_v_list,get_v_of_wall
from sdf import phy_seed
from jax import random
import numpy as np
import time
i32 = jnp.int32
f32 = jnp.float64
Array = jnp.ndarray
#from matplotlib import pyplot as plt
T_max:int=10
#所运行的时间
Delta_t:float=0.0001
#最小时间间隔
key = random.PRNGKey(374954787)
boundary_space = jnp.asarray([10,10],dtype = f32)
# V_list=random.uniform(key,(100,3),dtype=f32,minval=-0.5, maxval=0.5)

# R_list = np.zeros((100,3),dtype = f32)
# for i in range(100):
#     x = jnp.floor(i/10) 
#     y = i%10
#     R_list[i][0] = x+0.5
#     R_list[i][1] = y+0.5

#以上为对于初始位置和初始速度的设定，100个粒子
V_list = jnp.asarray([[0.5,0,0],[-0.5,0,0]],dtype = f32)
R_list = jnp.asarray([[5,5,0],[6,5,0]],dtype = f32)

#以上为对于初始位置和初始速度的设定，2个粒子

def get_force(p_batch):
	force_list =get_force_list(p_batch)+get_force_of_wall(p_batch)
	return force_list

#以上为力的计算，分别计算了墙的作用力和粒子相互作用力
@jit
def leap_frog(R_list,V_list,Delta_t:float=Delta_t):
	t_V_list=(get_force(R_list))*Delta_t*0.5+V_list
	r_R_list=R_list+t_V_list*Delta_t
	r_V_list=t_V_list+(get_force(r_R_list))*Delta_t*0.5
	return r_R_list,r_V_list
#跳蛙积分法
def dynamic(R_list,V_list,Delta_t:float=Delta_t,T_max:int=T_max):
	N_time=int(T_max/Delta_t)
	r_list = []
	v_list = []
	for i in range(N_time):
	    R_list,V_list=leap_frog(R_list,V_list)
	    t_r = R_list.reshape(-1)
	    t_v = V_list.reshape(-1)
	    r_list.append(t_r)
	    v_list.append(t_v)
	    if (i%1000 == 0):
		    print(i/1000)

	return r_list,v_list
#做若干个循环
x=jnp.arange(int(T_max/Delta_t),dtype=float)
start=time.perf_counter()
y,z=dynamic(R_list,V_list)
end=time.perf_counter()
t=end-start
print(t)
with open('test_y.npy', 'wb') as f:
    jnp.save(f,y)

f.close()
with open('test_z.npy','wb') as f:
    jnp.save(f,z)
    
f.close()
    
#记录运动过程中的每时每刻的粒子们的速度跟位置