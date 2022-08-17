import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
import math
Pi = math.pi
i32 = jnp.int32
f32 = jnp.float32
def sdf(r_batched,a=0.7):
	result=jnp.max(jnp.abs(r_batched),axis=0)-0.5*a
	return result
dirction=grad(sdf)
@jit
def vmap_sdf(r_batched):
	return vmap(sdf)(r_batched)
@jit
def vmap_dirction(r_batched):
	return vmap(dirction)(r_batched)

a=jnp.arange(-160,160,1,dtype=f32)+0.5
a = a*0.7*(1/320)
b=jnp.ones(320,dtype=f32)*0.35
ax_list=jnp.stack((a,b,-1*a,-1*b)).reshape(1280,1)
ay_list=jnp.stack((b,-1*a,-1*b,a)).reshape(1280,1)
a_list=jnp.append(ax_list,ay_list,axis=1).reshape(1280,2)
phy_seed = a_list
#正方形的SDF以及配套的导数值

# def sdf(r_batched,a = 0.5):
#     t = r_batched*r_batched
#     t_sum =jnp.sum(t).reshape(1)
#     q = jnp.sqrt(t_sum)-a
#     return q[0]

# a = jnp.asarray([0,1],dtype = f32)

# dirction = grad(sdf)

# @jit
# def vmap_sdf(r_batched):
#     return vmap(sdf)(r_batched)

# @jit

# def vmap_dirction(r_batched):
#     return vmap(dirction)(r_batched)
# phi_list = jnp.arange(0,Pi,Pi*0.001,dtype = f32)
# ax_list = jnp.cos(phi_list).reshape(1000,1)*0.5
# ay_list = jnp.sin(phi_list).reshape(1000,1)*0.5
# a_list = jnp.append(ax_list,ay_list,axis=1).reshape(1000,2)
# phy_seed = a_list
