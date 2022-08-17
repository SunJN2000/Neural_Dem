import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
from sdf import vmap_sdf,vmap_dirction,phy_seed
from partition import prune_neighbour_list
import math
pi = math.pi
i32 = jnp.int32
f32 = jnp.float32
Array = jnp.ndarray
from jax import random
key = random.PRNGKey(3232323)
boundary_space = jnp.asarray([10,10],dtype = f32) 

vv = lambda x, y: jnp.dot(x, y)
v_dot = vmap(vv, (0, 0), 0)

def rotation(q,r_batch):
	Trans = jnp.array([[jnp.cos(q),-jnp.sin(q)],[jnp.sin(q),jnp.cos(q)]],dtype=f32).reshape(2,2)
	Temp_batch = jnp.transpose(r_batch,axes=(1,0))
	Tr_batch = jnp.dot(Trans,Temp_batch)
	Tr_batch = jnp.transpose(Tr_batch,axes=(1,0))

	return Tr_batch
	

def get_energy(p_1:Array,p_2:Array,phy_seed:Array = phy_seed,k_n:f32 = 0.5*0.04375):
    q_1 = p_1[2]
    p = p_2-p_1
    x = jnp.asarray([[p[0],p[1]]],dtype = f32)
    x = rotation(-q_1,x).reshape(-1)
    q = p[2]
    s_list = vmap_dirction(phy_seed)
    s_list_1 = rotation(q,s_list)
    note_list = rotation(q,phy_seed) +x
    temp_1 = vmap_sdf(note_list)
    mask_1 = jnp.where(temp_1<=0,1,0).reshape(-1,1)
    s_1 = jnp.sum(mask_1,axis = 0)
    
    r_list_1 =jnp.where(mask_1>0,note_list,0)
    vol_1 = v_dot(r_list_1,s_list_1)*k_n
    
    vol_1 = jnp.sum(vol_1,axis = 0).reshape(1)
    
    x_t = rotation(-q,(-1)*x[None,:])
    note_list_t = rotation(-q,phy_seed)
    note_list_t = note_list_t + x_t
    temp_2 = vmap_sdf(note_list_t)
    mask_2 = jnp.where(temp_2<=0,1,0).reshape(-1,1)
    s_2 = jnp.sum(mask_2,axis = 0)
    
    r_list_2 = jnp.where(mask_2>0,phy_seed,0)
    vol_2 = v_dot(r_list_2,s_list)*k_n
    vol_2 = jnp.sum(vol_2,axis = 0).reshape(1)
    
    vol = vol_1 + vol_2
    return vol[0]


def get_force(p_1,p_2):
    r_force = -grad(get_energy,argnums=0)(p_1,p_2)
    return r_force

def get_energy_bound(p:Array,k:float = 1e2,bound_space:Array = boundary_space):
    x = p[0]
    y = p[1]
    x_r = bound_space[0]
    y_r = bound_space[1]
    n_1 = jnp.where(x>x_r,x-x_r,0)
    n_2 = jnp.where(y>y_r,y-y_r,0)
    n_3 = jnp.where(x<0.0,x,0)
    n_4 = jnp.where(y<0.0,y,0)
    vol = (n_1*n_1+n_2*n_2+n_3*n_3+n_4*n_4)*k
    return vol
def get_force_wall(p):
    r_force = -grad(get_energy_bound,argnums=0)(p)
    return r_force
   
    
    
    
    
    