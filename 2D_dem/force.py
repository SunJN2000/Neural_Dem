import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge
from sdf import vmap_sdf,vmap_dirction,phy_seed
from partition import prune_neighbour_list
from energy import get_energy,get_force,get_energy_bound,get_force_wall
import numpy as np
 
i32 = jnp.int32
f32 = jnp.float32
Array = jnp.ndarray
from jax import random
key = random.PRNGKey(3232323)
boundary_space = jnp.asarray([10,10],dtype = f32) 

R_list = np.zeros((100,3),dtype = f32)
for i in range(100):
    x = jnp.floor(i/10) 
    y = i%10
    R_list[i][0] = x+0.5
    R_list[i][1] = y+0.5


def get_reaction(index1:i32,index2:i32,p_batch:Array):
	n_objects = p_batch.shape[0]
	p_1 = p_batch[index1].reshape(-1)
	p_2 = p_batch[index2].reshape(-1)
	mask_1 = jnp.where(index2 == n_objects,1,0)
	mask_2 = jnp.where(index2 == index1,1,0)
	mask = mask_1 +mask_2
	def f3(_):
		f_list = get_force(p_1,p_2)
		return f_list
	def f2(_):
		
		f_list =(-1)*get_force(p_2,p_1)
		return jax.lax.cond(index1 < index2, f3,lambda _:f_list, None)
	def f1(_):
		f_list = jnp.zeros(3,)
		return f_list	
	return jax.lax.cond(mask>0 , f1, f2, None)


batch_get_reaction_1 = vmap(get_reaction,(None,0,None),0)
batch_get_reaction_2 = vmap(batch_get_reaction_1,(0,0,None),0)

@jit
def get_force_list(p_batch,k:float = 0.1):
	x_batch = p_batch[:,:2]
	neighour_list = prune_neighbour_list(x_batch)
	n_objects = p_batch.shape[0]
	id_list = jnp.arange(0, n_objects,1, dtype=int)
	f_list=batch_get_reaction_2(id_list,neighour_list,p_batch)*k
	f_list = jnp.sum(f_list,axis = 1)
	return f_list
	
def get_energy_list(index1:i32,index2:i32,p_batch:Array):
    
    n_objects = p_batch.shape[0]
    p_1 = p_batch[index1].reshape(-1)
    p_2 = p_batch[index2].reshape(-1)
    mask_1 = jnp.where(index2 == n_objects,1,0)
    mask_2 = jnp.where(index2 == index1,1,0)
    mask = mask_1 +mask_2
    
    def f3(_):
        e_list = get_energy(p_1,p_2)
        return e_list
    def f2(_):
        e_list =get_energy(p_2,p_1)
        return jax.lax.cond(index1 < index2, f3,lambda _:e_list, None)
    def f1(_):
        e_list = 0.0
        return e_list	
    return jax.lax.cond(mask>0 , f1, f2, None)

batch_get_e_1 = vmap(get_energy_list,(None,0,None),0)
batch_get_e_2 = vmap(batch_get_e_1,(0,0,None),0)
@jit
def get_v_list(p_batch,k:float = 0.1):
    x_batch = p_batch[:,:2]
    neighour_list = prune_neighbour_list(x_batch)
    n_objects = p_batch.shape[0]
    id_list = jnp.arange(0, n_objects,1, dtype=int)
    e_list=batch_get_e_2(id_list,neighour_list,p_batch)*k*0.25
    e_list = jnp.sum(e_list)
    return e_list


def get_reaction_of_wall(p,k = 100):
    x_list = p[0]
    y_list = p[1]
    mask_1 = jnp.where(x_list>1,1,0)
    mask_2 = jnp.where(x_list<9,1,0)
    mask_3 = jnp.where(y_list>1,1,0)
    mask_4 = jnp.where(y_list<9,1,0)
    t_1 = mask_1*mask_2
    t_2 = mask_3*mask_4
    mask = t_1*t_2
    def f2(_):
        f_list = get_force_wall(p)
        return f_list*k
    def f1(_):
        f_list = jnp.zeros(3,dtype =f32)
        return f_list
    return jax.lax.cond(mask==0,f2,f1,None)
    
def get_force_of_wall(p_batch):
    return vmap(get_reaction_of_wall)(p_batch)

def get_e_wall(p):
    x_list = p[0]
    y_list = p[1]
    mask_1 = jnp.where(x_list>1,1,0)
    mask_2 = jnp.where(x_list<9,1,0)
    mask_3 = jnp.where(y_list>1,1,0)
    mask_4 = jnp.where(y_list<9,1,0)
    t_1 = mask_1*mask_2
    t_2 = mask_3*mask_4
    mask = t_1*t_2
    def f2(_):
        f_list = get_energy_bound(p)
        return f_list
    def f1(_):
        f_list = 0.0
        return f_list
    return jax.lax.cond(mask==0,f2,f1,None)
    
def get_v_of_wall(p_batch):
    return vmap(get_e_wall)(p_batch)
    
    

    
    
    