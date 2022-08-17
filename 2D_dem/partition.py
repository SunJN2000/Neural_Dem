import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lib import xla_bridge as xla
import time
i32 = jnp.int32
f32 = jnp.float32
Array = jnp.ndarray
from functools import partial 
from jax import random
r_max  = 1.00
dimensionality = 2
boundary_space = jnp.asarray([10,10],dtype = f32)
#key = random.PRNGKey(3232323)
#p_batch=random.uniform(key,(10,2),dtype=f32,minval=0.0, maxval=5.0)

def get_id(p_batch,r_c : f32=r_max):
    t_batch = p_batch * (1 / r_c)
    r_batch = jnp.floor(t_batch)
    r_batch = r_batch+1
    r_batch = r_batch.astype(i32)
    return r_batch
def get_box(space ,r_c : f32=r_max):
    t_batch = space * (1 / r_c)
    r_batch = jnp.ceil(t_batch)
    r_batch =r_batch.astype(i32)
    return r_batch
def _compute_hash_multiplier(cell_size,
                            dim:i32 = dimensionality)  :
    dim_cell = cell_size.size
    if (dim_cell == dim):
        t_one = jnp.asarray([1],dtype=i32)
        cell_size = cell_size[::-1]
        t_batch = jnp.concatenate((t_one,cell_size[:-1]))
        r_batch = jnp.cumprod(t_batch)[::-1]

        return r_batch 
    else:
        raise ValueError()


def _unflatten_cell_buffer(arr: Array,
                           cells_per_side: Array,
                           dim: int = dimensionality) -> Array:
    if (isinstance(cells_per_side, int) or
        isinstance(cells_per_side, float) or
        (isinstance(cells_per_side, jnp.ndarray) and not cells_per_side.shape)):
        cells_per_side = (int(cells_per_side),) * dim
    elif isinstance(cells_per_side, jnp.ndarray) and len(cells_per_side.shape) == 1:
        cells_per_side = tuple([int(x) for x in cells_per_side[::-1]])
    elif isinstance(cells_per_side, jnp.ndarray) and len(cells_per_side.shape) == 2:
        cells_per_side = tuple([int(x) for x in cells_per_side[0][::-1]])
    else:
        raise ValueError()
    return np.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])



boxsize = get_box(boundary_space)
cell_perside = boxsize + 2
Cell_perside = cell_perside.astype(i32)
temp_batch = jnp.cumprod(Cell_perside)
Total_num_cell = temp_batch[-1]


def get_grids(p_batch ,r_c : f32 = r_max,cell_perside = Cell_perside):
    hash_multiplier = _compute_hash_multiplier(cell_perside)
    num_c : i32 = 10
    indices=get_id(p_batch)
    N=p_batch.shape[0]
    temp_batch = jnp.cumprod(cell_perside)
    total_num_cell = temp_batch[-1]
    cell_id = N * jnp.ones((1440,1), dtype=i32)
    #cell_id = N * jnp.ones((int(144*num_c),1), dtype=int)
    hashes = jnp.sum(indices * hash_multiplier, axis=1)
    particle_id = jax.lax.iota(i32, N)
    sort_map = jnp.argsort(hashes)
    sorted_hash = hashes[sort_map]
    sorted_id = particle_id[sort_map]
    sorted_cell_id = jnp.mod(particle_id, num_c)
    sorted_cell_id = sorted_hash * num_c + sorted_cell_id
    sorted_id = jnp.reshape(sorted_id, (N, 1))
    cell_id = jax.ops.index_update(cell_id, sorted_cell_id, sorted_id)
    cell_id = cell_id.reshape(12,12,10)
    return  cell_id,indices



def get_neighbour_indices(indices,dim:int=dimensionality):
    offset = jnp.arange(-1, 2, 1, dtype=jnp.int32)
    offsets = jnp.stack(jnp.meshgrid(*([offset]*dim), indexing='ij')).reshape(dim, -1)
    expanded_indices =  indices[:, :, None] + offsets[None, :, :]
    return jnp.transpose(expanded_indices, axes=(0, 2, 1))
@jit
def prune_neighbour_list(p_batch, r_c : float = r_max,num_c : i32 = 10,dim:int=dimensionality,cell_perside = Cell_perside,total_num_cell = Total_num_cell):
    cell_id,indices = get_grids(p_batch)
    N=p_batch.shape[0]
    neighbour_list=get_neighbour_indices(indices)
    hash_multiplier=_compute_hash_multiplier(cell_perside)
    hashes = jnp.sum(neighbour_list* hash_multiplier, axis=2)
    temp_cell=cell_id.reshape(total_num_cell,-1)
    neighour_ids=temp_cell[hashes]
    dim_num = int(3**dim)
    n_occupancy = 10
    x=p_batch
    n_objects = x.shape[0]
    neighour_x = x[neighour_ids].reshape(N,dim_num*num_c,-1)
    neighour_ids=neighour_ids.reshape(N,dim_num*num_c)
    mutual_vectors = neighour_x - x[:, None, :]
    mutual_distances = jnp.sqrt(jnp.sum(mutual_vectors**2, axis=-1))
    mutual_intersected_distances = (-1)* mutual_distances+r_c
    mask = jnp.logical_and(mutual_intersected_distances > 0, neighour_ids < n_objects)
    out_idx = n_objects * jnp.ones(neighour_ids.shape, jnp.int32)
    cumsum = jnp.cumsum(mask, axis=1)
    index = jnp.where(mask, cumsum - 1, neighour_ids.shape[1] - 1)
    p_index = jnp.arange(neighour_ids.shape[0])[:, None]
    out_idx = jax.ops.index_update(out_idx, jax.ops.index[p_index, index], neighour_ids)
    return out_idx[:, :n_occupancy]


