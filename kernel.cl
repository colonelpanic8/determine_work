#define LOCAL_WORK_SIZE 32
#define LOCAL_WORK_SHIFT 5 //Set to log base 2 of LOCAL_WORK_SIZE
#define MOD32 31

__kernel void test_determine_work_item(
				       __global uint *sums, //NOTE: this array has size num_active_vertices + 1
				       __global uint *results,
				       uint num_active_vertices				 
				       )
{
  
  uint local_id = get_local_id(0);
  uint thread_id = get_global_id(0);
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint loading_id = local_id + 1;

  uint __local size = num_active_vertices >> LOCAL_WORK_SHIFT;
  uint __local location = 0;
  uint __local vals[LOCAL_WORK_SIZE+1];

  vals[LOCAL_WORK_SIZE] = start + 1;
  
  while(size > 0) {
    uint pos = location + size*(loading_id);
    vals[local_id] = sums[pos]; //this is redundant for the last thread... but whatever
    if(vals[local_id] <= start && vals[local_id+1] > start) {
      //this will only be true for one of our threads
      location = pos;
    }
    size = size >> LOCAL_WORK_SHIFT;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  vals[local_id] = location + local_id < num_active_vertices+1 ? 
    sums[location+local_id] : 100000000;
  int i;
  for(i = 0; i < LOCAL_WORK_SIZE-1; i++) {
    if(vals[i+1] > thread_id)
      break;
  }
  results[thread_id] = location + i;
}

/*

typedef struct _edge {
  int dest;
  float weight;
} edge;

typedef stuct _update {
  float val;
  uint vertex;
  uint source
} update;


__kernel void determine_min(
		       __global float *distances,
		       __global update *updates,
		       )
{
  

}
*/

/*
__kernel void relax_edge(
		    __global float *distances,
		    __global edge *edges,
		    __global int *work_items,
		    __global int *vertex_edge_indices,
		    __global int *vertex_list,
		    __global update *updates
		    )
{
  int my_edge = determine_edge();
  edge __local the_edges[LOCAL_WORK_SIZE];
  the_edges[local_id] = edges[edge_ids[my_edge]];
  
  
  if(the_edges[local_id] +

  
  
  
  

}
*/
