#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

/*--------------------------------------------------------------------------------*/

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
    #include <mach/mach_time.h>
    typedef uint64_t                    time_delta_t;
    typedef mach_timebase_info_data_t   frequency_t;
#else
    #include <CL/cl.h>
    typedef struct timeval              time_delta_t;
    typedef double                      frequency_t;
#endif

/*--------------------------------------------------------------------------------*/

#define LOCAL_WORK_SIZE 32
#define MAX_RANDOM_FLOAT 10
#define DEFAULT_NUM_VERTICES 512*512
#define DEFAULT_NUM_EDGES 8

#define DEFAULT_KERNEL_FILENAME ("kernel.cl")
#define problem(...) fprintf(stderr, __VA_ARGS__)
#define BAR "--------------------------------------------------------------------------------\n"

/*--------------------------------------------------------------------------------*/

static const char*
GetErrorString(cl_int error) {
    switch(error)
    {
    case(CL_SUCCESS):                           return "Success";
    case(CL_DEVICE_NOT_FOUND):                  return "Device not found!";
    case(CL_DEVICE_NOT_AVAILABLE):              return "Device not available!";
    case(CL_MEM_OBJECT_ALLOCATION_FAILURE):     return "Memory object allocation failure!";
    case(CL_OUT_OF_RESOURCES):                  return "Out of resources!";
    case(CL_OUT_OF_HOST_MEMORY):                return "Out of host memory!";
    case(CL_PROFILING_INFO_NOT_AVAILABLE):      return "Profiling information not available!";
    case(CL_MEM_COPY_OVERLAP):                  return "Overlap detected in memory copy operation!";
    case(CL_IMAGE_FORMAT_MISMATCH):             return "Image format mismatch detected!";
    case(CL_IMAGE_FORMAT_NOT_SUPPORTED):        return "Image format not supported!";
    case(CL_INVALID_VALUE):                     return "Invalid value!";
    case(CL_INVALID_DEVICE_TYPE):               return "Invalid device type!";
    case(CL_INVALID_DEVICE):                    return "Invalid device!";
    case(CL_INVALID_CONTEXT):                   return "Invalid context!";
    case(CL_INVALID_QUEUE_PROPERTIES):          return "Invalid queue properties!";
    case(CL_INVALID_COMMAND_QUEUE):             return "Invalid command queue!";
    case(CL_INVALID_HOST_PTR):                  return "Invalid host pointer address!";
    case(CL_INVALID_MEM_OBJECT):                return "Invalid memory object!";
    case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):   return "Invalid image format descriptor!";
    case(CL_INVALID_IMAGE_SIZE):                return "Invalid image size!";
    case(CL_INVALID_SAMPLER):                   return "Invalid sampler!";
    case(CL_INVALID_BINARY):                    return "Invalid binary!";
    case(CL_INVALID_BUILD_OPTIONS):             return "Invalid build options!";
    case(CL_INVALID_PROGRAM):                   return "Invalid program object!";
    case(CL_INVALID_PROGRAM_EXECUTABLE):        return "Invalid program executable!";
    case(CL_INVALID_KERNEL_NAME):               return "Invalid kernel name!";
    case(CL_INVALID_KERNEL):                    return "Invalid kernel object!";
    case(CL_INVALID_ARG_INDEX):                 return "Invalid index for kernel argument!";
    case(CL_INVALID_ARG_VALUE):                 return "Invalid value for kernel argument!";
    case(CL_INVALID_ARG_SIZE):                  return "Invalid size for kernel argument!";
    case(CL_INVALID_KERNEL_ARGS):               return "Invalid kernel arguments!";
    case(CL_INVALID_WORK_DIMENSION):            return "Invalid work dimension!";
    case(CL_INVALID_WORK_GROUP_SIZE):           return "Invalid work group size!";
    case(CL_INVALID_GLOBAL_OFFSET):             return "Invalid global offset!";
    case(CL_INVALID_EVENT_WAIT_LIST):           return "Invalid event wait list!";
    case(CL_INVALID_EVENT):                     return "Invalid event!";
    case(CL_INVALID_OPERATION):                 return "Invalid operation!";
    case(CL_INVALID_GL_OBJECT):                 return "Invalid OpenGL object!";
    case(CL_INVALID_BUFFER_SIZE):               return "Invalid buffer size!";
    case(CL_COMPILER_NOT_AVAILABLE):            return "Compiler not available!";
    case(CL_BUILD_PROGRAM_FAILURE):             return "Build failure!";
    default:                                    return "Unknown error!";
    };
    return "Unknown error";
}

/*--------------------------------------------------------------------------------*/

static unsigned long ReadFromTextFile(FILE *fh, char* buffer, size_t buffer_size) {
    unsigned long count = (unsigned long)fread(buffer, buffer_size, 1, fh);
    buffer[buffer_size] = '\0';
    return count;
}

static char *LoadTextFromFile(const char *filename, unsigned long *size /* returned file size in bytes */) {
  FILE* fh;
  struct stat statbuf;
  
  //Open File.
  fh = fopen(filename, "r");
  if (!fh)
    problem("File did not open successfully\n");
  
  //Get file size.
  stat(filename, &statbuf);
  unsigned long bytes = (*size);
  
  if(size)
    (*size) = (unsigned long)statbuf.st_size;
  bytes = *size;

  //To be returned.
  char *text = (char*)malloc(*size + 1);
  if(!text)
    return 0;
  
  ReadFromTextFile(fh, text, bytes);
  fclose(fh);
  
  return text;
}

/*--------------------------------------------------------------------------------*/

void check_failure(cl_int err) {
  if (err != CL_SUCCESS) {
    problem("%s", GetErrorString(err));
    exit(err);
  }
}

void printArray(cl_float *matrix, cl_int num) {
  int i;
  for(i = 0; i < num; i++) {
    printf("%4.0f ", matrix[i]);
  }
}

void UIprintArray(cl_uint *matrix, cl_uint num) {
  cl_uint i;
  for(i = 0; i < num; i++) {
    printf("(%ud, %ud) ", i, matrix[i]);
  }
}


/*--------------------------------------------------------------------------------*/

struct timeval tv_delta(struct timeval start, struct timeval end){
  struct timeval delta = end;
  delta.tv_sec -= start.tv_sec;
  delta.tv_usec -= start.tv_usec;
  if (delta.tv_usec < 0) {
    delta.tv_usec += 1000000;
    delta.tv_sec--;
  }
  return delta;
}

/*--------------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  cl_int err;

  //Get device id.
  cl_device_id device_id;
  cl_platform_id platform = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device_id, NULL);
  check_failure(err);
  
  //Output the name of our device.
  cl_char vendor_name[1024];
  cl_char device_name[1024];
  err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
  err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  check_failure(err);
  printf("Using %s %s. \n", vendor_name, device_name);
  printf(BAR);
  
  //Create a context.
  cl_context context;
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  check_failure(err);

  //Create a command queue.
  cl_command_queue commands;
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  check_failure(err);

  //Load kernel from file into a string.
  char *source;
  unsigned long source_length = 0;
  if(argc < 2) {
    source = LoadTextFromFile(DEFAULT_KERNEL_FILENAME, &source_length);
  } else {
    source = LoadTextFromFile(argv[1], &source_length);
  }
  
  //Create our kernel.
  cl_program program;
  cl_kernel kernel;
  program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  check_failure(err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char buffer[9999];
    
    problem("ERROR: Failed to build program executable! %s\n", GetErrorString(err));
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				sizeof(buffer), buffer, NULL);
    check_failure(err);
    problem("okay...%s\n", buffer);
    return EXIT_FAILURE;
  }
  kernel = clCreateKernel(program, "test_determine_work_item", &err);
  check_failure(err);
  
  cl_uint num_vertices = DEFAULT_NUM_VERTICES;
  cl_uint num_edges = DEFAULT_NUM_EDGES;

  if(argc > 2) {
    num_vertices = atoi(argv[1]);
    num_edges = atoi(argv[2]);
  }
    
  cl_uint  *edge_sums = (cl_uint *)malloc(sizeof(cl_uint)*num_vertices+1);
  cl_uint i, sum;
  sum = 0;
  for(i = 0; i < num_vertices; i++) {
    edge_sums[i] = sum;
    sum += (rand() % (num_edges-1)) + 1;
  }
  edge_sums[num_vertices] = sum+200;
  UIprintArray(edge_sums + (num_vertices-100), 100);
  
  cl_mem _edge_sums;
  cl_mem _results;
  cl_int thread_count = sum & 31 ? sum + LOCAL_WORK_SIZE - (sum & 31) : sum;
  

  
  
  printf("Creating data buffers.\n");
  printf(BAR);
  //Create data buffers on the device.
  

  _edge_sums = clCreateBuffer(context,  CL_MEM_READ_ONLY,
				 sizeof(cl_uint)*num_vertices+1,  NULL, NULL);
  _results = clCreateBuffer(context, CL_MEM_READ_WRITE,
				 sizeof(cl_uint)*thread_count, NULL, NULL);

  
  if(!_edge_sums || !_results) {
    problem("Failed to allocate device memory.\n");
    exit(-1);
  }
  
  printf("Putting data into device memory.\n");
  printf(BAR);
  //Put data into device Memory.
  err  =  clEnqueueWriteBuffer(commands, _edge_sums, CL_TRUE, 0, 
			       sizeof(cl_uint)*num_vertices+1, edge_sums, 0, NULL, NULL);

  check_failure(err);


  
  int a = 0;
  printf("Setting Kernel Arguments.\n");
  printf(BAR);
  //Set arguments.


  err  =  clSetKernelArg(kernel, a++, sizeof(cl_mem), &_edge_sums);
  err |=  clSetKernelArg(kernel, a++, sizeof(cl_mem), &_results);
  err |=  clSetKernelArg(kernel, a++, sizeof(cl_uint), &num_vertices);
  
  check_failure(err);

  clFinish(commands);
  printf("Running.\n");
  printf(BAR);

  size_t global[] = {thread_count};
  size_t local[] = {LOCAL_WORK_SIZE};
  //Run our program.

  
  struct timeval start, end, delta;
  gettimeofday(&start, NULL);


  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global, local, 0, NULL, NULL);

  
  check_failure(err);
  clFinish(commands);
  gettimeofday(&end, NULL);
  delta = tv_delta(start, end);
  printf("GPU Time: %ld.%06ld\n", 
	 (long int)delta.tv_sec, 
	 (long int)delta.tv_usec);
  printf(BAR);

  
  cl_uint *result;
  result = (cl_uint *)malloc(sizeof(cl_uint)*sum);
  
  printf("Getting data.\n");
  printf(BAR);
  //Retrieve and print output.
  err = clEnqueueReadBuffer(commands, _results, CL_TRUE, 0, sizeof(cl_uint)*sum,
			    result, 0, NULL, NULL );
  printf("%ud\n", thread_count - sum);
  UIprintArray(result + (thread_count-100), 100);
  check_failure(err);
  
  free(result);
  result = NULL;
  printf(BAR);
  printf("Cleanup.\n");
  //Device Cleanup.
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  clReleaseMemObject(_edge_sums);
    clReleaseMemObject(_results);
  //Memory Cleanup.
    free(edge_sums);
  
  
  return 0;
}
