#ifdef __APPLE__
	#include<OpenCL/opencl.h>
#else
	#include<CL/cl.h>
#endif

#ifndef __CLEnv__
	#define __CLEnv__
	typedef struct _CLEnv{
		cl_platform_id* platforms;
		cl_platform_id* platform;
		cl_device_id* devices;
		cl_device_id* device;
		cl_context context;
		cl_command_queue queue;
	}CLEnv;
#endif

extern int initCLEnv(CLEnv* cl);
extern int releaseCLEnv(CLEnv* cl);
extern int compileProgram(CLEnv* cl, char* file_name, cl_program* myprogram);

extern int gpuFFT(CLEnv* cl, int N, int R, float* data0);
extern int gpuFFTCoalesced(CLEnv* cl, int N, int R, float* data0);
extern int gpuAmdFFT(CLEnv* cl, int N, int R, float* data0);
