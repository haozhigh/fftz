#include<stdio.h>
#include"fftz.h"

static cl_program program = NULL;
static cl_kernel kernel_gpuFFTLocal = NULL;
static cl_kernel kernel_gpuFFTLocalPhase = NULL;
static cl_kernel kernel_gpuFFT = NULL;
static cl_mem buffer_data0 = NULL;
static cl_mem buffer_data1 = NULL;

static void releaseStuff(){
	if(buffer_data1 != NULL)
		clReleaseMemObject(buffer_data1);
	if(buffer_data0 != NULL)
		clReleaseMemObject(buffer_data0);
	if(kernel_gpuFFTLocalPhase != NULL)
		clReleaseKernel(kernel_gpuFFTLocalPhase);
	if(kernel_gpuFFTLocal != NULL)
		clReleaseKernel(kernel_gpuFFTLocal);
	if(kernel_gpuFFT != NULL)
		clReleaseKernel(kernel_gpuFFT);
	if(program != NULL)
		clReleaseProgram(program);
	return;
}

int gpuFFTLocal(CLEnv *cl, int N, float *data0){
	int Ns;
	int even_odd = 0;
	size_t globalws[1];
	size_t localws[1];
	cl_event myevent;
	cl_ulong task_queued, task_start, task_end;
	double total_time = 0;
	int err;

	if(compileProgram(cl, "gpuFFTLocal.cl", &program) < 0){
		printf("Runtime Error: gpuFFTLocal. Failed to call compileProgram.\n");
		releaseStuff();
		return -1;
	}

	kernel_gpuFFTLocal = clCreateKernel(program, "gpuFFTLocal", &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clCreateKernel failed with code %d.\n", err);
		releaseStuff();
		return err;
	}
	kernel_gpuFFTLocalPhase = clCreateKernel(program, "gpuFFTLocalPhase", &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clCreateKernel failed with code %d.\n", err);
		releaseStuff();
		return err;
	}
	kernel_gpuFFT = clCreateKernel(program, "gpuFFT", &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clCreateKernel failed with code %d.\n", err);
		releaseStuff();
		return err;
	}


	buffer_data0 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clCreateBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}
	buffer_data1 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clCreateBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	err = clEnqueueWriteBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clEnqueueWriteBUffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	if(N <= 512){
		err += clSetKernelArg(kernel_gpuFFTLocal, 0, sizeof(cl_mem), &buffer_data0);
		err += clSetKernelArg(kernel_gpuFFTLocal, 1, sizeof(float) * N * 2, NULL);
		err += clSetKernelArg(kernel_gpuFFTLocal, 2, sizeof(int), &N);
		if(err < 0){
			printf("Runtime Error: gpuFFTLocal. Setting args of kernel_gpuFFTLocal failed with code %d.\n", err);
			releaseStuff();
			return err;
		}
		globalws[0] = N / 2;
		localws[0] = N / 2;
		err = clEnqueueNDRangeKernel(cl->queue, kernel_gpuFFTLocal, 1, NULL, globalws, localws, 0, NULL, &myevent);
		if(err < 0){
			printf("Runtime Error: gpuFFTLocal. Enqueuing kernel_gpuFFTLocall failed with code %d.\n", err);
			releaseStuff();
			return err;
		}
		clFinish(cl->queue);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
		total_time = (double)(task_end - task_start) / 1000.0;	//us
		clReleaseEvent(myevent);
	}
	else{
		err += clSetKernelArg(kernel_gpuFFTLocalPhase, 0, sizeof(cl_mem), &buffer_data0);
		err += clSetKernelArg(kernel_gpuFFTLocalPhase, 1, sizeof(float) * 1024, NULL);
		err += clSetKernelArg(kernel_gpuFFTLocalPhase, 2, sizeof(int), &N);
		if(err < 0){
			printf("Runtime Error: gpuFFTLocal. Setting args of kernel_gpuFFTLocalPhase failed with code %d.\n", err);
			releaseStuff();
			return err;
		}
		
		globalws[0] = N / 2;
		localws[0] = 256;
		err = clEnqueueNDRangeKernel(cl->queue, kernel_gpuFFTLocalPhase, 1, NULL, globalws, localws, 0, NULL, &myevent);
		if(err < 0){
			printf("Runtime Error: gpuFFTLocal. Enqueuing kernel_gpuFFTLocalPhase failed with code %d.\n", err);
			releaseStuff();
			return err;
		}
		clFinish(cl->queue);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
		total_time += (double)(task_end - task_start) / 1000.0;	//us
		clReleaseEvent(myevent);

		for(Ns = 512; Ns < N; Ns *= 2){
			err += clSetKernelArg(kernel_gpuFFT, even_odd, sizeof(cl_mem), &buffer_data0);
			err += clSetKernelArg(kernel_gpuFFT, 1 - even_odd, sizeof(cl_mem), &buffer_data1);
			err += clSetKernelArg(kernel_gpuFFT, 2, sizeof(int), &N);
			err += clSetKernelArg(kernel_gpuFFT, 3, sizeof(int), &Ns);
			even_odd = 1 - even_odd;
			if(err < 0){
				printf("Runtime Error: gpuFFTLocal. Setting args of kernel_gpuFFT failed with code %d.\n", err);
				releaseStuff();
				return err;
			}

			globalws[0] = N / 2;
			localws[0] = 256;
			err = clEnqueueNDRangeKernel(cl->queue, kernel_gpuFFT, 1, NULL, globalws, localws, 0, NULL, &myevent);
			if(err < 0){
				printf("Runtime Error: gpuFFTLocal. Enqueuing kernel_gpuFFT failed with code %d.\n", err);
				releaseStuff();
				return err;
			}
			clFinish(cl->queue);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
			total_time += (double)(task_end - task_start) / 1000.0;	//us
			clReleaseEvent(myevent);
		}
	}

	if(even_odd == 1)
		err = clEnqueueReadBuffer(cl->queue, buffer_data1, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	else
		err = clEnqueueReadBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Runtime Error: gpuFFTLocal. Calling clEnqueueReadBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	clFinish(cl->queue);
	releaseStuff();
	return (int)total_time;
}
