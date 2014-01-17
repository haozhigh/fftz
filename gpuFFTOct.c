#include<stdio.h>
#include"fftz.h"

static cl_program program = NULL;
static cl_kernel kernel_gpuFFTOct = NULL;
static cl_mem buffer_data0 = NULL;
static cl_mem buffer_data1 = NULL;

static void releaseStuff(){
	if(buffer_data1 != NULL)
		clReleaseMemObject(buffer_data1);
	if(buffer_data0 != NULL)
		clReleaseMemObject(buffer_data0);
	if(kernel_gpuFFTOct != NULL)
		clReleaseKernel(kernel_gpuFFTOct);
	if(program != NULL)
		clReleaseProgram(program);
	return;
}

int gpuFFTOct(CLEnv *cl, int N, float *data0){
	int Ns, Is;
	int even_odd = 0;
	size_t globalws[1];
	size_t localws[1];
	cl_event myevent;
	cl_ulong task_queued, task_start, task_end;
	double total_time = 0;
	int err;

	if(compileProgram(cl, "gpuFFTOct.cl", &program) < 0){
		printf("Runtime Error: gpuFFTOct. Failed to call compileProgram.\n");
		releaseStuff();
		return -1;
	}

	kernel_gpuFFTOct = clCreateKernel(program, "gpuFFTOct", &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTOct. Calling clCreateKernel failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	buffer_data0 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTOct. Calling clCreateBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}
	buffer_data1 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		printf("Runtime Error: gpuFFTOct. Calling clCreateBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	err = clEnqueueWriteBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Runtime Error: gpuFFTOct. Calling clEnqueueWriteBUffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	for(Ns = 1, Is = 0; Ns < N; Ns *= 8, Is += 3){
		err += clSetKernelArg(kernel_gpuFFTOct, even_odd, sizeof(cl_mem), &buffer_data0);
		err += clSetKernelArg(kernel_gpuFFTOct, 1 - even_odd, sizeof(cl_mem), &buffer_data1);
		err += clSetKernelArg(kernel_gpuFFTOct, 2, sizeof(int), &N);
		err += clSetKernelArg(kernel_gpuFFTOct, 3, sizeof(int), &Ns);
		even_odd = 1 - even_odd;
		if(err < 0){
			printf("Runtime Error: gpuFFTOct. Setting args of kernel_gpuFFTOct failed with code %d.\n", err);
			releaseStuff();
			return err;
		}

		globalws[0] = N / 8;
		if(N / 8 < 256)
			localws[0] = N / 8;
		else
			localws[0] = 256;
		err = clEnqueueNDRangeKernel(cl->queue, kernel_gpuFFTOct, 1, NULL, globalws, localws, 0, NULL, &myevent);
		if(err < 0){
			printf("Runtime Error: gpuFFTOct. Enqueuing kernel_gpuFFTOct failed with code %d.\n", err);
			releaseStuff();
			return err;
		}
		clFinish(cl->queue);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
		total_time += (double)(task_end - task_start) / 1000.0;	//us
		clReleaseEvent(myevent);
	}

	if(even_odd == 1)
		err = clEnqueueReadBuffer(cl->queue, buffer_data1, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	else
		err = clEnqueueReadBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Runtime Error: gpuFFTOct. Calling clEnqueueReadBuffer failed with code %d.\n", err);
		releaseStuff();
		return err;
	}

	clFinish(cl->queue);
	releaseStuff();
	return (int)total_time;
}
