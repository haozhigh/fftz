#include<stdio.h>
#include"fftz.h"
#include<clAmdFft.h>

int gpuAmdFFT(CLEnv* cl, int N, int R, float* data0){
	clAmdFftSetupData fft_setup_data;
	clAmdFftPlanHandle fft_plan;
	
	int i;
	int err;
	size_t size = N;
	cl_event myevent;
	cl_ulong task_queued, task_start, task_end;
	double total_time = 0;
	
	cl_mem buffer0[2]; 
	cl_mem buffer1[2];
	
	buffer0[0] = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
	if(err < 0){
		printf("Error: Failed to create buffer0[0]!\n");
		return -1;
	}
	buffer0[1] = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
	if(err < 0){
		printf("Error: Failed to create buffer0[1]!\n");
		clReleaseMemObject(buffer0[0]);
		return -1;
	}
	buffer1[0] = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
	if(err < 0){
		printf("Error: Failed to create buffer1[0]!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		return -1;
	}
	buffer1[1] = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
	if(err < 0){
		printf("Error: Failed to create buffer1[1]!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		return -1;
	}
	
	err = clEnqueueWriteBuffer(cl->queue, buffer0[0], CL_TRUE, 0, sizeof(float) * N, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Error: Failed to Write buffer0!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	err = clEnqueueWriteBuffer(cl->queue, buffer0[1], CL_TRUE, 0, sizeof(float) * N, data0 + N, 0, NULL, NULL);
	if(err < 0){
		printf("Error: Failed to Write buffer0!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	
	// Get FFT version  
	err = clAmdFftInitSetupData(&fft_setup_data);
	if(err < 0){
		printf("Error: Failed to setup amdfft!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	//printf("Using clAmdFft %u.%u.%u\n",fft_setup_data.major,fft_setup_data.minor,fft_setup_data.patch);

	//FFT Setup
	err = clAmdFftSetup(&fft_setup_data);
	if(err < 0){
		printf("Error: Failed to setup amdfft!\n");
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}

	// Create FFT plan  
	err = clAmdFftCreateDefaultPlan(&fft_plan, cl->context, CLFFT_1D, &size);
	if(err < 0){
		printf("Error: Failed to create default plan!\n");
		clAmdFftTeardown();
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	
	clAmdFftSetPlanPrecision(fft_plan, CLFFT_SINGLE);
	clAmdFftSetPlanBatchSize(fft_plan, 1);
	clAmdFftSetLayout(fft_plan, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
	clAmdFftSetResultLocation(fft_plan, CLFFT_OUTOFPLACE);
	
	err = clAmdFftEnqueueTransform(fft_plan, CLFFT_FORWARD, 1, &cl->queue, 0, NULL, &myevent, buffer0, buffer1, NULL);
	if(err < 0){
		printf("Error: Failed to enqueue transform!\n");
		clAmdFftDestroyPlan(&fft_plan);
		clAmdFftTeardown();
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	clFinish(cl->queue);
	clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
	clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
	total_time = (double)(task_end - task_start) / 1000.0;//um
	
	clReleaseEvent(myevent);
	
	err = clEnqueueReadBuffer(cl->queue, buffer1[0], CL_TRUE, 0, sizeof(float) * N, data0, 0, NULL, NULL);
	if(err < 0){
		printf("Error: Failed to read buffer1!\n");
		clAmdFftDestroyPlan(&fft_plan);
		clAmdFftTeardown();
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	err = clEnqueueReadBuffer(cl->queue, buffer1[1], CL_TRUE, 0, sizeof(float) * N, data0 + N, 0, NULL, NULL);
	if(err < 0){
		printf("Error: Failed to read buffer1!\n");
		clAmdFftDestroyPlan(&fft_plan);
		clAmdFftTeardown();
		clReleaseMemObject(buffer0[0]);
		clReleaseMemObject(buffer0[1]);
		clReleaseMemObject(buffer1[0]);
		clReleaseMemObject(buffer1[1]);
		return -1;
	}
	clFinish(cl->queue);
		
	clAmdFftDestroyPlan(&fft_plan);
	clAmdFftTeardown();	
	clReleaseMemObject(buffer0[0]);
	clReleaseMemObject(buffer0[1]);
	clReleaseMemObject(buffer1[0]);
	clReleaseMemObject(buffer1[1]);
	
	return (int)total_time;
}
