#include<stdio.h>
#include"fftz.h"

#define AMD_HD_7570_CW 16
int gpuFFTCoalesced(CLEnv* cl, int N, int R, float* data0){
	cl_program myprogram;
	cl_kernel kernel_gpufft, kernel_gpufft_coalesced;
	cl_mem buffer_data0;
	cl_mem buffer_data1;
	int Ns;
	int even_odd = 0;
	size_t globalws[1];
	size_t localws[1];
	cl_event myevent;
	cl_ulong task_queued, task_start, task_end;
	double total_time = 0;
	int err;

	//compile program
	compileProgram(cl, "gpu_fft.cl", &myprogram);

	//get kernel from the compiled program
	kernel_gpufft = clCreateKernel(myprogram, "gpuFFT", &err);
	if(err < 0){
		clReleaseProgram(myprogram);
		printf("Error: Creating kernel gpuFFT failed!\n");
		return -1;
	}
	kernel_gpufft_coalesced = clCreateKernel(myprogram, "gpuFFTCoalesced", &err);
	if(err < 0){
		clReleaseKernel(kernel_gpufft);
		clReleaseProgram(myprogram);
		printf("Error: Creating kernel gpuFFTCoalesced failed!\n");
		return -1;
	}
	
	//create memory objects
	buffer_data0 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		clReleaseKernel(kernel_gpufft_coalesced);
		clReleaseKernel(kernel_gpufft);
		clReleaseProgram(myprogram);
		printf("Error: Creating Buffer data0 failed with errorcode %d!\n", err);
		return -1;
	}
	buffer_data1 = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * N * 2, NULL, &err);
	if(err < 0){
		clReleaseMemObject(buffer_data0);
		clReleaseKernel(kernel_gpufft_coalesced);
		clReleaseKernel(kernel_gpufft);
		clReleaseProgram(myprogram);
		printf("Error: Creating Buffer data1 failed with errorcode %d!\n", err);
		return -1;
	}
	
	//transfer data from CPU to GPU
	err = clEnqueueWriteBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		clReleaseMemObject(buffer_data1);
		clReleaseMemObject(buffer_data0);
		clReleaseKernel(kernel_gpufft_coalesced);
		clReleaseKernel(kernel_gpufft);
		clReleaseProgram(myprogram);
		printf("Error: Calling clEnqueueWriteBuffer failed!\n");
		return -1;
	}
	
	//run the kernel
	for(Ns = 1; Ns < N; Ns *= R){
		if(Ns >= AMD_HD_7570_CW){
			err += clSetKernelArg(kernel_gpufft, even_odd, sizeof(cl_mem), &buffer_data0);
			err += clSetKernelArg(kernel_gpufft, 1 - even_odd, sizeof(cl_mem), &buffer_data1);
			err += clSetKernelArg(kernel_gpufft, 2, sizeof(int), &N);
			err += clSetKernelArg(kernel_gpufft, 3, sizeof(int), &R);
			err += clSetKernelArg(kernel_gpufft, 4, sizeof(int), &Ns);
			even_odd = 1 - even_odd;
			if(err < 0){
				clReleaseMemObject(buffer_data1);
				clReleaseMemObject(buffer_data0);
				clReleaseKernel(kernel_gpufft_coalesced);
				clReleaseKernel(kernel_gpufft);
				clReleaseProgram(myprogram);
				printf("Error: Calling clSetKernelArg failed!\n");
				return -1;
			}
		
			globalws[0] = N / R;
			if(N / R > 256)
				localws[0] = 256;
			else
				localws[0] = N / R;
			err = clEnqueueNDRangeKernel(cl->queue, kernel_gpufft, 1, NULL, globalws, localws, 0, NULL, &myevent);
			if(err < 0){
				clReleaseMemObject(buffer_data1);
				clReleaseMemObject(buffer_data0);
				clReleaseKernel(kernel_gpufft_coalesced);
				clReleaseKernel(kernel_gpufft);
				clReleaseProgram(myprogram);
				printf("Error: Calling clEnqueueNDRangeKernel failed with error code %d!\n", err);
				return -1;
			}
			clFinish(cl->queue);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
			total_time += (double)(task_end - task_start) / 1000.0;//um
			
			clReleaseEvent(myevent);
		}
		else{
			err += clSetKernelArg(kernel_gpufft_coalesced, even_odd, sizeof(cl_mem), &buffer_data0);
			err += clSetKernelArg(kernel_gpufft_coalesced, 1 - even_odd, sizeof(cl_mem), &buffer_data1);
			err += clSetKernelArg(kernel_gpufft_coalesced, 2, sizeof(int), &N);
			err += clSetKernelArg(kernel_gpufft_coalesced, 3, sizeof(int), &R);
			err += clSetKernelArg(kernel_gpufft_coalesced, 4, sizeof(int), &Ns);
			err += clSetKernelArg(kernel_gpufft_coalesced, 5, sizeof(float) * 512, NULL);
			even_odd = 1 - even_odd;
			if(err < 0){
				clReleaseMemObject(buffer_data1);
				clReleaseMemObject(buffer_data0);
				clReleaseKernel(kernel_gpufft_coalesced);
				clReleaseKernel(kernel_gpufft);
				clReleaseProgram(myprogram);
				printf("Error: Calling clSetKernelArg failed!\n");
				return -1;
			}
		
			globalws[0] = N / R;
			if(N / R > 256)
				localws[0] = 256;
			else
				localws[0] = N / R;
			err = clEnqueueNDRangeKernel(cl->queue, kernel_gpufft_coalesced, 1, NULL, globalws, localws, 0, NULL, &myevent);
			if(err < 0){
				clReleaseMemObject(buffer_data1);
				clReleaseMemObject(buffer_data0);
				clReleaseKernel(kernel_gpufft_coalesced);
				clReleaseKernel(kernel_gpufft);
				clReleaseProgram(myprogram);
				printf("Error: Calling clEnqueueNDRangeKernel failed with error code %d!\n", err);
				return -1;
			}
			clFinish(cl->queue);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &task_start, NULL);
			clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &task_end, NULL);
			total_time += (double)(task_end - task_start) / 1000.0;//um
			
			clReleaseEvent(myevent);
		}
	}
	
	//get results
	if(even_odd == 1)
		err = clEnqueueReadBuffer(cl->queue, buffer_data1, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	else
		err = clEnqueueReadBuffer(cl->queue, buffer_data0, CL_FALSE, 0, sizeof(float) * N * 2, data0, 0, NULL, NULL);
	if(err < 0){
		clReleaseMemObject(buffer_data1);
		clReleaseMemObject(buffer_data0);
		clReleaseKernel(kernel_gpufft_coalesced);
		clReleaseKernel(kernel_gpufft);
		clReleaseProgram(myprogram);
		printf("Error: Calling clEnqueueReadBuffer failed!\n");
		return -1;
	}
	clFinish(cl->queue);

	clReleaseMemObject(buffer_data1);
	clReleaseMemObject(buffer_data0);
	clReleaseKernel(kernel_gpufft_coalesced);
	clReleaseKernel(kernel_gpufft);
	clReleaseProgram(myprogram);
	
	return (int)total_time;
}


