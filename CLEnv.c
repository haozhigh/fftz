#include"fftz.h"
#include<stdio.h>
#include<string.h>

int initCLEnv(CLEnv* cl){
	int num_platforms;
	int num_devices;
	int err;
	int i, j;
	char* str;
	size_t str_size;
	char vendor_amd[] = "Advanced Micro Devices";
	
	//get platform ids
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	if(err < 0){
		printf("Error: Calling clGetPlatformIDs the first time failed!\n");
		return err;
	}
	cl->platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, cl->platforms, NULL);
	if(err < 0){
		free(cl->platforms);
		printf("Error: Calling clGetPlatformIDs the second time failed!\n");
		return err;
	}
	
	for(i = 0; i < num_platforms; i++){
		err = clGetDeviceIDs(cl->platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		if(err < 0){
			printf("Error: Calling clGetDeviceIDs the first time failed.\n");
			free(cl->platforms);
			return err;
		}
		cl->devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		clGetDeviceIDs(cl->platforms[i], CL_DEVICE_TYPE_ALL, num_devices, cl->devices, NULL);
		if(err < 0){
			printf("Error: Calling clGetDeviceIDs the second time failed.\n");
			free(cl->devices);
			free(cl->platforms);
			return err;
		}

		for(j = 0; j < num_devices; j++){
			clGetDeviceInfo(cl->devices[j], CL_DEVICE_VENDOR, 0, NULL, &str_size);
			str = (cl_char*)malloc(sizeof(cl_char) * str_size + 1);
			clGetDeviceInfo(cl->devices[j], CL_DEVICE_VENDOR, str_size, str, NULL);
			if(strstr(str, vendor_amd) >= 0){
				printf("Selected device info:\n");
				free(str);

				clGetPlatformInfo(cl->platforms[i], CL_PLATFORM_NAME, 0, NULL, &str_size);
				str = (cl_char*)malloc(sizeof(cl_char) * str_size + 1);
				clGetPlatformInfo(cl->platforms[i], CL_PLATFORM_NAME, str_size, str, NULL);
				printf("  Platform Name: %s\n", str);
				free(str);

				clGetDeviceInfo(cl->devices[j], CL_DEVICE_NAME, 0, NULL, &str_size);
				str = (cl_char*)malloc(sizeof(cl_char) * str_size + 1);
				clGetDeviceInfo(cl->devices[j], CL_DEVICE_NAME, str_size, str, NULL);
				printf("  Device Name: %s\n", str);
				free(str);

				clGetDeviceInfo(cl->devices[j], CL_DEVICE_VENDOR, 0, NULL, &str_size);
				str = (cl_char*)malloc(sizeof(cl_char) * str_size + 1);
				clGetDeviceInfo(cl->devices[j], CL_DEVICE_VENDOR, str_size, str, NULL);
				printf("  Device Vendor: %s\n", str);
				free(str);

				cl->platform = &cl->platforms[i];
				cl->device = &cl->devices[j];

				break;
			}
			free(str);
		}
	}
	
	//create opencl host context
	cl->context = clCreateContext(NULL, 1, cl->device, NULL, NULL, &err);
	if(err < 0){
		free(cl->platforms);
		free(cl->devices);
		printf("Error: Calling clCreateContext failed!\n");
		return err;
	}
	
	//create opencl command queue
	cl->queue = clCreateCommandQueue(cl->context, cl->device[0], CL_QUEUE_PROFILING_ENABLE, &err);
	if(err < 0){
		free(cl->platforms);
		free(cl->devices);
		clReleaseContext(cl->context);
		printf("Error: Calling clCreateCommandQueue failed!\n");
		return err;
	}
	
	//init succeeded
	return 0;
}

int releaseCLEnv(CLEnv* cl){
	clReleaseCommandQueue(cl->queue);
	clReleaseContext(cl->context);
	free(cl->devices);
	free(cl->platforms);
	return 0;
}

int compileProgram(CLEnv* cl, char* file_name, cl_program* myprogram){
	FILE* file;
	size_t source_size;
	char* source_str;
	size_t program_build_log_size;
	char* program_build_log;
	int err;
	
	//read program source file
	file = fopen(file_name, "rb");
	if(!file){
		printf("Error: Failed to open the kernel file!\n");
		return -1;
	}
	fseek(file, 0, SEEK_END);
	source_size = ftell(file);
	rewind(file);
	source_str = (char*)malloc(sizeof(char) * (source_size + 1));
	source_str[source_size] = '\0';
	fread(source_str, sizeof(char), source_size, file);
	fclose(file);
	
	//compile program
	(*myprogram) = clCreateProgramWithSource(cl->context, 1, (const char**)&source_str, NULL, &err);
	err += clBuildProgram((*myprogram), 1, cl->device, NULL, NULL, NULL);
	if(err < 0){
		//print program compiling compiling information
		clGetProgramBuildInfo((*myprogram), cl->device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &program_build_log_size);
		program_build_log = (char*)malloc(sizeof(char) * program_build_log_size);
		clGetProgramBuildInfo((*myprogram), cl->device[0], CL_PROGRAM_BUILD_LOG, program_build_log_size, program_build_log, NULL);
		printf("%s\n", program_build_log);
		free(program_build_log);
		
		free(source_str);
		printf("Error: Compiling program failed!\n");
		return err;
	}
	
	free(source_str);
	return 0;
}

static int mystrncpy(char *dest, char *src, int n){
	int i;

	for(i = 0; i < n; i++)
		*dest++ = *src++;
	*dest = '\0';
	return 0;
}

int strrep(char *src, char * src_temp, char *match, char *replace){
	char *find_pos;

	find_pos = strstr(src, match);
	if((!find_pos) || (!match))
		return -1;
	while(find_pos){
		mystrncpy(src_temp, src, find_pos - src);
		strcat(src_temp, replace);
		strcat(src_temp, find_pos + strlen(match));
		strcpy(src, src_temp);

		find_pos = strstr(src, match);
	}

	return 0;
}
