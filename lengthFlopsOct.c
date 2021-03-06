#include<stdio.h>
#include"fftz.h"

#define I 24
#define N 16777216
#define repeat 10

static void writeResult(int *time, char* file_name){
	FILE* file;
	int i, n;
	
	file = fopen(file_name, "w");
	for(i = 3, n = 8; i <= I; i+=3, n *= 8){
		fprintf(file, "%2d %d %f\n", i, time[i], (double)(5 * i * n) / time[i] / 1000);
	}
	fclose(file);
}

int main(int argc, char** argv){
	CLEnv cl;	
	float* data0;
	int i, j, n;
	int time[I + 1];
	int current_time;
	int least_time;
	
	data0 = (float*)malloc(sizeof(float) * N * 2);
	if(data0 == NULL){
		printf("Error: Failed to call malloc!\n");
		return -1;
	}
	
	if(initCLEnv(&cl) < 0){
		printf("Error: Failed to init CLEnv!\n");
		return -1;
	}
	
	for(i = 0; i < N; i++){
		data0[i] = i;
		data0[i + N] = 0;
	}	
	for(n = 8, i = 3; n <= N; n *= 8, i+=3){
		printf("gpuFFTOct %5d %10d start...\n", i, n);
		least_time = gpuFFTOct(&cl, n, data0);
		for(j = 0; j < repeat - 1; j++){
			current_time = gpuFFTOct(&cl, n, data0);
			if(current_time < least_time)	least_time = current_time;
		}
		time[i] = least_time;
	}
	printf("Writing results to file...\n");
	writeResult(time, "./output/lengthFlopsOct");
	
	releaseCLEnv(&cl);
	free(data0);
	return 0;
}
