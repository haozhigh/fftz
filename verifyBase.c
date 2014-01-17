#include<stdio.h>
#include<stdlib.h>
#include"fftz.h"
#include<fftw3.h>

static int fftw_results(int N){
	fftw_complex* in;
	fftw_complex* out;
	fftw_plan p;
	int i;
	FILE* file;
	
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	
	for(i = 0; i < N; i++){
		in[i][0] = i * i;
		in[i][1] = i;
	}
	
	fftw_execute(p);
	
	file = fopen("output/verifyBase0", "w");
	for(i = 0; i < N; i++){
		fprintf(file, "%.2f %.2f\n", out[i][0], out[i][1]);
	}	
	fclose(file);
		
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);	
	return 0;
}

int main(int argc, char** argv){
	CLEnv cl;
	float *data0;
	int i;
	FILE* file;
	int N;
	
	if(argc != 2){
		printf("Error: argc doesn't match.\n");
		return -1;
	}
	if(initCLEnv(&cl) < 0){
		printf("Error: Failed to init CLEnv!\n");
		return -1;
	}

	N = atoi(argv[1]);
	data0 = (float*)malloc(sizeof(float) * N * 2);

	printf("fftw start...\n");
	fftw_results(N);

	printf("gpuFFTBase start...\n");
	for(i = 0; i < N; i++){
		data0[i] = i * i;
		data0[i + N] = i;
	}
	gpuFFTBase(&cl, N, data0);
	file = fopen("output/verifyBase1", "w");
	for(i = 0; i < N; i++){
		fprintf(file, "%.2f %.2f\n", data0[i], data0[i + N]);
	}	
	fclose(file);
	
	free(data0);
	releaseCLEnv(&cl);
	return 0;
}
