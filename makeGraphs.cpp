#include<mgl/mgl.h>
#include<mgl/mgl_zb.h>
#include<stdio.h>
#include<math.h>

static int graph_length_flops(){
	FILE *file1, *file2, *file3, *file4;
	const int N = 24;
	mglData length(N);
	float time;
	mglData flops1(N), flops2(N), flops3(N), flops4(N);
	int i;
	mglGraphZB gr;
	float min_flops = 1000, max_flops = 0;
	
	file1 = fopen("./output/output_length_flops_fftw", "r");
	file2 = fopen("./output/output_length_flops_gpuFFT", "r");
	file3 = fopen("./output/output_length_flops_gpuFFTCoalesced", "r");
	file4 = fopen("./output/output_length_flops_gpuAmdFFT", "r");
	if(file1 == NULL || file2 == NULL || file3 == NULL || file4 == NULL){
		printf("Error: Failed to open data files!\n");
		return -1;
	}
	
	for(i = 0; i < N; i++){
		fscanf(file1, "%f %f %f", &length.a[i], &time, &flops1.a[i]);
		fscanf(file2, "%f %f %f", &length.a[i], &time, &flops2.a[i]);
		fscanf(file3, "%f %f %f", &length.a[i], &time, &flops3.a[i]);
		fscanf(file4, "%f %f %f", &length.a[i], &time, &flops4.a[i]);
		
		if(flops1.a[i] < min_flops) min_flops = flops1.a[i];
		if(flops2.a[i] < min_flops) min_flops = flops2.a[i];
		if(flops3.a[i] < min_flops) min_flops = flops3.a[i];
		if(flops4.a[i] < min_flops) min_flops = flops4.a[i];
		if(flops1.a[i] > max_flops) max_flops = flops1.a[i];
		if(flops2.a[i] > max_flops) max_flops = flops2.a[i];
		if(flops3.a[i] > max_flops) max_flops = flops3.a[i];
		if(flops4.a[i] > max_flops) max_flops = flops4.a[i];
	}
	fclose(file1);
	fclose(file2);
	fclose(file3);
	fclose(file4);
	
	gr.SetSize(960, 320);
	gr.Alpha(false);
	gr.Light(false);
	
	gr.SetRanges(0, 25, min_flops, max_flops);
	gr.SetTicks('x', 5, 4);
	gr.Axis();
	gr.Grid();
	gr.Label('x', "FFT size I");
	gr.Label('y', "Gflops");
	gr.Plot(length, flops1, "k");
	gr.Plot(length, flops2, "g");
	gr.Plot(length, flops3, "b");
	gr.Plot(length, flops4, "r");
	
	gr.WriteJPEG("output/length_flops.jpg");
	
	return 0;
}

int main(int argc,char* argv[]){
	graph_length_flops();
	
	return 0;
}
