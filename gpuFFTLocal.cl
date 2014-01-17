__kernel void gpuFFTLocal(__global float *data0, __local float *data0_local, int N){
	int idxS, idxD;
	int Ns;
	float v0, v1;
	float v[4];
	float angle;

	idxS = get_global_id(0);
	data0_local[idxS] = data0[idxS];
	data0_local[idxS + N] = data0[idxS + N];
	data0_local[idxS + N / 2] = data0[idxS + N / 2];
	data0_local[idxS + N / 2 + N] = data0[idxS + N / 2 + N];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(Ns = 1; Ns < N; Ns = Ns * 2){
		v[0] = data0_local[idxS];
		v[1] = data0_local[idxS + N];
		v[2] = data0_local[idxS + N / 2];
		v[3] = data0_local[idxS + N / 2 + N];
		barrier(CLK_LOCAL_MEM_FENCE);

		angle = -M_PI_F * (idxS % Ns) / Ns;
		v0 = v[2] * cos(angle) - v[3] * sin(angle);
		v1 = v[2] * sin(angle) + v[3] * cos(angle);
		v[2] = v[0] - v0;
		v[3] = v[1] - v1;
		v[0] = v[0] + v0;
		v[1] = v[1] + v1;

		idxD = (idxS / Ns) * Ns * 2 + (idxS % Ns);
		data0_local[idxD] = v[0];
		data0_local[idxD + N] = v[1];
		data0_local[idxD + Ns] = v[2];
		data0_local[idxD + Ns + N] = v[3];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	data0[idxS] = data0_local[idxS];
	data0[idxS + N] = data0_local[idxS + N];
	data0[idxS + N / 2] = data0_local[idxS + N / 2];
	data0[idxS + N / 2 + N] = data0_local[idxS + N / 2 + N];
}
	
__kernel void gpuFFTLocalPhase(__global float *data0, __local float *data0_local, int N){
	int idxS, idxD;
	int Ns;
	float v0, v1;
	float v[4];
	float angle;
	int group;
	int local_id;

	group = get_group_id(0);
	local_id = get_local_id(0);
	idxS = group + N / 512 * get_local_id(0);
	data0_local[local_id] = data0[idxS];
	data0_local[local_id + 512] = data0[idxS + N];
	data0_local[local_id + 256] = data0[idxS + N / 2];
	data0_local[local_id + 768] = data0[idxS + N / 2 + N];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(Ns = 1; Ns < 512; Ns = Ns * 2){
		v[0] = data0_local[local_id];
		v[1] = data0_local[local_id + 512];
		v[2] = data0_local[local_id + 256];
		v[3] = data0_local[local_id + 768];
		barrier(CLK_LOCAL_MEM_FENCE);

		angle = -M_PI_F * (local_id % Ns) / Ns;
		v0 = v[2] * cos(angle) - v[3] * sin(angle);
		v1 = v[2] * sin(angle) + v[3] * cos(angle);
		v[2] = v[0] - v0;
		v[3] = v[1] - v1;
		v[0] = v[0] + v0;
		v[1] = v[1] + v1;

		idxD = (local_id / Ns) * Ns * 2 + (local_id % Ns);
		data0_local[idxD] = v[0];
		data0_local[idxD + 512] = v[1];
		data0_local[idxD + Ns] = v[2];
		data0_local[idxD + Ns + 512] = v[3];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	idxD = 512 * group + local_id;
	data0[idxD] = data0_local[local_id];
	data0[idxD + N] = data0_local[local_id + 512];
	data0[idxD + 256] = data0_local[local_id + 256];
	data0[idxD + 256 + N] = data0_local[local_id + 768];
}

__kernel void gpuFFT(__global float* data0, __global float* data1, int N, int Ns){
	float v[4];
	float angle;
	float v0, v1;
	int idxS, idxD;
	
	idxS = get_global_id(0);
	v[0] = data0[idxS];
	v[1] = data0[idxS + N];
	v[2] = data0[idxS + N / 2];
	v[3] = data0[idxS + N / 2 + N];
	
	angle = -M_PI_F * (idxS % Ns) / Ns;
	v0 = v[2] * cos(angle) - v[3] * sin(angle);
	v1 = v[2] * sin(angle) + v[3] * cos(angle);
	v[2] = v[0] - v0;
	v[3] = v[1] - v1;
	v[0] = v[0] + v0;
	v[1] = v[1] + v1;
	
	idxD = (idxS / Ns) * Ns * 2 + (idxS % Ns);
	data1[idxD] = v[0];
	data1[idxD + N] = v[1];
	data1[idxD + Ns] = v[2];
	data1[idxD + Ns + N] = v[3];
}
