__kernel void gpuFFTCoalesced(__global float* data0, __global float* data1, int N, int R, int Ns, __local float* data_local){
	float v[4];
	float angle;
	float v0, v1;
	int idxS, idxD, idxS_local, idxD_local;
	int T;
	
	T = get_local_size(0);
	idxS = get_global_id(0);
	idxD = get_group_id(0) * T * 2 + get_local_id(0);
	idxS_local = get_local_id(0);
	idxD_local = (idxS_local / Ns) * Ns * R + (idxS_local % Ns);
	angle = -2 * M_PI_F * (idxS % Ns) / (Ns * R);
	v[0] = data0[idxS];
	v[2] = data0[idxS + N / R];
	v[1] = data0[idxS + N];
	v[3] = data0[idxS + N / R + N];
	
	v0 = v[2];
	v1 = v[3];
	v[2] = v0 * cos(angle) - v1 * sin(angle);
	v[3] = v0 * sin(angle) + v1 * cos(angle);
	
	v0 = v[0];
	v1 = v[1];
	v[0] = v0 + v[2];
	v[1] = v1 + v[3];
	v[2] = v0 - v[2];
	v[3] = v1 - v[3];
	
	/******************************************exchange******************************************/
	barrier(CLK_GLOBAL_MEM_FENCE);
	data_local[idxD_local] = v[0];
	data_local[idxD_local + Ns] = v[2];
	data_local[idxD_local + T * 2] = v[1];
	data_local[idxD_local + Ns + T * 2] = v[3];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	v[0] = data_local[idxS_local];
	v[2] = data_local[idxS_local + T];
	v[1] = data_local[idxS_local + T * 2];
	v[3] = data_local[idxS_local + T + T * 2];
	
	/***************************************write to global***************************************/
	data1[idxD] = v[0];
	data1[idxD + T] = v[2];
	data1[idxD + N] = v[1];
	data1[idxD + T + N] = v[3];
}
