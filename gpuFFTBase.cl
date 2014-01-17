__kernel void gpuFFTBase(__global float* data0, __global float* data1, int N, int Ns){
	float v[4];
	float angle;
	float v0, v1;
	int idxS, idxD;
	
	idxS = get_global_id(0);
	v[0] = data0[idxS];
	v[1] = data0[idxS + N];
	v[2] = data0[idxS + (N>>1)];
	v[3] = data0[idxS + (N>>1) + N];
	
	angle = -M_PI_F * (idxS % Ns) / Ns;
	v0 = v[2] * cos(angle) - v[3] * sin(angle);
	v1 = v[2] * sin(angle) + v[3] * cos(angle);
	v[2] = v[0] - v0;
	v[3] = v[1] - v1;
	v[0] = v[0] + v0;
	v[1] = v[1] + v1;
	
	idxD = (idxS / Ns) * (Ns<<1) + (idxS % Ns);
	data1[idxD] = v[0];
	data1[idxD + N] = v[1];
	data1[idxD + Ns] = v[2];
	data1[idxD + Ns + N] = v[3];
}
