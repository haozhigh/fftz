__kernel void gpuFFTFour(	__global float* data0, __global float* data1, int N, int Ns){
	int idx;
	float c1, s1, c2, s2, c3, s3;
	float vo[8];
	float vi0, vi1;
	float rcis, rsic;
	
	idx = get_global_id(0);
	vi0 = data0[idx];
	vi1 = data0[idx + N];
	vo[0] = vi0;
	vo[1] = vi1;
	vo[2] = vi0;
	vo[3] = vi1;
	vo[4] = vi0;
	vo[5] = vi1;
	vo[6] = vi0;
	vo[7] = vi1;

	vi0 = data0[idx + N / 4];
	vi1 = data0[idx + N / 4 + N];
	s1 = -M_PI_F * (idx % Ns) / (Ns << 1);
	c1 = cos(s1);
	s1 = sin(s1);
	rcis = vi0 * c1 - vi1 * s1;
	rsic = vi0 * s1 + vi1 * c1;
	vo[0] = vo[0] + rcis;
	vo[1] = vo[1] + rsic;
	vo[2] = vo[2] + rsic;
	vo[3] = vo[3] - rcis;
	vo[4] = vo[4] - rcis;
	vo[5] = vo[5] - rsic;
	vo[6] = vo[6] - rsic;
	vo[7] = vo[7] + rcis;

	vi0 = data0[idx + N / 2];
	vi1 = data0[idx + N / 2 + N];
	c2 = c1 * c1 * 2 - 1;
	s2 = s1 * c1 * 2;
	rcis = vi0 * c2 - vi1 * s2;
	rsic = vi0 * s2 + vi1 * c2;
	vo[0] = vo[0] + rcis;
	vo[1] = vo[1] + rsic;
	vo[2] = vo[2] - rcis;
	vo[3] = vo[3] - rsic;
	vo[4] = vo[4] + rcis;
	vo[5] = vo[5] + rsic;
	vo[6] = vo[6] - rcis;
	vo[7] = vo[7] - rsic;

	vi0 = data0[idx + N / 4 * 3];
	vi1 = data0[idx + N / 4 * 3 + N];
	c3 = c1 * c2 - s1 * s2;
	s3 = s1 * c2 + c1 * s2;
	rcis = vi0 * c3 - vi1 * s3;
	rsic = vi0 * s3 + vi1 * c3;
	vo[0] = vo[0] + rcis;
	vo[1] = vo[1] + rsic;
	vo[6] = vo[6] + rsic;
	vo[7] = vo[7] - rcis;
	vo[4] = vo[4] - rcis;
	vo[5] = vo[5] - rsic;
	vo[2] = vo[2] - rsic;
	vo[3] = vo[3] + rcis;

	idx = (idx / Ns) * (Ns << 2) + (idx % Ns);
	data1[idx] = vo[0];
	data1[idx + N] = vo[1];
	data1[idx + Ns] = vo[2];
	data1[idx + Ns + N] = vo[3];
	data1[idx + Ns * 2] = vo[4];
	data1[idx + Ns * 2 + N] = vo[5];
	data1[idx + Ns * 3] = vo[6];
	data1[idx + Ns * 3 + N] = vo[7];
}
