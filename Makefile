objs =	CLEnv.o \
		gpuFFTLocal.o verifyLocal.o lengthFlopsLocal.o \
		gpuFFTBase.o verifyBase.o lengthFlopsBase.o \
		gpuFFTFour.o verifyFour.o lengthFlopsFour.o \
		gpuFFTOct.o verifyOct.o lengthFlopsOct.o
exes =	verifyLocal lengthFlopsLocal \
		verifyBase lengthFlopsBase \
		verifyFour lengthFlopsFour \
		verifyOct lengthFlopsOct

all:					$(exes)		
								
CLEnv.o:				CLEnv.c
						gcc -c -g CLEnv.c
					
gpuFFTLocal.o:			gpuFFTLocal.c
						gcc -c gpuFFTLocal.c
verifyLocal:			verifyLocal.o CLEnv.o gpuFFTLocal.o
						gcc verifyLocal.o CLEnv.o gpuFFTLocal.o -lOpenCL -lfftw3 -lm -o verifyLocal
verifyLocal.o:			verifyLocal.c
						gcc -c verifyLocal.c
lengthFlopsLocal:		lengthFlopsLocal.o CLEnv.o gpuFFTLocal.o
						gcc lengthFlopsLocal.o CLEnv.o gpuFFTLocal.o -lOpenCL -o lengthFlopsLocal
lengthFlopsLocal.o:		lengthFlopsLocal.c
						gcc -c lengthFlopsLocal.c

gpuFFTBase.o:			gpuFFTBase.c
						gcc -c gpuFFTBase.c
verifyBase:				verifyBase.o CLEnv.o gpuFFTBase.o
						gcc verifyBase.o CLEnv.o gpuFFTBase.o -lOpenCL -lfftw3 -lm -o verifyBase
verifyBase.o:			verifyBase.c
						gcc -c verifyBase.c
lengthFlopsBase:		lengthFlopsBase.o CLEnv.o gpuFFTLocal.o
						gcc lengthFlopsBase.o CLEnv.o gpuFFTBase.o -lOpenCL -o lengthFlopsBase
lengthFlopsBase.o:		lengthFlopsBase.c
						gcc -c lengthFlopsBase.c

gpuFFTFour.o:			gpuFFTFour.c
						gcc -c -g gpuFFTFour.c
verifyFour:				verifyFour.o CLEnv.o gpuFFTFour.o
						gcc verifyFour.o CLEnv.o gpuFFTFour.o -lOpenCL -lfftw3 -lm -o verifyFour
verifyFour.o:			verifyFour.c
						gcc -c -g verifyFour.c
lengthFlopsFour:		lengthFlopsFour.o CLEnv.o gpuFFTFour.o
						gcc lengthFlopsFour.o CLEnv.o gpuFFTFour.o -lOpenCL -o lengthFlopsFour
lengthFlopsFour.o:		lengthFlopsFour.c
						gcc -c -g lengthFlopsFour.c

gpuFFTOct.o:			gpuFFTOct.c
						gcc -c -g gpuFFTOct.c
verifyOct:				verifyOct.o CLEnv.o gpuFFTOct.o
						gcc verifyOct.o CLEnv.o gpuFFTOct.o -lOpenCL -lfftw3 -lm -o verifyOct
verifyOct.o:			verifyOct.c
						gcc -c -g verifyOct.c
lengthFlopsOct:			lengthFlopsOct.o CLEnv.o gpuFFTOct.o
						gcc lengthFlopsOct.o CLEnv.o gpuFFTOct.o -lOpenCL -o lengthFlopsOct
lengthFlopsOct.o:		lengthFlopsOct.c
						gcc -c -g lengthFlopsOct.c


	
clean:
						rm $(objs) $(exes)
