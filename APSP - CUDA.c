#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <time.h>



int n;		//num of rows and collumns
double p;	//probability
int inf=100;	//-1 is the value for infinity
int w=30;	//all weights will value from 0 to 30;



__global__ void kernel_1(int * array, int size, int k){
	int ix=blockIdx.x * blockDim.x + threadIdx.x;
	int jx=blockIdx.y * blockDim.y + threadIdx.y;

	if (array[size*ix+jx] > (array[size*ix+k] + array[size*k+jx])){
			array[size*ix+jx] = array[size*ix+k] + array[size*k+jx];
		}
}

__global__ void kernel_2(int * array, int size, int k){
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=size)return;
	int idx=size*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=array[size*blockIdx.y+k];
	__syncthreads();
	if(best==100)return;
	int tmp_b=array[k*size+col];
	if(tmp_b==100)return;
	int cur=best+tmp_b;
	if(cur<array[idx]){
		array[idx]=cur;
	}
}

__global__ void kernel_3(int * array, int size, int k){
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=size)return;
	int idx=size*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=array[size*blockIdx.y+k];
	__syncthreads();
	if(best==100)return;
	int tmp_b=array[k*size+col];
	if(tmp_b==100)return;
	int cur=best+tmp_b;
	if(cur<array[idx]){
		array[idx]=cur;
	}
	col=col+size/2;
	__syncthreads();
	if(col>=size)return;
	idx=size*blockIdx.y+col;
	if(best==100)return;
	tmp_b=array[k*size+col];
	if(tmp_b==100)return;
	cur=best+tmp_b;
	if(cur<array[idx]){
		array[idx]=cur;
	}
}


int main(int argc, char **argv) {
 
	if (argc != 3) {
	printf("Usage: %s q\n  where n=2^q is problem size (power of two)\n", 
	   argv[0]);
	exit(1);
	}

	n = 1<<atoi(argv[1]);
	p = atof(argv[2])/100;


	time_t t;
	int i,j,k;
	//Create arrays for filling.
	int *adjArray;
	adjArray=(int*)malloc(n*n*sizeof(int));
	int *adjArraySerial;
	adjArraySerial=(int*)malloc(n*n*sizeof(int));
	int *adjArrayKernel2;
	adjArrayKernel2=(int*)malloc(n*n*sizeof(int));
	int *adjArrayKernel3;
	adjArrayKernel3=(int*)malloc(n*n*sizeof(int));

	struct timeval startwtime, endwtime;

	srand(time(&t));

	//Initializing the adjency array using C. (The reason of using C, is told in the reference)
	for (i=0; i<n; i++){
		for (j=0; j<n; j++){
			if (((double)rand()/(double)RAND_MAX)>p){
				adjArray[n*i+j]=inf;
			}
			else{
				adjArray[n*i+j]=(int)(((double)rand()/(double)RAND_MAX)*w);
			}
		}
	}
	//Fill the primary diagonus with inf.
	int temp=0;
	for (i=0; i<n; i++){
		adjArray[i*n + temp]=inf;
		temp++;
	}

	//Initializing each array for each kernel according to the adjArray. The 1st kernel is using the adjArray itself.
	for (i=0; i<n*n; i++){
		adjArraySerial[i]=adjArray[i];
		adjArrayKernel2[i]=adjArray[i];
		adjArrayKernel3[i]=adjArray[i];

	}
	

	//=================================================SERIAL===============================================//

	gettimeofday(&startwtime,NULL);
	for (k=0; k<n; k++){
		for (i=0; i<n; i++){
			for (j=0; j<n; j++){
				if (adjArraySerial[n*i+j] > (adjArraySerial[n*i+k] + adjArraySerial[n*k+j])){
					adjArraySerial[n*i+j] = adjArraySerial[n*i+k] + adjArraySerial[n*k+j];
				}
			}
		}
	}
	gettimeofday(&endwtime,NULL);
	printf("Time of Serial Algorithm: %lf\n",(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));
	////=======================================================================================================//

	//dev_a is for Device use. dev_b is for Host use.
	int* dev_a;
	int* dev_b;
	cudaMalloc((void **)&dev_a, n*n*sizeof(int));
	dev_b=(int*)malloc(n*n*sizeof(int));
	

	//=================================================KERNEL 1===============================================//
	gettimeofday(&startwtime,NULL);
	cudaMemcpy(dev_a,adjArray,n*n*sizeof(int),cudaMemcpyHostToDevice);
	k=0;

	int threads = 32;	
	dim3 dimBlock(threads,threads);
	dim3 dimGrid((n)/dimBlock.x,(n)/dimBlock.y);
	for(k=0; k<n; k++){
		kernel_1<<< dimGrid, dimBlock >>>(dev_a,n,k);
		cudaThreadSynchronize();
	}

	cudaMemcpy(dev_b, dev_a, n*n*sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday(&endwtime,NULL);
	printf("Time of Kernel 1: %lf\n",(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));
	//Comparison.
	for (i=0; i<n; i++){
		for(j=0; j<n; j++){
			if (dev_b[n*i+j]!=adjArraySerial[n*i+j]){
				printf("Fault in Kernel 1! Exiting with code 1.\n");
				exit(1);
			}
		}
	}
	printf("SUCCESIVE comparison between Kernel 3 and Serial Algorithm.\n");

	int BLOCK_SIZE=256;
	dim3 dimGrid1((n+BLOCK_SIZE-1)/BLOCK_SIZE,n);

	//=================================================KERNEL 2===============================================//

	gettimeofday(&startwtime,NULL);
	cudaMemcpy(dev_a,adjArrayKernel2,n*n*sizeof(int),cudaMemcpyHostToDevice);
	for(k=0; k<n; k++){
		kernel_2<<< dimGrid1, BLOCK_SIZE >>>(dev_a,n,k);
		cudaThreadSynchronize();
	}

	cudaMemcpy(dev_b, dev_a, n*n*sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday(&endwtime,NULL);
	printf("Time of Kernel 2: %lf\n",(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));
	//Comparison.
	for (i=0; i<n; i++){
		for(j=0; j<n; j++){
			if (dev_b[n*i+j]!=adjArraySerial[n*i+j]){
				printf("Fault in Kernel 2! Exiting with code 2.\n");
				exit(2);
			}
		}
	}
	printf("SUCCESIVE comparison between Kernel 2 and Serial Algorithm.\n");


	dim3 dimGrid2(n,n);

	//=================================================KERNEL 3===============================================//
	gettimeofday(&startwtime,NULL);

	cudaMemcpy(dev_a,adjArrayKernel3,n*n*sizeof(int),cudaMemcpyHostToDevice);

	for(k=0; k<n; k++){
		kernel_3<<< dimGrid1, BLOCK_SIZE/2 >>>(dev_a,n,k);
		cudaThreadSynchronize();
	}

	cudaMemcpy(dev_b, dev_a, n*n*sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday(&endwtime,NULL);

	printf("Time of Kernel 3: %lf\n",(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));
	//Comparison.
	for (i=0; i<n; i++){
		for(j=0; j<n; j++){
			if (dev_b[n*i+j]!=adjArraySerial[n*i+j]){
				printf("Fault in Kernel 3! Exiting with code 3.\n");
				exit(3);
			}

		}
	}
	printf("SUCCESIVE comparison between Kernel 3 and Serial Algorithm.\n");

	//Free memory
	cudaFree(dev_a);
	free(dev_b);
	

	return 0;
}