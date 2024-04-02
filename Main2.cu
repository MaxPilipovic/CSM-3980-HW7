#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void matrixmultiply_kernel(float* A, float* B, float* C, unsigned int N) {
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for(unsigned int i = 0; i < N; ++i) {
            sum += A[row*N + i]*B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}

void matrixMultiply(float* a, float* b, float* c, int N) {

    //Allocate GPU memory
    float *a_d, *b_d, *c_d;

    cudaMalloc((void**) &a_d, N*N*sizeof(float));
    cudaMalloc((void**) &b_d, N*N*sizeof(float));
    cudaMalloc((void**) &c_d, N*N*sizeof(float));

    //Copy data to GPU memory
    cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

    //Start time
    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    //Perform computation on GPU
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    matrixmultiply_kernel<<<numBlocks, numThreadsPerBlock>>>(a_d, b_d, c_d, N);

    //Synchronize
    cudaDeviceSynchronize();

    //End time
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time to generate:  %3.1f ms \n", time);

    //Copy data from GPU memory
    cudaMemcpy(c, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    //Deallocate GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void readMultiply(char* filename, float **matrix, int *N) {
    FILE* read = fopen(filename, "r");
    if(read != NULL) {
        int row;
        int column;
        fscanf(read, "%d %d", &row, &column);
        //Ask about this
        *N = row;

        *matrix = (float*)malloc(row * column * sizeof(float));

        for (int i = 0; i < row * column; i++) {
            fscanf(read, "%f", &((*matrix)[i]));
        }
        fclose(read);
    } else {
        printf("Nothing in file");
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("3 Arguments are required");
        return 1;
    }

    float *A;
    float *B;
    float *C;
    //int rowA;
    //int columnA;
    //int rowB;
    //int columnB;
    int N;

    //READ A
    //wtf?
    readMultiply(argv[1], &A, &N);

    //READ B
    readMultiply(argv[2], &B, &N);

    //ALLOCATE MEMORY
    C = (float*)malloc(N*N*sizeof(float));

    //PERFORM MATRIX MULTIPLICATION
    matrixMultiply(A, B, C, N);

    //WRITE TO MATRIX C
    FILE* write = fopen(argv[3], "w");
    if(write != NULL) {
        fprintf(write, "%d %d\n", N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(write, "%f ", C[i * N + j]);
            }
            fprintf(write, "\n");
        }
        fclose(write);
    } else {
        printf("Nothing in file");
        return 1;
    }

    //FREE ALLOCATED MEMORY

    free(A);
    free(B);
    free(C);

    return 0;
}