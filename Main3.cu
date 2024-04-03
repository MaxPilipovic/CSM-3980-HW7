#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TILE_DIM 16
#define COARSE_FACTOR 3

__global__ void tiled_matrixmultiply_kernel(float* A, float* B, float* C, unsigned int N, unsigned int M, unsigned int K) {
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    float sum[COARSE_FACTOR];
    for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }
    for (unsigned int tile = 0; tile < N/TILE_DIM; tile++) {
        //Load A tile
        A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
            unsigned int col = colStart + c*TILE_DIM;
            //Load B tile
            B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
            __syncthreads();
            //Compute with tile
            for (unsigned int i = 0; i < TILE_DIM; i++) {
                sum[c] += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for (unsigned int c = 0; c < COARSE_FACTOR; c++) {
        unsigned int col = colStart + c*TILE_DIM;
        C[row*N + col] = sum[c];
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Perform computation on GPU
    int M = N;
    int K = N;
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((N + TILE_DIM - 1)/TILE_DIM/COARSE_FACTOR, (N + TILE_DIM - 1)/TILE_DIM);
    tiled_matrixmultiply_kernel<<<numBlocks, numThreadsPerBlock>>>(a_d, b_d, c_d, N, M, K);

    //Synchronize
    cudaDeviceSynchronize();

    //End time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);

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
