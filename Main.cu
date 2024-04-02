#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void matrixMultiply(float *A, float *B, float *C, unsigned int N) {
    for (unsigned int row = 0; row < N; row++) {
        for (unsigned int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
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
}