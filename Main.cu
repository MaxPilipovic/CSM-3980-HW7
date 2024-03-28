#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void readMultiply(char* filename, float ***matrix, int *row, int *column) {
    FILE* read = fopen(filename, "r");
    if(read != NULL) {
        fscanf(read, "%d %d", row, column);
        //Ask about this
        *matrix = (float**)malloc(*row * sizeof(float*));
        for (int i = 0; i < *row; i++) {
            (*matrix)[i] = (float*)malloc(*column * sizeof(float));
            for (int j = 0; j < *column; j++) {
                fscanf(read, "%f", &((*matrix)[i][j]));
            }
        }
        fclose(read);
    } else {
        printf("Nothing in file");
        exit(1);
    }
}

void matrixMultiply(float **A, float **B, float **C, int rowA, int columnA, int rowB, int columnB) {
    if (columnA != rowB) {
        printf("Dimensions not going to work");
        exit(1);
    }

    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < columnB; j++) {
            C[i][j] = 0; //start with 0
            for (int n = 0; n < rowB; n++) {
                C[i][j] += A[i][n] * B[n][j];
            }
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("3 Arguments are required");
        return 1;
    }

    float **A;
    float **B;
    float **C;
    int rowA;
    int columnA;
    int rowB;
    int columnB;

    //READ A
    //wtf?
    readMultiply(argv[1], &A, &rowA, &columnA);

    //READ B
    readMultiply(argv[2], &B, &rowB, &columnB);

    //ALLOCATE MEMORY
    C = (float**)malloc(rowA * sizeof(float*));
    for (int z = 0; z < rowA; z++) {
        C[z] = (float*)malloc(columnB * sizeof(float));
    }

    //PERFORM MATRIX MULTIPLICATION
    matrixMultiply(A, B, C, rowA, columnA, rowB, columnB);

    //WRITE TO MATRIX C
    FILE* write = fopen(argv[3], "w");
    if(write != NULL) {
        fprintf(write, "%d %d\n", rowA, columnB);
        for (int i = 0; i < rowA; i++) {
            for (int j = 0; j < columnB; j++) {
                fprintf(write, "%f ", C[i][j]);
            }
            fprintf(write, "\n");
        }
        fclose(write);
    } else {
        printf("Nothing in file");
        return 1;
    }
    
    //FREE ALLOCATED MEMORY
    for (int q = 0; q < rowA; q++) {
        free(A[q]);
    }
    free(A);

    for (int x = 0; x < rowB; x++) {
        free(B[x]);
    }
    free(B);

    for (int e = 0; e < rowA; e++) {
        free(C[e]);
    }
    free(C);
}