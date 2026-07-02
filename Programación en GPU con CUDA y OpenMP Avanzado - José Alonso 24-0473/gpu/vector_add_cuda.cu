/*
 * vector_add_cuda.cu
 * ---------------------------------------------------------
 * Fase 2: Suma de vectores en GPU usando CUDA.
 *
 * Universidad Iberoamericana (UNIBE)
 * Estudiante : José Armando Alonso Polanco
 * Matrícula  : 24-0473
 * Materia    : Programación Distribuida
 * ---------------------------------------------------------
 * Compilación:
 *   nvcc -O2 vector_add_cuda.cu -o vector_add_cuda
 *
 * Ejecución:
 *   ./vector_add_cuda
 *
 * Requiere una GPU NVIDIA con drivers y CUDA Toolkit instalados
 * (por ejemplo, Google Colab con entorno de ejecución GPU T4).
 * ---------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1048576)          /* 1,048,576 = 1 M elementos       */
#define HILOS_POR_BLOQUE 256 /* tamano de bloque recomendado    */

/* -------------------------------------------------------------
 * Macro de verificacion de errores CUDA. Envuelve cada llamada
 * a la API de CUDA para detener el programa de inmediato si algo
 * falla (memoria insuficiente, transferencia invalida, etc.)
 * en lugar de continuar con datos corruptos.
 * ------------------------------------------------------------- */
#define CUDA_CHECK(llamada)                                              \
    do {                                                                 \
        cudaError_t err = (llamada);                                     \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "Error CUDA en %s:%d -> %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* -------------------------------------------------------------
 * Kernel: cada hilo calcula UN solo elemento de C.
 * El indice global se calcula combinando el indice de bloque,
 * el tamano de bloque y el indice de hilo dentro del bloque.
 * Se valida "idx < n" porque el numero total de hilos lanzados
 * (bloques * hilos_por_bloque) puede ser mayor que N.
 * ------------------------------------------------------------- */
__global__ void add_vectors(const double *A, const double *B, double *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    size_t bytes = N * sizeof(double);

    /* -------------------------------------------------------------
     * 1. Reserva de memoria en el HOST (CPU).
     * ------------------------------------------------------------- */
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Error: no se pudo reservar memoria en host.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = (double)i * 0.5;
        h_B[i] = (double)i * 1.5;
    }

    /* -------------------------------------------------------------
     * 2. Eventos CUDA para medir tiempo con precisión en la GPU.
     *    Se usan cudaEvent_t en lugar de un reloj de CPU porque
     *    permiten medir con exactitud lo que ocurre en el device,
     *    incluyendo la sincronización real del kernel.
     * ------------------------------------------------------------- */
    cudaEvent_t inicio_total, fin_total;
    CUDA_CHECK(cudaEventCreate(&inicio_total));
    CUDA_CHECK(cudaEventCreate(&fin_total));

    CUDA_CHECK(cudaEventRecord(inicio_total));

    /* -------------------------------------------------------------
     * 3. Reserva de memoria en el DEVICE (GPU) con cudaMalloc.
     * ------------------------------------------------------------- */
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* -------------------------------------------------------------
     * 4. Transferencia Host -> Device (Host to Device, H2D).
     * ------------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* -------------------------------------------------------------
     * 5. Configuracion de la malla de ejecucion.
     *    256 hilos por bloque (multiplo del tamano de warp = 32).
     *    El numero de bloques se redondea hacia arriba para cubrir
     *    todos los N elementos, incluso si N no es multiplo de 256.
     * ------------------------------------------------------------- */
    int bloques = (N + HILOS_POR_BLOQUE - 1) / HILOS_POR_BLOQUE;

    add_vectors<<<bloques, HILOS_POR_BLOQUE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());      /* error de lanzamiento del kernel */
    CUDA_CHECK(cudaDeviceSynchronize()); /* esperar a que el kernel termine */

    /* -------------------------------------------------------------
     * 6. Transferencia Device -> Host (Device to Host, D2H).
     * ------------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(fin_total));
    CUDA_CHECK(cudaEventSynchronize(fin_total));

    float tiempo_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_gpu_ms, inicio_total, fin_total));
    double tiempo_gpu_s = tiempo_gpu_ms / 1000.0;

    /* -------------------------------------------------------------
     * 7. Verificacion de resultados contra el calculo esperado.
     * ------------------------------------------------------------- */
    int errores = 0;
    for (int i = 0; i < N; i++) {
        double esperado = h_A[i] + h_B[i];
        if (h_C[i] != esperado) {
            errores++;
        }
    }

    /* -------------------------------------------------------------
     * 8. Reporte de resultados.
     * ------------------------------------------------------------- */
    printf("======================================================\n");
    printf(" Suma de vectores - Version GPU (CUDA)\n");
    printf("======================================================\n");
    printf(" Tamano del vector (N)        : %d elementos\n", N);
    printf(" Hilos por bloque             : %d\n", HILOS_POR_BLOQUE);
    printf(" Numero de bloques            : %d\n", bloques);
    printf(" Tiempo total GPU (H2D+kernel+D2H): %f segundos\n", tiempo_gpu_s);
    printf(" Verificacion                 : %s (%d errores)\n",
           errores == 0 ? "CORRECTA" : "FALLIDA", errores);
    printf(" Muestra C[0], C[N/2], C[N-1] : %.2f, %.2f, %.2f\n",
           h_C[0], h_C[N / 2], h_C[N - 1]);
    printf("======================================================\n");

    FILE *f = fopen("tiempo_gpu.txt", "w");
    if (f != NULL) {
        fprintf(f, "%f\n", tiempo_gpu_s);
        fclose(f);
    }

    /* -------------------------------------------------------------
     * 9. Liberación de memoria (device y host) y destrucción
     *    de los eventos CUDA.
     * ------------------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(inicio_total));
    CUDA_CHECK(cudaEventDestroy(fin_total));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
