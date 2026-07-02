/*
 * vector_add_omp.c
 * ---------------------------------------------------------
 * Fase 1: Suma de vectores en CPU paralelizada con OpenMP.
 *
 * Universidad Iberoamericana (UNIBE)
 * Estudiante : José Armando Alonso Polanco
 * Matrícula  : 24-0473
 * Materia    : Programación Distribuida
 * ---------------------------------------------------------
 * Compilación:
 *   gcc -O2 -fopenmp vector_add_omp.c -o vector_add_omp
 *
 * Ejecución:
 *   ./vector_add_omp              (usa todos los hilos disponibles)
 *   OMP_NUM_THREADS=4 ./vector_add_omp   (fija el número de hilos)
 * ---------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1048576) /* 1,048,576 = 1 M elementos */

int main(void) {
    /* -------------------------------------------------------------
     * 1. Reserva de memoria para los tres vectores (A, B y C).
     *    Se usa malloc en lugar de arreglos estáticos porque 1 M de
     *    doubles (8 MB por vector) es demasiado para la pila (stack).
     * ------------------------------------------------------------- */
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: no se pudo reservar memoria.\n");
        return 1;
    }

    /* -------------------------------------------------------------
     * 2. Inicialización de los vectores de entrada.
     *    También se paraleliza porque, con N grande, inicializar en
     *    serie ya toma un tiempo no despreciable.
     * ------------------------------------------------------------- */
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        A[i] = (double)i * 0.5;
        B[i] = (double)i * 1.5;
    }

    /* -------------------------------------------------------------
     * 3. Suma paralela: C[i] = A[i] + B[i]
     *    omp_get_wtime() da un reloj de pared (wall-clock) de alta
     *    resolución, adecuado para medir tiempo real transcurrido
     *    (a diferencia de clock(), que mide tiempo de CPU).
     * ------------------------------------------------------------- */
    double t_inicio = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    double t_fin = omp_get_wtime();
    double tiempo_omp = t_fin - t_inicio;

    /* -------------------------------------------------------------
     * 4. Verificación de correctitud: se revisan todos los
     *    elementos y se reporta si hubo algún error.
     * ------------------------------------------------------------- */
    int errores = 0;
    for (int i = 0; i < N; i++) {
        double esperado = A[i] + B[i];
        if (C[i] != esperado) {
            errores++;
        }
    }

    /* -------------------------------------------------------------
     * 5. Reporte de resultados.
     * ------------------------------------------------------------- */
    printf("======================================================\n");
    printf(" Suma de vectores - Version CPU (OpenMP)\n");
    printf("======================================================\n");
    printf(" Tamano del vector (N)      : %d elementos\n", N);
    printf(" Hilos disponibles (max)    : %d\n", omp_get_max_threads());
    printf(" Tiempo de ejecucion (suma) : %f segundos\n", tiempo_omp);
    printf(" Verificacion               : %s (%d errores)\n",
           errores == 0 ? "CORRECTA" : "FALLIDA", errores);
    printf(" Muestra C[0], C[N/2], C[N-1]: %.2f, %.2f, %.2f\n",
           C[0], C[N / 2], C[N - 1]);
    printf("======================================================\n");

    /* -------------------------------------------------------------
     * 6. Guardar el tiempo en un archivo de texto para poder
     *    reutilizarlo luego en el calculo del speedup.
     * ------------------------------------------------------------- */
    FILE *f = fopen("tiempo_omp.txt", "w");
    if (f != NULL) {
        fprintf(f, "%f\n", tiempo_omp);
        fclose(f);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
