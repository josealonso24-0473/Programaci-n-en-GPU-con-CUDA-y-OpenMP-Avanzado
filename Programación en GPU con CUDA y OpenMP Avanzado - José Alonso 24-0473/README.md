# Suma de Vectores Híbrida: OpenMP (CPU) + CUDA (GPU)

Proyecto de Programación Distribuida — José Armando Alonso Polanco (24-0473), UNIBE.

## Estructura

```
cpu/vector_add_omp.c       -> Versión CPU paralela con OpenMP
gpu/vector_add_cuda.cu     -> Versión GPU con CUDA
gpu/Colab_CUDA_FinanZen.ipynb -> Notebook para correr la versión CUDA en Google Colab (GPU gratis)
Informe_CUDA_OpenMP_Alonso_240473.docx -> Informe completo
Script_Video_Presentacion.md -> Guion para el video de presentación
```

## Cómo compilar y correr

### CPU (OpenMP) — cualquier máquina con gcc
```bash
cd cpu
gcc -O2 -fopenmp vector_add_omp.c -o vector_add_omp
OMP_NUM_THREADS=8 ./vector_add_omp     # ajusta al número de núcleos de tu máquina
```

### GPU (CUDA) — requiere GPU NVIDIA
```bash
cd gpu
nvcc -O2 vector_add_cuda.cu -o vector_add_cuda
./vector_add_cuda
```

Si no tienes GPU local, abre `Colab_CUDA_FinanZen.ipynb` en Google Colab, activa el
entorno de ejecución GPU (T4) y ejecuta las celdas en orden.

## Resultados

Cada programa escribe su tiempo medido en `tiempo_omp.txt` / `tiempo_gpu.txt`.
El speedup se calcula como `Tiempo_CPU / Tiempo_GPU` (ver informe, sección 6).
