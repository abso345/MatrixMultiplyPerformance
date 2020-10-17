/*
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
CC = cc
OPT = -Ofast -mavx2 -mfma -funroll-loops -msse2
CFLAGS = -Wall -std=gnu99 $(OPT)
LDLIBS = -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -L/usr/local/lib -lblas -lpthread -lm -ldl
*/

#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>

const char* dgemm_desc = "Blocked dgemm. avx2, copy optimization A";

#define likely(x) __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#define min(a,b) (((a)<(b))?(a):(b))

typedef int bool;
#define true 1
#define false 0

#define MAX_ARRAY_SIZE 128

int array[MAX_ARRAY_SIZE][3] = { {32, 16, 36}, {64, 32, 40}, {96, 100, 96}, {128, 26, 16}, {160, 56, 88}, {192, 32, 64}, {224, 58, 76}, {256, 12, 32}, {288, 48, 98}, {320, 36, 66}, {352, 60, 90}, {384, 24, 64}, {416, 60, 108}, {448, 46, 90}, {480, 70, 102}, {512, 34, 46}, {544, 70, 110}, {576, 42, 98}, {608, 44, 102}, {640, 24, 64}, {672, 48, 100}, {704, 44, 64}, {736, 62, 108}, {768, 12, 50}};

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
inline void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
  {
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij0 = C[i+0+(j)*lda];

      for (int k = 0; k < K; ++k)
      {
        cij0 += A[i+0+k*lda] * B[k+j*lda];
      }

      C[i+0+j*lda] = cij0;
    }
  }
}

inline void do_block_staticavx16 (int lda, int M, int N, int I, double* restrict A, double* restrict B, double* restrict C, bool prefetchb)
{
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6,  ymm7, ymm8, ymm9, ymm10, ymm11, ymm14, ymm16, ymm17;

  // load C0-C7 into two 256 bit registers
  for (int row = 0; row < I; row+=2)
  {
    int rows = min (2, I - row);

    ymm0 = _mm256_loadu_pd(&C[0 + row*lda]);
    ymm1 = _mm256_loadu_pd(&C[4 + row*lda]);
    ymm7 = _mm256_loadu_pd(&C[8 + row*lda]);
    ymm8 = _mm256_loadu_pd(&C[12 + row*lda]);

    if (likely(rows == 2))
    {
      ymm11 = _mm256_loadu_pd(&C[0 + (row+1)*lda]);
      ymm14 = _mm256_loadu_pd(&C[4 + (row+1)*lda]);
      ymm16 = _mm256_loadu_pd(&C[8 + (row+1)*lda]);
      ymm17 = _mm256_loadu_pd(&C[12 + (row+1)*lda]);
    }

    for (int i = 0; i < N; i++)
    {
      // Load B and broadcast to two 256 bit registers.
      ymm2 = _mm256_broadcast_sd(&B[i + row*lda]);
      ymm6 = ymm2;

      // load A0-A7 into two registers.
      // multiply B by A0-A3 and A4-A7 and accumulate in C.
      ymm3 = _mm256_loadu_pd(&A[i*16]);
      ymm4 = _mm256_loadu_pd(&A[4+i*16]);
      ymm9 = _mm256_loadu_pd(&A[8+i*16]);
      ymm10 = _mm256_loadu_pd(&A[12+i*16]);

      ymm0 = _mm256_fmadd_pd(ymm3, ymm2, ymm0);
      ymm1 = _mm256_fmadd_pd(ymm4, ymm6, ymm1);
      ymm7 = _mm256_fmadd_pd(ymm9, ymm2, ymm7);
      ymm8 = _mm256_fmadd_pd(ymm10, ymm6, ymm8);

      if (likely(rows == 2))
      {
        ymm2 = _mm256_broadcast_sd(&B[i + (row+1)*lda]);
        ymm6 = ymm2;

        ymm11 = _mm256_fmadd_pd(ymm3, ymm2, ymm11);
        ymm14 = _mm256_fmadd_pd(ymm4, ymm6, ymm14);
        ymm16 = _mm256_fmadd_pd(ymm9, ymm2, ymm16);
        ymm17 = _mm256_fmadd_pd(ymm10, ymm6, ymm17);
      }
    }

    // Stores C registers
    _mm256_storeu_pd(&C[0 + row*lda], ymm0);
    _mm256_storeu_pd(&C[4 + row*lda], ymm1);
    _mm256_storeu_pd(&C[8 + row*lda], ymm7);
    _mm256_storeu_pd(&C[12 + row*lda], ymm8);

    if (likely(rows == 2))
    {
      _mm256_storeu_pd(&C[0 + (row+1)*lda], ymm11);
      _mm256_storeu_pd(&C[4 + (row+1)*lda], ymm14);
      _mm256_storeu_pd(&C[8 + (row+1)*lda], ymm16);
      _mm256_storeu_pd(&C[12 + (row+1)*lda], ymm17);
    }
  }
}

inline void do_block_staticavx8 (int lda, int M, int N, int I, double* restrict A, double* restrict B, double* restrict C, bool prefetchb)
{
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13;

  int rows = 4;

  // load C0-C7 into two 256 bit registers
  for (int row = 0; row < I; row+=rows)
  {
    rows = min (4, I - row);

    if (rows < 4)
      rows = 1;

    ymm0 = _mm256_loadu_pd(&C[0 + row*lda]);
    ymm1 = _mm256_loadu_pd(&C[4 + row*lda]);

    if (likely(rows == 4))
    {
        ymm5 = _mm256_loadu_pd(&C[0 + (row+1)*lda]);
        ymm6 = _mm256_loadu_pd(&C[4 + (row+1)*lda]);
        ymm8 = _mm256_loadu_pd(&C[0 + (row+2)*lda]);
        ymm9 = _mm256_loadu_pd(&C[4 + (row+2)*lda]);
        ymm10 = _mm256_loadu_pd(&C[0 + (row+3)*lda]);
        ymm11 = _mm256_loadu_pd(&C[4 + (row+3)*lda]);
    }

    for (int i = 0; i < N; i++)
    {
      // Load B and broadcast to two 256 bit registers.
      ymm2 = _mm256_broadcast_sd(&B[i + row*lda]);

      // load A0-A7 into two registers.
      // multiply B by A0-A3 and A4-A7 and accumulate in C.
      ymm3 = _mm256_loadu_pd(&A[i*lda]);
      ymm4 = _mm256_loadu_pd(&A[4+i*lda]);

      ymm0 = _mm256_fmadd_pd(ymm3, ymm2, ymm0);
      ymm1 = _mm256_fmadd_pd(ymm4, ymm2, ymm1);

      if (likely(rows == 4))
      {
        ymm7 = _mm256_broadcast_sd(&B[i + (row+1)*lda]);
        ymm12 = _mm256_broadcast_sd(&B[i + (row+2)*lda]);
        ymm13 = _mm256_broadcast_sd(&B[i + (row+3)*lda]);

        ymm5 = _mm256_fmadd_pd(ymm3, ymm7, ymm5);
        ymm9 = _mm256_fmadd_pd(ymm4, ymm12, ymm9);
        ymm6 = _mm256_fmadd_pd(ymm4, ymm7, ymm6);
        ymm10 = _mm256_fmadd_pd(ymm3, ymm13, ymm10);
        ymm8 = _mm256_fmadd_pd(ymm3, ymm12, ymm8);
        ymm11 = _mm256_fmadd_pd(ymm4, ymm13, ymm11);
      }
    }

    // Stores C registers
    _mm256_storeu_pd(&C[0 + row*lda], ymm0);
    _mm256_storeu_pd(&C[4 + row*lda], ymm1);

    if (likely(rows == 4))
    {
        _mm256_storeu_pd(&C[0 + (row+1)*lda], ymm5);
        _mm256_storeu_pd(&C[4 + (row+1)*lda], ymm6);
        _mm256_storeu_pd(&C[0 + (row+2)*lda], ymm8);
        _mm256_storeu_pd(&C[4 + (row+2)*lda], ymm9);
        _mm256_storeu_pd(&C[0 + (row+3)*lda], ymm10);
        _mm256_storeu_pd(&C[4 + (row+3)*lda], ymm11);
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their inpuvxt values. */
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
  int jmax = 32;
  int kmax = lda;
  int imax = lda;

  double* altA = malloc((lda)*(lda)*sizeof(double));

  int count = 0;
  int max = 16;

  for (int l = 0; l < lda; l+=max)
  {
    max = min(16, lda - l);

    for (int i = 0; i < lda; i++)
    {
      for (int j = 0; j < max; j++)
      {
        altA[count] = A[l+j+i*lda];
        count++;
      }
    }
  }

  for (int j = 0; j < lda; j+=jmax)
  {
    jmax = 32;
    int M = min (jmax, lda - j);

    if (M < 32 && M >= 16)
      M = 16;
    else if (M < 16 && M >= 8)
      M = 8;

    jmax = M;
    for (int i = 0; i < lda; i+=imax)
    {
      int I = min (imax, lda - i);

      for (int k = 0; k < lda; k+=kmax)
      {
        int K = min (kmax, lda - k);

        if (likely(M == 32))
        {
          do_block_staticavx16(lda, M, K, I, &altA[(j/16)*16*lda + 16*k], &B[k + i*lda], &C[j+i*lda], false);
          do_block_staticavx16(lda, M, K, I, &altA[((j/16) + 1)*16*lda+16*k], &B[k + i*lda], &C[j+16+i*lda], false);
        }
        else if (M == 16)
        {
          do_block_staticavx16(lda, M, K, I, &altA[(j/16)*16*lda + 16*k], &B[k + i*lda], &C[j+i*lda], false);
        }
        else if (M == 8)
        {
          do_block_staticavx8(lda, M, K, I, &A[j + k*lda], &B[k + i*lda], &C[j + i*lda], false);
        }
        else
        {
          do_block(lda, M, I, K, &A[j + k*lda], &B[k + i*lda], &C[j + i*lda]);
        }
      }
    }
  }

  free(altA);
}
