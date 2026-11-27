/*
 * Portable C11 + SIMD Fast Walsh-Hadamard Transform (FWHT) and fused
 * dense / sparse-CSR SORF projection kernels for mrff.
 *
 * Uses only the standard CPython Buffer Protocol (no NumPy C-API ABI coupling).
 * Supports ARM NEON (arm64 / Apple Silicon), x86 AVX2 / SSE2, and scalar C.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define MRFF_SIMD_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MRFF_SIMD_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define MRFF_SIMD_SSE2 1
#endif

static inline int is_power_of_two(Py_ssize_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static inline void fwht_row_f64(double *a, Py_ssize_t p, double inv_sqrt_p) {
    if (p <= 1) {
        return;
    }

    /* Stage h = 1 (distance 1) */
    for (Py_ssize_t i = 0; i < p; i += 2) {
        double u = a[i];
        double v = a[i + 1];
        a[i] = u + v;
        a[i + 1] = u - v;
    }

    /* Stage h = 2 (distance 2) */
    if (p >= 4) {
        for (Py_ssize_t i = 0; i < p; i += 4) {
#if defined(MRFF_SIMD_NEON)
            float64x2_t u = vld1q_f64(a + i);
            float64x2_t v = vld1q_f64(a + i + 2);
            vst1q_f64(a + i, vaddq_f64(u, v));
            vst1q_f64(a + i + 2, vsubq_f64(u, v));
#elif defined(MRFF_SIMD_SSE2)
            __m128d u = _mm_loadu_pd(a + i);
            __m128d v = _mm_loadu_pd(a + i + 2);
            _mm_storeu_pd(a + i, _mm_add_pd(u, v));
            _mm_storeu_pd(a + i + 2, _mm_sub_pd(u, v));
#else
            double u0 = a[i], u1 = a[i + 1];
            double v0 = a[i + 2], v1 = a[i + 3];
            a[i] = u0 + v0;
            a[i + 1] = u1 + v1;
            a[i + 2] = u0 - v0;
            a[i + 3] = u1 - v1;
#endif
        }
    }

    /* Stages h = 4, 8, ..., p / 2 */
    for (Py_ssize_t h = 4; h < p; h <<= 1) {
        Py_ssize_t step = h << 1;
        for (Py_ssize_t i = 0; i < p; i += step) {
            double *p0 = a + i;
            double *p1 = a + i + h;
            Py_ssize_t j = 0;
#if defined(MRFF_SIMD_AVX2)
            for (; j + 4 <= h; j += 4) {
                __m256d u = _mm256_loadu_pd(p0 + j);
                __m256d v = _mm256_loadu_pd(p1 + j);
                _mm256_storeu_pd(p0 + j, _mm256_add_pd(u, v));
                _mm256_storeu_pd(p1 + j, _mm256_sub_pd(u, v));
            }
#elif defined(MRFF_SIMD_NEON)
            for (; j + 4 <= h; j += 4) {
                float64x2_t u0 = vld1q_f64(p0 + j);
                float64x2_t v0 = vld1q_f64(p1 + j);
                float64x2_t u1 = vld1q_f64(p0 + j + 2);
                float64x2_t v1 = vld1q_f64(p1 + j + 2);
                vst1q_f64(p0 + j, vaddq_f64(u0, v0));
                vst1q_f64(p1 + j, vsubq_f64(u0, v0));
                vst1q_f64(p0 + j + 2, vaddq_f64(u1, v1));
                vst1q_f64(p1 + j + 2, vsubq_f64(u1, v1));
            }
#elif defined(MRFF_SIMD_SSE2)
            for (; j + 2 <= h; j += 2) {
                __m128d u = _mm_loadu_pd(p0 + j);
                __m128d v = _mm_loadu_pd(p1 + j);
                _mm_storeu_pd(p0 + j, _mm_add_pd(u, v));
                _mm_storeu_pd(p1 + j, _mm_sub_pd(u, v));
            }
#endif
            for (; j < h; ++j) {
                double u = p0[j];
                double v = p1[j];
                p0[j] = u + v;
                p1[j] = u - v;
            }
        }
    }

    if (inv_sqrt_p != 1.0) {
        Py_ssize_t i = 0;
#if defined(MRFF_SIMD_AVX2)
        __m256d s = _mm256_set1_pd(inv_sqrt_p);
        for (; i + 4 <= p; i += 4) {
            _mm256_storeu_pd(a + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), s));
        }
#elif defined(MRFF_SIMD_NEON)
        float64x2_t s = vdupq_n_f64(inv_sqrt_p);
        for (; i + 2 <= p; i += 2) {
            vst1q_f64(a + i, vmulq_f64(vld1q_f64(a + i), s));
        }
#elif defined(MRFF_SIMD_SSE2)
        __m128d s = _mm_set1_pd(inv_sqrt_p);
        for (; i + 2 <= p; i += 2) {
            _mm_storeu_pd(a + i, _mm_mul_pd(_mm_loadu_pd(a + i), s));
        }
#endif
        for (; i < p; ++i) {
            a[i] *= inv_sqrt_p;
        }
    }
}

static inline void mul_inplace_f64(double *buf, const double *rhs, Py_ssize_t n) {
    Py_ssize_t i = 0;
#if defined(MRFF_SIMD_AVX2)
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(buf + i, _mm256_mul_pd(_mm256_loadu_pd(buf + i), _mm256_loadu_pd(rhs + i)));
    }
#elif defined(MRFF_SIMD_NEON)
    for (; i + 2 <= n; i += 2) {
        vst1q_f64(buf + i, vmulq_f64(vld1q_f64(buf + i), vld1q_f64(rhs + i)));
    }
#elif defined(MRFF_SIMD_SSE2)
    for (; i + 2 <= n; i += 2) {
        _mm_storeu_pd(buf + i, _mm_mul_pd(_mm_loadu_pd(buf + i), _mm_loadu_pd(rhs + i)));
    }
#endif
    for (; i < n; ++i) {
        buf[i] *= rhs[i];
    }
}

static PyObject *py_fwht_inplace_f64(PyObject *self, PyObject *args) {
    Py_buffer buf;
    Py_ssize_t n_rows, p;
    int normalized;

    if (!PyArg_ParseTuple(args, "w*nnp", &buf, &n_rows, &p, &normalized)) {
        return NULL;
    }
    if (!is_power_of_two(p)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "p must be a positive power of two.");
        return NULL;
    }
    if (buf.len < n_rows * p * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "Buffer too small for (n_rows, p) float64.");
        return NULL;
    }

    double *data = (double *)buf.buf;
    double inv_sqrt_p = normalized ? (1.0 / sqrt((double)p)) : 1.0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < n_rows; ++r) {
        fwht_row_f64(data + r * p, p, inv_sqrt_p);
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

static PyObject *py_sorf_dense_f64(PyObject *self, PyObject *args) {
    Py_buffer x_buf, inv_ls_buf, signs_buf, radii_buf, out_buf;
    Py_ssize_t n_samples, d, p, n_blocks, n_stacks, n_freqs;

    if (!PyArg_ParseTuple(
            args,
            "y*y*y*y*w*nnnnnn",
            &x_buf,
            &inv_ls_buf,
            &signs_buf,
            &radii_buf,
            &out_buf,
            &n_samples,
            &d,
            &p,
            &n_blocks,
            &n_stacks,
            &n_freqs)) {
        return NULL;
    }
    if (!is_power_of_two(p) || d > p) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&inv_ls_buf);
        PyBuffer_Release(&signs_buf);
        PyBuffer_Release(&radii_buf);
        PyBuffer_Release(&out_buf);
        PyErr_SetString(PyExc_ValueError, "p must be a power of two >= d.");
        return NULL;
    }

    double *work = (double *)malloc((size_t)p * sizeof(double));
    if (work == NULL) {
        PyBuffer_Release(&x_buf);
        PyBuffer_Release(&inv_ls_buf);
        PyBuffer_Release(&signs_buf);
        PyBuffer_Release(&radii_buf);
        PyBuffer_Release(&out_buf);
        return PyErr_NoMemory();
    }

    const double *x = (const double *)x_buf.buf;
    const double *inv_ls = (const double *)inv_ls_buf.buf;
    const double *signs = (const double *)signs_buf.buf;
    const double *radii = (const double *)radii_buf.buf;
    double *out = (double *)out_buf.buf;
    const double inv_sqrt_p = 1.0 / sqrt((double)p);

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n_samples; ++i) {
        const double *x_row = x + i * d;
        double *out_row = out + i * n_freqs;
        Py_ssize_t offset = 0;

        for (Py_ssize_t b = 0; b < n_blocks && offset < n_freqs; ++b) {
            const double *block_signs = signs + b * n_stacks * p;
            const double *s0 = block_signs;

            for (Py_ssize_t j = 0; j < d; ++j) {
                work[j] = x_row[j] * inv_ls[j] * s0[j];
            }
            if (p > d) {
                memset(work + d, 0, (size_t)(p - d) * sizeof(double));
            }
            fwht_row_f64(work, p, inv_sqrt_p);

            for (Py_ssize_t s = 1; s < n_stacks; ++s) {
                mul_inplace_f64(work, block_signs + s * p, p);
                fwht_row_f64(work, p, inv_sqrt_p);
            }

            Py_ssize_t take = p < (n_freqs - offset) ? p : (n_freqs - offset);
            for (Py_ssize_t j = 0; j < take; ++j) {
                out_row[offset + j] = work[j] * radii[offset + j];
            }
            offset += take;
        }
    }
    Py_END_ALLOW_THREADS

    free(work);
    PyBuffer_Release(&x_buf);
    PyBuffer_Release(&inv_ls_buf);
    PyBuffer_Release(&signs_buf);
    PyBuffer_Release(&radii_buf);
    PyBuffer_Release(&out_buf);
    Py_RETURN_NONE;
}

static PyObject *py_sorf_csr_f64(PyObject *self, PyObject *args) {
    Py_buffer data_buf, idx_buf, ptr_buf, inv_ls_buf, signs_buf, radii_buf, out_buf;
    Py_ssize_t n_samples, d, p, n_blocks, n_stacks, n_freqs;

    if (!PyArg_ParseTuple(
            args,
            "y*y*y*y*y*y*w*nnnnnn",
            &data_buf,
            &idx_buf,
            &ptr_buf,
            &inv_ls_buf,
            &signs_buf,
            &radii_buf,
            &out_buf,
            &n_samples,
            &d,
            &p,
            &n_blocks,
            &n_stacks,
            &n_freqs)) {
        return NULL;
    }
    if (!is_power_of_two(p) || d > p) {
        PyBuffer_Release(&data_buf);
        PyBuffer_Release(&idx_buf);
        PyBuffer_Release(&ptr_buf);
        PyBuffer_Release(&inv_ls_buf);
        PyBuffer_Release(&signs_buf);
        PyBuffer_Release(&radii_buf);
        PyBuffer_Release(&out_buf);
        PyErr_SetString(PyExc_ValueError, "p must be a power of two >= d.");
        return NULL;
    }

    double *work = (double *)malloc((size_t)p * sizeof(double));
    if (work == NULL) {
        PyBuffer_Release(&data_buf);
        PyBuffer_Release(&idx_buf);
        PyBuffer_Release(&ptr_buf);
        PyBuffer_Release(&inv_ls_buf);
        PyBuffer_Release(&signs_buf);
        PyBuffer_Release(&radii_buf);
        PyBuffer_Release(&out_buf);
        return PyErr_NoMemory();
    }

    const double *data = (const double *)data_buf.buf;
    const int64_t *indices = (const int64_t *)idx_buf.buf;
    const int64_t *indptr = (const int64_t *)ptr_buf.buf;
    const double *inv_ls = (const double *)inv_ls_buf.buf;
    const double *signs = (const double *)signs_buf.buf;
    const double *radii = (const double *)radii_buf.buf;
    double *out = (double *)out_buf.buf;
    const double inv_sqrt_p = 1.0 / sqrt((double)p);

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n_samples; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        double *out_row = out + i * n_freqs;
        Py_ssize_t offset = 0;

        for (Py_ssize_t b = 0; b < n_blocks && offset < n_freqs; ++b) {
            const double *block_signs = signs + b * n_stacks * p;
            const double *s0 = block_signs;

            memset(work, 0, (size_t)p * sizeof(double));
            for (int64_t jj = start; jj < end; ++jj) {
                int64_t col = indices[jj];
                work[col] += data[jj] * inv_ls[col] * s0[col];
            }
            fwht_row_f64(work, p, inv_sqrt_p);

            for (Py_ssize_t s = 1; s < n_stacks; ++s) {
                mul_inplace_f64(work, block_signs + s * p, p);
                fwht_row_f64(work, p, inv_sqrt_p);
            }

            Py_ssize_t take = p < (n_freqs - offset) ? p : (n_freqs - offset);
            for (Py_ssize_t j = 0; j < take; ++j) {
                out_row[offset + j] = work[j] * radii[offset + j];
            }
            offset += take;
        }
    }
    Py_END_ALLOW_THREADS

    free(work);
    PyBuffer_Release(&data_buf);
    PyBuffer_Release(&idx_buf);
    PyBuffer_Release(&ptr_buf);
    PyBuffer_Release(&inv_ls_buf);
    PyBuffer_Release(&signs_buf);
    PyBuffer_Release(&radii_buf);
    PyBuffer_Release(&out_buf);
    Py_RETURN_NONE;
}

static PyObject *py_simd_backend(PyObject *self, PyObject *Py_UNUSED(ignored)) {
#if defined(MRFF_SIMD_NEON)
    return PyUnicode_FromString("neon");
#elif defined(MRFF_SIMD_AVX2)
    return PyUnicode_FromString("avx2");
#elif defined(MRFF_SIMD_SSE2)
    return PyUnicode_FromString("sse2");
#else
    return PyUnicode_FromString("scalar");
#endif
}

static PyMethodDef FwhtMethods[] = {
    {"fwht_inplace_f64", py_fwht_inplace_f64, METH_VARARGS,
     "In-place SIMD Fast Walsh-Hadamard Transform on contiguous float64 rows."},
    {"sorf_dense_f64", py_sorf_dense_f64, METH_VARARGS,
     "Fused dense SORF projection in L1 cache."},
    {"sorf_csr_f64", py_sorf_csr_f64, METH_VARARGS,
     "Zero-densification sparse CSR SORF projection."},
    {"simd_backend", py_simd_backend, METH_NOARGS,
     "Return active SIMD backend name ('neon', 'avx2', 'sse2', or 'scalar')."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fwhtmodule = {
    PyModuleDef_HEAD_INIT,
    "_fwht_c",
    "Portable C11 + SIMD FWHT and fused SORF kernels.",
    -1,
    FwhtMethods
};

PyMODINIT_FUNC PyInit__fwht_c(void) {
    return PyModule_Create(&fwhtmodule);
}
