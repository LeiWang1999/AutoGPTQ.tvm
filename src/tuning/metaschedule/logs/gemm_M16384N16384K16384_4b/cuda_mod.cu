#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#endif
#include <mma.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) main_kernel0(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_reindex_shared_dyn_wmma_accumulator[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_decompress_reindex_shared_dyn_wmma_matrix_b[8];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[4], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[5], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[6], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[7], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[8], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[9], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[12], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[13], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[10], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[11], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[14], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[15], 0.000000e+00f);
  for (int ax2_0_0 = 0; ax2_0_0 < 512; ++ax2_0_0) {
    __syncthreads();
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 5120)) = *(uint2*)(A + (((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 6400)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 524288));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7680)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1048576));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8960)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1572864));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10240)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2097152));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 11520)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2621440));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12800)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 3145728));
    *(uint2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 14080)) = *(uint2*)(A + ((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 3670016));
    uint1 __1;
    int2 __2;
      int2 __3;
        int2 v_ = make_int2(((int)B[(((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15))]), ((int)B[(((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15))]));
        int2 v__1 = make_int2((0)+(4*0), (0)+(4*1));
        __3.x = (v_.x >> v__1.x);
        __3.y = (v_.y >> v__1.y);
      int2 v__2 = make_int2(15, 15);
      __2.x = (__3.x & v__2.x);
      __2.y = (__3.y & v__2.y);
    ((half2*)(&(__1.x)))->x = (half)(__2.x);
    ((half2*)(&(__1.x)))->y = (half)(__2.y);
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2))) = __1;
    uint1 __4;
    int2 __5;
      int2 __6;
        int2 v__3 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 131072)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 131072)]));
        int2 v__4 = make_int2((0)+(4*0), (0)+(4*1));
        __6.x = (v__3.x >> v__4.x);
        __6.y = (v__3.y >> v__4.y);
      int2 v__5 = make_int2(15, 15);
      __5.x = (__6.x & v__5.x);
      __5.y = (__6.y & v__5.y);
    ((half2*)(&(__4.x)))->x = (half)(__5.x);
    ((half2*)(&(__4.x)))->y = (half)(__5.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 640)) = __4;
    uint1 __7;
    int2 __8;
      int2 __9;
        int2 v__6 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 262144)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 262144)]));
        int2 v__7 = make_int2((0)+(4*0), (0)+(4*1));
        __9.x = (v__6.x >> v__7.x);
        __9.y = (v__6.y >> v__7.y);
      int2 v__8 = make_int2(15, 15);
      __8.x = (__9.x & v__8.x);
      __8.y = (__9.y & v__8.y);
    ((half2*)(&(__7.x)))->x = (half)(__8.x);
    ((half2*)(&(__7.x)))->y = (half)(__8.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 1280)) = __7;
    uint1 __10;
    int2 __11;
      int2 __12;
        int2 v__9 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 393216)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 393216)]));
        int2 v__10 = make_int2((0)+(4*0), (0)+(4*1));
        __12.x = (v__9.x >> v__10.x);
        __12.y = (v__9.y >> v__10.y);
      int2 v__11 = make_int2(15, 15);
      __11.x = (__12.x & v__11.x);
      __11.y = (__12.y & v__11.y);
    ((half2*)(&(__10.x)))->x = (half)(__11.x);
    ((half2*)(&(__10.x)))->y = (half)(__11.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 1920)) = __10;
    uint1 __13;
    int2 __14;
      int2 __15;
        int2 v__12 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 524288)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 524288)]));
        int2 v__13 = make_int2((0)+(4*0), (0)+(4*1));
        __15.x = (v__12.x >> v__13.x);
        __15.y = (v__12.y >> v__13.y);
      int2 v__14 = make_int2(15, 15);
      __14.x = (__15.x & v__14.x);
      __14.y = (__15.y & v__14.y);
    ((half2*)(&(__13.x)))->x = (half)(__14.x);
    ((half2*)(&(__13.x)))->y = (half)(__14.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 2560)) = __13;
    uint1 __16;
    int2 __17;
      int2 __18;
        int2 v__15 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 655360)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 655360)]));
        int2 v__16 = make_int2((0)+(4*0), (0)+(4*1));
        __18.x = (v__15.x >> v__16.x);
        __18.y = (v__15.y >> v__16.y);
      int2 v__17 = make_int2(15, 15);
      __17.x = (__18.x & v__17.x);
      __17.y = (__18.y & v__17.y);
    ((half2*)(&(__16.x)))->x = (half)(__17.x);
    ((half2*)(&(__16.x)))->y = (half)(__17.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 3200)) = __16;
    uint1 __19;
    int2 __20;
      int2 __21;
        int2 v__18 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 786432)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 786432)]));
        int2 v__19 = make_int2((0)+(4*0), (0)+(4*1));
        __21.x = (v__18.x >> v__19.x);
        __21.y = (v__18.y >> v__19.y);
      int2 v__20 = make_int2(15, 15);
      __20.x = (__21.x & v__20.x);
      __20.y = (__21.y & v__20.y);
    ((half2*)(&(__19.x)))->x = (half)(__20.x);
    ((half2*)(&(__19.x)))->y = (half)(__20.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 3840)) = __19;
    uint1 __22;
    int2 __23;
      int2 __24;
        int2 v__21 = make_int2(((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 917504)]), ((int)B[((((((((((int)blockIdx.y) & 63) * 2097152) + ((((int)blockIdx.x) & 1) * 1048576)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (ax2_0_0 * 16)) + (((int)threadIdx.x) & 15)) + 917504)]));
        int2 v__22 = make_int2((0)+(4*0), (0)+(4*1));
        __24.x = (v__21.x >> v__22.x);
        __24.y = (v__21.y >> v__22.y);
      int2 v__23 = make_int2(15, 15);
      __23.x = (__24.x & v__23.x);
      __23.y = (__24.y & v__23.y);
    ((half2*)(&(__22.x)))->x = (half)(__23.x);
    ((half2*)(&(__22.x)))->y = (half)(__23.y);
    *(uint1*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 80) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)) + 4480)) = __22;
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 5120)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 5136)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 5760)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 5776)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[4], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 6400)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[5], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 6416)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[6], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 7040)])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[7], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 2560) + 7056)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) & 1) * 2560)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 16)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 640)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 656)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 1280)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 1296)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 1920)])), 40);
    nvcuda::wmma::load_matrix_sync(B_decompress_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 2560) + 1936)])), 40);
    for (int ax0_0_3 = 0; ax0_0_3 < 2; ++ax0_0_3) {
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[(ax0_0_3 * 8)], A_reindex_shared_dyn_wmma_matrix_a[(ax0_0_3 * 4)], B_decompress_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[(ax0_0_3 * 8)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 1)], A_reindex_shared_dyn_wmma_matrix_a[(ax0_0_3 * 4)], B_decompress_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 1)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 4)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 2)], B_decompress_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 4)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 5)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 2)], B_decompress_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 5)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[(ax0_0_3 * 8)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 1)], B_decompress_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[(ax0_0_3 * 8)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 1)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 1)], B_decompress_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 1)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 4)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 3)], B_decompress_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 4)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 5)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 3)], B_decompress_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 5)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 2)], A_reindex_shared_dyn_wmma_matrix_a[(ax0_0_3 * 4)], B_decompress_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 2)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 3)], A_reindex_shared_dyn_wmma_matrix_a[(ax0_0_3 * 4)], B_decompress_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 3)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 6)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 2)], B_decompress_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 6)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 7)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 2)], B_decompress_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 7)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 2)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 1)], B_decompress_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 2)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 3)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 1)], B_decompress_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 3)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 6)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 3)], B_decompress_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 6)]);
      nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 7)], A_reindex_shared_dyn_wmma_matrix_a[((ax0_0_3 * 4) + 3)], B_decompress_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[((ax0_0_3 * 8) + 7)]);
    }
  }
  for (int ax2 = 0; ax2 < 4; ++ax2) {
    __syncthreads();
    nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 5120)])), C_reindex_shared_dyn_wmma_accumulator[(ax2 * 4)], 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 5376)])), C_reindex_shared_dyn_wmma_accumulator[((ax2 * 4) + 1)], 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 5632)])), C_reindex_shared_dyn_wmma_accumulator[((ax2 * 4) + 2)], 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 5888)])), C_reindex_shared_dyn_wmma_accumulator[((ax2 * 4) + 3)], 16, nvcuda::wmma::mem_row_major);
    __syncthreads();
    *(uint2*)(C + ((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 5120));
    *(uint2*)(C + (((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 64)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 6144));
    *(uint2*)(C + (((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 1048576)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 7168));
    *(uint2*)(C + ((((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 3072) >> 11) * 1048576)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 64)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 8192));
    *(uint2*)(C + (((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2097152)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 9216));
    *(uint2*)(C + ((((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 5120) >> 11) * 1048576)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 64)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 10240));
    *(uint2*)(C + (((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 3145728)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 11264));
    *(uint2*)(C + ((((((((((((((int)blockIdx.y) >> 6) * 16777216) + ((((int)blockIdx.x) >> 1) * 4194304)) + (((((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 7168) >> 11) * 1048576)) + (ax2 * 262144)) + ((((int)threadIdx.y) & 1) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)blockIdx.y) & 63) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.y) >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 64)) = *(uint2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 12288));
  }
}

