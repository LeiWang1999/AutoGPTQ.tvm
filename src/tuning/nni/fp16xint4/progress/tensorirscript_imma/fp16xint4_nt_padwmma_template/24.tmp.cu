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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif
__device__ int4 make_int4(signed char _0, signed char _1, signed char _2, signed char _3, signed char _4, signed char _5, signed char _6, signed char _7, signed char _8, signed char _9, signed char _10, signed char _11, signed char _12, signed char _13, signed char _14, signed char _15) {
  return make_int4(
    *((int *)&make_char4(_0, _1, _2, _3)),
    *((int *)&make_char4(_4, _5, _6, _7)),
    *((int *)&make_char4(_8, _9, _10, _11)),
    *((int *)&make_char4(_12, _13, _14, _15)));
}
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if (__CUDACC_VER_MAJOR__ >= 11) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

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
extern "C" __global__ void __launch_bounds__(32) main_kernel0(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[8];
  __shared__ half A_shared[2048];
  __shared__ half B_decompress_shared[16384];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_decompress_shared_wmma_matrix_b[8];
  for (int j_0_2_init = 0; j_0_2_init < 8; ++j_0_2_init) {
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[j_0_2_init], 0.000000e+00f);
  }
  for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((ax0_ax1_fused_2 * 256) + (((int)threadIdx.x) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((ax0_ax1_fused_2 * 256) + (((int)threadIdx.x) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((int)blockIdx.y) * 262144) + (ax0_ax1_fused_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1) {
    uint2 __1;
    int4 __2;
      int4 __3;
        int4 __4;
        int4 __5;
          int4 v_ = make_int4((((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_1 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_1 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_1 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_1 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + ((((int)threadIdx.x) & 15) * 2)));
          int4 __6;
            int4 v__1 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
            int4 v__2 = make_int4(2, 2, 2, 2);
            __6.x = (v__1.x%v__2.x);
            __6.y = (v__1.y%v__2.y);
            __6.z = (v__1.z%v__2.z);
            __6.w = (v__1.w%v__2.w);
          int4 __7;
            int4 v__3 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
            int4 v__4 = make_int4(2, 2, 2, 2);
            __7.x = (v__3.x/v__4.x);
            __7.y = (v__3.y/v__4.y);
            __7.z = (v__3.z/v__4.z);
            __7.w = (v__3.w/v__4.w);
          int4 __8;
          ushort4 __9;
            ushort4 __10;
              ushort4 __11;
                int4 v__5 = make_int4(2, 2, 2, 2);
                int4 v__6 = make_int4(0, 0, 0, 0);
                __11.x = (v__5.x>=v__6.x);
                __11.y = (v__5.y>=v__6.y);
                __11.z = (v__5.z>=v__6.z);
                __11.w = (v__5.w>=v__6.w);
              ushort4 __12;
                int4 v__7 = make_int4(0, 0, 0, 0);
                __12.x = (__6.x>=v__7.x);
                __12.y = (__6.y>=v__7.y);
                __12.z = (__6.z>=v__7.z);
                __12.w = (__6.w>=v__7.w);
              __10.x = (__11.x&&__12.x);
              __10.y = (__11.y&&__12.y);
              __10.z = (__11.z&&__12.z);
              __10.w = (__11.w&&__12.w);
            ushort4 __13;
              ushort4 __14;
                int4 v__8 = make_int4(2, 2, 2, 2);
                int4 v__9 = make_int4(0, 0, 0, 0);
                __14.x = (v__8.x<v__9.x);
                __14.y = (v__8.y<v__9.y);
                __14.z = (v__8.z<v__9.z);
                __14.w = (v__8.w<v__9.w);
              ushort4 __15;
                int4 v__10 = make_int4(0, 0, 0, 0);
                __15.x = (__6.x<=v__10.x);
                __15.y = (__6.y<=v__10.y);
                __15.z = (__6.z<=v__10.z);
                __15.w = (__6.w<=v__10.w);
              __13.x = (__14.x&&__15.x);
              __13.y = (__14.y&&__15.y);
              __13.z = (__14.z&&__15.z);
              __13.w = (__14.w&&__15.w);
            __9.x = (__10.x||__13.x);
            __9.y = (__10.y||__13.y);
            __9.z = (__10.z||__13.z);
            __9.w = (__10.w||__13.w);
          int4 __16;
            int4 v__11 = make_int4(1, 1, 1, 1);
            __16.x = (__7.x-v__11.x);
            __16.y = (__7.y-v__11.y);
            __16.z = (__7.z-v__11.z);
            __16.w = (__7.w-v__11.w);
          __8.x = (bool(__9.x)?__7.x:__16.x);
          __8.y = (bool(__9.y)?__7.y:__16.y);
          __8.z = (bool(__9.z)?__7.z:__16.z);
          __8.w = (bool(__9.w)?__7.w:__16.w);
          __5.x = (v_.x+__8.x);
          __5.y = (v_.y+__8.y);
          __5.z = (v_.z+__8.z);
          __5.w = (v_.w+__8.w);
        int v__12 = ((0x000000ff << 0) & (B[__5.x] << 0))|((0x000000ff << 8) & (B[__5.y] << 8))|((0x000000ff << 16) & (B[__5.z] << 16))|((0x000000ff << 24) & (B[__5.w] << 24));
        __4.x = (int)(((char)(v__12 >> 0)));
        __4.y = (int)(((char)(v__12 >> 8)));
        __4.z = (int)(((char)(v__12 >> 16)));
        __4.w = (int)(((char)(v__12 >> 24)));
        int4 __17;
          int4 __18;
            int4 v__13 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
            int4 v__14 = make_int4(2, 2, 2, 2);
            __18.x = (v__13.x%v__14.x);
            __18.y = (v__13.y%v__14.y);
            __18.z = (v__13.z%v__14.z);
            __18.w = (v__13.w%v__14.w);
          int4 __19;
          ushort4 __20;
            ushort4 __21;
              ushort4 __22;
                int4 v__15 = make_int4(2, 2, 2, 2);
                int4 v__16 = make_int4(0, 0, 0, 0);
                __22.x = (v__15.x>=v__16.x);
                __22.y = (v__15.y>=v__16.y);
                __22.z = (v__15.z>=v__16.z);
                __22.w = (v__15.w>=v__16.w);
              ushort4 __23;
                int4 v__17 = make_int4(0, 0, 0, 0);
                __23.x = (__18.x>=v__17.x);
                __23.y = (__18.y>=v__17.y);
                __23.z = (__18.z>=v__17.z);
                __23.w = (__18.w>=v__17.w);
              __21.x = (__22.x&&__23.x);
              __21.y = (__22.y&&__23.y);
              __21.z = (__22.z&&__23.z);
              __21.w = (__22.w&&__23.w);
            ushort4 __24;
              ushort4 __25;
                int4 v__18 = make_int4(2, 2, 2, 2);
                int4 v__19 = make_int4(0, 0, 0, 0);
                __25.x = (v__18.x<v__19.x);
                __25.y = (v__18.y<v__19.y);
                __25.z = (v__18.z<v__19.z);
                __25.w = (v__18.w<v__19.w);
              ushort4 __26;
                int4 v__20 = make_int4(0, 0, 0, 0);
                __26.x = (__18.x<=v__20.x);
                __26.y = (__18.y<=v__20.y);
                __26.z = (__18.z<=v__20.z);
                __26.w = (__18.w<=v__20.w);
              __24.x = (__25.x&&__26.x);
              __24.y = (__25.y&&__26.y);
              __24.z = (__25.z&&__26.z);
              __24.w = (__25.w&&__26.w);
            __20.x = (__21.x||__24.x);
            __20.y = (__21.y||__24.y);
            __20.z = (__21.z||__24.z);
            __20.w = (__21.w||__24.w);
          int4 __27;
            int4 v__21 = make_int4(2, 2, 2, 2);
            __27.x = (__18.x+v__21.x);
            __27.y = (__18.y+v__21.y);
            __27.z = (__18.z+v__21.z);
            __27.w = (__18.w+v__21.w);
          __19.x = (bool(__20.x)?__18.x:__27.x);
          __19.y = (bool(__20.y)?__18.y:__27.y);
          __19.z = (bool(__20.z)?__18.z:__27.z);
          __19.w = (bool(__20.w)?__18.w:__27.w);
          int4 v__22 = make_int4(4, 4, 4, 4);
          __17.x = (__19.x*v__22.x);
          __17.y = (__19.y*v__22.y);
          __17.z = (__19.z*v__22.z);
          __17.w = (__19.w*v__22.w);
        __3.x = (__4.x >> __17.x);
        __3.y = (__4.y >> __17.y);
        __3.z = (__4.z >> __17.z);
        __3.w = (__4.w >> __17.w);
      int4 v__23 = make_int4(15, 15, 15, 15);
      __2.x = (__3.x & v__23.x);
      __2.y = (__3.y & v__23.y);
      __2.z = (__3.z & v__23.z);
      __2.w = (__3.w & v__23.w);
    ((half2*)(&(__1.x)))->x = (half)(__2.x);
    ((half2*)(&(__1.x)))->y = (half)(__2.y);
    ((half2*)(&(__1.y)))->x = (half)(__2.z);
    ((half2*)(&(__1.y)))->y = (half)(__2.w);
    *(uint2*)(B_decompress_shared + ((ax0_ax1_fused_2_1 * 128) + (((int)threadIdx.x) * 4))) = __1;
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0_0 = 0; k_0_0 < 255; ++k_0_0) {
    __syncthreads();
    for (int ax0_ax1_fused_2_2 = 0; ax0_ax1_fused_2_2 < 4; ++ax0_ax1_fused_2_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((k_0_0 + 1) & 1) * 1024) + (ax0_ax1_fused_2_2 * 256)) + (((int)threadIdx.x) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((k_0_0 + 1) & 1) * 1024) + (ax0_ax1_fused_2_2 * 256)) + (((int)threadIdx.x) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((int)blockIdx.y) * 262144) + (ax0_ax1_fused_2_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16)
    );
  }
    }
    for (int ax0_ax1_fused_2_3 = 0; ax0_ax1_fused_2_3 < 64; ++ax0_ax1_fused_2_3) {
      uint2 __28;
      int4 __29;
        int4 __30;
          int4 __31;
          int4 __32;
            int4 __33;
              int4 v__24 = make_int4(((((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_3 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_3 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_3 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 8388608) + (((int)blockIdx.z) * 1048576)) + (ax0_ax1_fused_2_3 * 16384)) + ((((int)threadIdx.x) >> 4) * 8192)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)));
              int4 __34;
                int4 v__25 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 v__26 = make_int4(2, 2, 2, 2);
                __34.x = (v__25.x%v__26.x);
                __34.y = (v__25.y%v__26.y);
                __34.z = (v__25.z%v__26.z);
                __34.w = (v__25.w%v__26.w);
              int4 __35;
                int4 v__27 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 v__28 = make_int4(2, 2, 2, 2);
                __35.x = (v__27.x/v__28.x);
                __35.y = (v__27.y/v__28.y);
                __35.z = (v__27.z/v__28.z);
                __35.w = (v__27.w/v__28.w);
              int4 __36;
              ushort4 __37;
                ushort4 __38;
                  ushort4 __39;
                    int4 v__29 = make_int4(2, 2, 2, 2);
                    int4 v__30 = make_int4(0, 0, 0, 0);
                    __39.x = (v__29.x>=v__30.x);
                    __39.y = (v__29.y>=v__30.y);
                    __39.z = (v__29.z>=v__30.z);
                    __39.w = (v__29.w>=v__30.w);
                  ushort4 __40;
                    int4 v__31 = make_int4(0, 0, 0, 0);
                    __40.x = (__34.x>=v__31.x);
                    __40.y = (__34.y>=v__31.y);
                    __40.z = (__34.z>=v__31.z);
                    __40.w = (__34.w>=v__31.w);
                  __38.x = (__39.x&&__40.x);
                  __38.y = (__39.y&&__40.y);
                  __38.z = (__39.z&&__40.z);
                  __38.w = (__39.w&&__40.w);
                ushort4 __41;
                  ushort4 __42;
                    int4 v__32 = make_int4(2, 2, 2, 2);
                    int4 v__33 = make_int4(0, 0, 0, 0);
                    __42.x = (v__32.x<v__33.x);
                    __42.y = (v__32.y<v__33.y);
                    __42.z = (v__32.z<v__33.z);
                    __42.w = (v__32.w<v__33.w);
                  ushort4 __43;
                    int4 v__34 = make_int4(0, 0, 0, 0);
                    __43.x = (__34.x<=v__34.x);
                    __43.y = (__34.y<=v__34.y);
                    __43.z = (__34.z<=v__34.z);
                    __43.w = (__34.w<=v__34.w);
                  __41.x = (__42.x&&__43.x);
                  __41.y = (__42.y&&__43.y);
                  __41.z = (__42.z&&__43.z);
                  __41.w = (__42.w&&__43.w);
                __37.x = (__38.x||__41.x);
                __37.y = (__38.y||__41.y);
                __37.z = (__38.z||__41.z);
                __37.w = (__38.w||__41.w);
              int4 __44;
                int4 v__35 = make_int4(1, 1, 1, 1);
                __44.x = (__35.x-v__35.x);
                __44.y = (__35.y-v__35.y);
                __44.z = (__35.z-v__35.z);
                __44.w = (__35.w-v__35.w);
              __36.x = (bool(__37.x)?__35.x:__44.x);
              __36.y = (bool(__37.y)?__35.y:__44.y);
              __36.z = (bool(__37.z)?__35.z:__44.z);
              __36.w = (bool(__37.w)?__35.w:__44.w);
              __33.x = (v__24.x+__36.x);
              __33.y = (v__24.y+__36.y);
              __33.z = (v__24.z+__36.z);
              __33.w = (v__24.w+__36.w);
            int4 v__36 = make_int4(32, 32, 32, 32);
            __32.x = (__33.x+v__36.x);
            __32.y = (__33.y+v__36.y);
            __32.z = (__33.z+v__36.z);
            __32.w = (__33.w+v__36.w);
          int v__37 = ((0x000000ff << 0) & (B[__32.x] << 0))|((0x000000ff << 8) & (B[__32.y] << 8))|((0x000000ff << 16) & (B[__32.z] << 16))|((0x000000ff << 24) & (B[__32.w] << 24));
          __31.x = (int)(((char)(v__37 >> 0)));
          __31.y = (int)(((char)(v__37 >> 8)));
          __31.z = (int)(((char)(v__37 >> 16)));
          __31.w = (int)(((char)(v__37 >> 24)));
          int4 __45;
            int4 __46;
              int4 v__38 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 v__39 = make_int4(2, 2, 2, 2);
              __46.x = (v__38.x%v__39.x);
              __46.y = (v__38.y%v__39.y);
              __46.z = (v__38.z%v__39.z);
              __46.w = (v__38.w%v__39.w);
            int4 __47;
            ushort4 __48;
              ushort4 __49;
                ushort4 __50;
                  int4 v__40 = make_int4(2, 2, 2, 2);
                  int4 v__41 = make_int4(0, 0, 0, 0);
                  __50.x = (v__40.x>=v__41.x);
                  __50.y = (v__40.y>=v__41.y);
                  __50.z = (v__40.z>=v__41.z);
                  __50.w = (v__40.w>=v__41.w);
                ushort4 __51;
                  int4 v__42 = make_int4(0, 0, 0, 0);
                  __51.x = (__46.x>=v__42.x);
                  __51.y = (__46.y>=v__42.y);
                  __51.z = (__46.z>=v__42.z);
                  __51.w = (__46.w>=v__42.w);
                __49.x = (__50.x&&__51.x);
                __49.y = (__50.y&&__51.y);
                __49.z = (__50.z&&__51.z);
                __49.w = (__50.w&&__51.w);
              ushort4 __52;
                ushort4 __53;
                  int4 v__43 = make_int4(2, 2, 2, 2);
                  int4 v__44 = make_int4(0, 0, 0, 0);
                  __53.x = (v__43.x<v__44.x);
                  __53.y = (v__43.y<v__44.y);
                  __53.z = (v__43.z<v__44.z);
                  __53.w = (v__43.w<v__44.w);
                ushort4 __54;
                  int4 v__45 = make_int4(0, 0, 0, 0);
                  __54.x = (__46.x<=v__45.x);
                  __54.y = (__46.y<=v__45.y);
                  __54.z = (__46.z<=v__45.z);
                  __54.w = (__46.w<=v__45.w);
                __52.x = (__53.x&&__54.x);
                __52.y = (__53.y&&__54.y);
                __52.z = (__53.z&&__54.z);
                __52.w = (__53.w&&__54.w);
              __48.x = (__49.x||__52.x);
              __48.y = (__49.y||__52.y);
              __48.z = (__49.z||__52.z);
              __48.w = (__49.w||__52.w);
            int4 __55;
              int4 v__46 = make_int4(2, 2, 2, 2);
              __55.x = (__46.x+v__46.x);
              __55.y = (__46.y+v__46.y);
              __55.z = (__46.z+v__46.z);
              __55.w = (__46.w+v__46.w);
            __47.x = (bool(__48.x)?__46.x:__55.x);
            __47.y = (bool(__48.y)?__46.y:__55.y);
            __47.z = (bool(__48.z)?__46.z:__55.z);
            __47.w = (bool(__48.w)?__46.w:__55.w);
            int4 v__47 = make_int4(4, 4, 4, 4);
            __45.x = (__47.x*v__47.x);
            __45.y = (__47.y*v__47.y);
            __45.z = (__47.z*v__47.z);
            __45.w = (__47.w*v__47.w);
          __30.x = (__31.x >> __45.x);
          __30.y = (__31.y >> __45.y);
          __30.z = (__31.z >> __45.z);
          __30.w = (__31.w >> __45.w);
        int4 v__48 = make_int4(15, 15, 15, 15);
        __29.x = (__30.x & v__48.x);
        __29.y = (__30.y & v__48.y);
        __29.z = (__30.z & v__48.z);
        __29.w = (__30.w & v__48.w);
      ((half2*)(&(__28.x)))->x = (half)(__29.x);
      ((half2*)(&(__28.x)))->y = (half)(__29.y);
      ((half2*)(&(__28.y)))->x = (half)(__29.z);
      ((half2*)(&(__28.y)))->y = (half)(__29.w);
      *(uint2*)(B_decompress_shared + (((((k_0_0 + 1) & 1) * 8192) + (ax0_ax1_fused_2_3 * 128)) + (((int)threadIdx.x) * 4))) = __28;
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(((k_0_0 & 1) * 1024) + (k_0_1 * 16))])), 64);
      for (int ax0_0 = 0; ax0_0 < 8; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(B_decompress_shared_wmma_matrix_b[ax0_0], (&(B_decompress_shared[((((k_0_0 & 1) * 8192) + (ax0_0 * 1024)) + (k_0_1 * 16))])), 64);
      }
      for (int j_0_2 = 0; j_0_2 < 8; ++j_0_2) {
        nvcuda::wmma::mma_sync(C_wmma_accumulator[j_0_2], A_shared_wmma_matrix_a[0], B_decompress_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[j_0_2]);
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1) {
    nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((k_0_1_1 * 16) + 1024)])), 64);
    for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {
      nvcuda::wmma::load_matrix_sync(B_decompress_shared_wmma_matrix_b[ax0_0_1], (&(B_decompress_shared[(((ax0_0_1 * 1024) + (k_0_1_1 * 16)) + 8192)])), 64);
    }
    for (int j_0_2_1 = 0; j_0_2_1 < 8; ++j_0_2_1) {
      nvcuda::wmma::mma_sync(C_wmma_accumulator[j_0_2_1], A_shared_wmma_matrix_a[0], B_decompress_shared_wmma_matrix_b[j_0_2_1], C_wmma_accumulator[j_0_2_1]);
    }
  }
  for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
    nvcuda::wmma::store_matrix_sync((&(C[((((((int)blockIdx.y) * 262144) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.z) * 128)) + (ax1_0 * 16))])), C_wmma_accumulator[ax1_0], 16384, nvcuda::wmma::mem_row_major);
  }
}

