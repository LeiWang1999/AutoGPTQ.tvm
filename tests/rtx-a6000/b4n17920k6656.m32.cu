#include <cuda_fp16.h>
#include <mma.h>

            static inline __device__ __host__ unsigned
            __pack_half2(const half x, const half y) {
            unsigned v0 = *((unsigned short *)&x);
            unsigned v1 = *((unsigned short *)&y);
            return (v1 << 16) | v0;
        }extern "C" __global__ void __launch_bounds__(256) tir_halfxint3_tensorop_32x128x64x2_t0_y1z8_K6656_align8(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ Scales, half* __restrict__ Zeros, half* __restrict__ C, int m) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[2];
  __shared__ half APad_shared[4608];
  __shared__ half B_rescale_shared[18432];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> APad_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[1];
  for (int i_0_2_init = 0; i_0_2_init < 2; ++i_0_2_init) {
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[i_0_2_init], 0.000000e+00f);
  }

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
      : "=r"(addr)
      : "l"((void *)(APad_shared + (((((int)threadIdx.z) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))))
    );
    int pred_guard = (int)(((((int)threadIdx.z) * 4) + (((int)threadIdx.x) >> 3)) < m);
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.cg.shared.global [%1], [%2], %3;"
      #endif
      "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)(A + (((((int)threadIdx.z) * 26624) + ((((int)threadIdx.x) >> 3) * 6656)) + ((((int)threadIdx.x) & 7) * 8)))), "n"(16), "r"(0), "r"(0), "r"(0),"r"(0)
    );
  }
  for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 8; ++ax0_ax1_fused_2) {
    uint2 __1;
      uint2 __2;
        uint2 __3;
        int4 __4;
          int4 __5;
            int4 __6;
            int4 __7;
              int4 v_ = make_int4((((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + ((((int)threadIdx.x) & 15) * 2)), (((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + ((((int)threadIdx.x) & 15) * 2)));
              int4 __8;
                int4 v__1 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 v__2 = make_int4(2, 2, 2, 2);
                __8.x = (v__1.x%v__2.x);
                __8.y = (v__1.y%v__2.y);
                __8.z = (v__1.z%v__2.z);
                __8.w = (v__1.w%v__2.w);
              int4 __9;
                int4 v__3 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 v__4 = make_int4(2, 2, 2, 2);
                __9.x = (v__3.x/v__4.x);
                __9.y = (v__3.y/v__4.y);
                __9.z = (v__3.z/v__4.z);
                __9.w = (v__3.w/v__4.w);
              int4 __10;
              ushort4 __11;
                ushort4 __12;
                  ushort4 __13;
                    int4 v__5 = make_int4(2, 2, 2, 2);
                    int4 v__6 = make_int4(0, 0, 0, 0);
                    __13.x = (v__5.x>=v__6.x);
                    __13.y = (v__5.y>=v__6.y);
                    __13.z = (v__5.z>=v__6.z);
                    __13.w = (v__5.w>=v__6.w);
                  ushort4 __14;
                    int4 v__7 = make_int4(0, 0, 0, 0);
                    __14.x = (__8.x>=v__7.x);
                    __14.y = (__8.y>=v__7.y);
                    __14.z = (__8.z>=v__7.z);
                    __14.w = (__8.w>=v__7.w);
                  __12.x = (__13.x&&__14.x);
                  __12.y = (__13.y&&__14.y);
                  __12.z = (__13.z&&__14.z);
                  __12.w = (__13.w&&__14.w);
                ushort4 __15;
                  ushort4 __16;
                    int4 v__8 = make_int4(2, 2, 2, 2);
                    int4 v__9 = make_int4(0, 0, 0, 0);
                    __16.x = (v__8.x<v__9.x);
                    __16.y = (v__8.y<v__9.y);
                    __16.z = (v__8.z<v__9.z);
                    __16.w = (v__8.w<v__9.w);
                  ushort4 __17;
                    int4 v__10 = make_int4(0, 0, 0, 0);
                    __17.x = (__8.x<=v__10.x);
                    __17.y = (__8.y<=v__10.y);
                    __17.z = (__8.z<=v__10.z);
                    __17.w = (__8.w<=v__10.w);
                  __15.x = (__16.x&&__17.x);
                  __15.y = (__16.y&&__17.y);
                  __15.z = (__16.z&&__17.z);
                  __15.w = (__16.w&&__17.w);
                __11.x = (__12.x||__15.x);
                __11.y = (__12.y||__15.y);
                __11.z = (__12.z||__15.z);
                __11.w = (__12.w||__15.w);
              int4 __18;
                int4 v__11 = make_int4(1, 1, 1, 1);
                __18.x = (__9.x-v__11.x);
                __18.y = (__9.y-v__11.y);
                __18.z = (__9.z-v__11.z);
                __18.w = (__9.w-v__11.w);
              __10.x = (bool(__11.x)?__9.x:__18.x);
              __10.y = (bool(__11.y)?__9.y:__18.y);
              __10.z = (bool(__11.z)?__9.z:__18.z);
              __10.w = (bool(__11.w)?__9.w:__18.w);
              __7.x = (v_.x+__10.x);
              __7.y = (v_.y+__10.y);
              __7.z = (v_.z+__10.z);
              __7.w = (v_.w+__10.w);
            int v__12 = ((0x000000ff << 0) & (B[__7.x] << 0))|((0x000000ff << 8) & (B[__7.y] << 8))|((0x000000ff << 16) & (B[__7.z] << 16))|((0x000000ff << 24) & (B[__7.w] << 24));
            __6.x = (int)(((char)(v__12 >> 0)));
            __6.y = (int)(((char)(v__12 >> 8)));
            __6.z = (int)(((char)(v__12 >> 16)));
            __6.w = (int)(((char)(v__12 >> 24)));
            int4 __19;
              int4 __20;
                int4 v__13 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 v__14 = make_int4(2, 2, 2, 2);
                __20.x = (v__13.x%v__14.x);
                __20.y = (v__13.y%v__14.y);
                __20.z = (v__13.z%v__14.z);
                __20.w = (v__13.w%v__14.w);
              int4 __21;
              ushort4 __22;
                ushort4 __23;
                  ushort4 __24;
                    int4 v__15 = make_int4(2, 2, 2, 2);
                    int4 v__16 = make_int4(0, 0, 0, 0);
                    __24.x = (v__15.x>=v__16.x);
                    __24.y = (v__15.y>=v__16.y);
                    __24.z = (v__15.z>=v__16.z);
                    __24.w = (v__15.w>=v__16.w);
                  ushort4 __25;
                    int4 v__17 = make_int4(0, 0, 0, 0);
                    __25.x = (__20.x>=v__17.x);
                    __25.y = (__20.y>=v__17.y);
                    __25.z = (__20.z>=v__17.z);
                    __25.w = (__20.w>=v__17.w);
                  __23.x = (__24.x&&__25.x);
                  __23.y = (__24.y&&__25.y);
                  __23.z = (__24.z&&__25.z);
                  __23.w = (__24.w&&__25.w);
                ushort4 __26;
                  ushort4 __27;
                    int4 v__18 = make_int4(2, 2, 2, 2);
                    int4 v__19 = make_int4(0, 0, 0, 0);
                    __27.x = (v__18.x<v__19.x);
                    __27.y = (v__18.y<v__19.y);
                    __27.z = (v__18.z<v__19.z);
                    __27.w = (v__18.w<v__19.w);
                  ushort4 __28;
                    int4 v__20 = make_int4(0, 0, 0, 0);
                    __28.x = (__20.x<=v__20.x);
                    __28.y = (__20.y<=v__20.y);
                    __28.z = (__20.z<=v__20.z);
                    __28.w = (__20.w<=v__20.w);
                  __26.x = (__27.x&&__28.x);
                  __26.y = (__27.y&&__28.y);
                  __26.z = (__27.z&&__28.z);
                  __26.w = (__27.w&&__28.w);
                __22.x = (__23.x||__26.x);
                __22.y = (__23.y||__26.y);
                __22.z = (__23.z||__26.z);
                __22.w = (__23.w||__26.w);
              int4 __29;
                int4 v__21 = make_int4(2, 2, 2, 2);
                __29.x = (__20.x+v__21.x);
                __29.y = (__20.y+v__21.y);
                __29.z = (__20.z+v__21.z);
                __29.w = (__20.w+v__21.w);
              __21.x = (bool(__22.x)?__20.x:__29.x);
              __21.y = (bool(__22.y)?__20.y:__29.y);
              __21.z = (bool(__22.z)?__20.z:__29.z);
              __21.w = (bool(__22.w)?__20.w:__29.w);
              int4 v__22 = make_int4(4, 4, 4, 4);
              __19.x = (__21.x*v__22.x);
              __19.y = (__21.y*v__22.y);
              __19.z = (__21.z*v__22.z);
              __19.w = (__21.w*v__22.w);
            __5.x = (__6.x >> __19.x);
            __5.y = (__6.y >> __19.y);
            __5.z = (__6.z >> __19.z);
            __5.w = (__6.w >> __19.w);
          int4 v__23 = make_int4(15, 15, 15, 15);
          __4.x = (__5.x & v__23.x);
          __4.y = (__5.y & v__23.y);
          __4.z = (__5.z & v__23.z);
          __4.w = (__5.w & v__23.w);
        ((half2*)(&(__3.x)))->x = (half)(__4.x);
        ((half2*)(&(__3.x)))->y = (half)(__4.y);
        ((half2*)(&(__3.y)))->x = (half)(__4.z);
        ((half2*)(&(__3.y)))->y = (half)(__4.w);
        uint2 v__24 = make_uint2(__pack_half2(Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))], Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))]), __pack_half2(Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))], Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))]));
        ((half2*)(&(__2.x)))->x = (((half2*)(&(__3.x)))->x*((half2*)(&(v__24.x)))->x);
        ((half2*)(&(__2.x)))->y = (((half2*)(&(__3.x)))->y*((half2*)(&(v__24.x)))->y);
        ((half2*)(&(__2.y)))->x = (((half2*)(&(__3.y)))->x*((half2*)(&(v__24.y)))->x);
        ((half2*)(&(__2.y)))->y = (((half2*)(&(__3.y)))->y*((half2*)(&(v__24.y)))->y);
      uint2 v__25 = make_uint2(__pack_half2(Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))], Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))]), __pack_half2(Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))], Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2 * 2)) + (((int)threadIdx.x) >> 4))]));
      ((half2*)(&(__1.x)))->x = (((half2*)(&(__2.x)))->x-((half2*)(&(v__25.x)))->x);
      ((half2*)(&(__1.x)))->y = (((half2*)(&(__2.x)))->y-((half2*)(&(v__25.x)))->y);
      ((half2*)(&(__1.y)))->x = (((half2*)(&(__2.y)))->x-((half2*)(&(v__25.y)))->x);
      ((half2*)(&(__1.y)))->y = (((half2*)(&(__2.y)))->y-((half2*)(&(v__25.y)))->y);
    *(uint2*)(B_rescale_shared + ((((((int)threadIdx.z) * 1152) + (ax0_ax1_fused_2 * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = __1;
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0_0 = 0; k_0_0 < 103; ++k_0_0) {
    __syncthreads();

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
      : "=r"(addr)
      : "l"((void *)(APad_shared + ((((((k_0_0 + 1) & 1) * 2304) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))))
    );
    int pred_guard = (int)(((((int)threadIdx.z) * 4) + (((int)threadIdx.x) >> 3)) < m);
    __asm__ __volatile__(
        "{  .reg .pred p;"
        "  setp.ne.b32 p, %0, 0;"
      #if TVM_ENABLE_L2_PREFETCH
        " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;"
      #else
        " @p cp.async.cg.shared.global [%1], [%2], %3;"
      #endif
      "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};}"
        :: "r"(pred_guard), "r"(addr), "l"((void*)(A + (((((((int)threadIdx.z) * 26624) + ((((int)threadIdx.x) >> 3) * 6656)) + (k_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64))), "n"(16), "r"(0), "r"(0), "r"(0),"r"(0)
    );
  }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 8; ++ax0_ax1_fused_2_1) {
      uint2 __30;
        uint2 __31;
          uint2 __32;
          int4 __33;
            int4 __34;
              int4 __35;
              int4 __36;
                int4 __37;
                  int4 v__26 = make_int4(((((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2_1 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2_1 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2_1 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)), ((((((((int)blockIdx.x) * 425984) + (((int)threadIdx.z) * 53248)) + (ax0_ax1_fused_2_1 * 6656)) + ((((int)threadIdx.x) >> 4) * 3328)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)));
                  int4 __38;
                    int4 v__27 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                    int4 v__28 = make_int4(2, 2, 2, 2);
                    __38.x = (v__27.x%v__28.x);
                    __38.y = (v__27.y%v__28.y);
                    __38.z = (v__27.z%v__28.z);
                    __38.w = (v__27.w%v__28.w);
                  int4 __39;
                    int4 v__29 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                    int4 v__30 = make_int4(2, 2, 2, 2);
                    __39.x = (v__29.x/v__30.x);
                    __39.y = (v__29.y/v__30.y);
                    __39.z = (v__29.z/v__30.z);
                    __39.w = (v__29.w/v__30.w);
                  int4 __40;
                  ushort4 __41;
                    ushort4 __42;
                      ushort4 __43;
                        int4 v__31 = make_int4(2, 2, 2, 2);
                        int4 v__32 = make_int4(0, 0, 0, 0);
                        __43.x = (v__31.x>=v__32.x);
                        __43.y = (v__31.y>=v__32.y);
                        __43.z = (v__31.z>=v__32.z);
                        __43.w = (v__31.w>=v__32.w);
                      ushort4 __44;
                        int4 v__33 = make_int4(0, 0, 0, 0);
                        __44.x = (__38.x>=v__33.x);
                        __44.y = (__38.y>=v__33.y);
                        __44.z = (__38.z>=v__33.z);
                        __44.w = (__38.w>=v__33.w);
                      __42.x = (__43.x&&__44.x);
                      __42.y = (__43.y&&__44.y);
                      __42.z = (__43.z&&__44.z);
                      __42.w = (__43.w&&__44.w);
                    ushort4 __45;
                      ushort4 __46;
                        int4 v__34 = make_int4(2, 2, 2, 2);
                        int4 v__35 = make_int4(0, 0, 0, 0);
                        __46.x = (v__34.x<v__35.x);
                        __46.y = (v__34.y<v__35.y);
                        __46.z = (v__34.z<v__35.z);
                        __46.w = (v__34.w<v__35.w);
                      ushort4 __47;
                        int4 v__36 = make_int4(0, 0, 0, 0);
                        __47.x = (__38.x<=v__36.x);
                        __47.y = (__38.y<=v__36.y);
                        __47.z = (__38.z<=v__36.z);
                        __47.w = (__38.w<=v__36.w);
                      __45.x = (__46.x&&__47.x);
                      __45.y = (__46.y&&__47.y);
                      __45.z = (__46.z&&__47.z);
                      __45.w = (__46.w&&__47.w);
                    __41.x = (__42.x||__45.x);
                    __41.y = (__42.y||__45.y);
                    __41.z = (__42.z||__45.z);
                    __41.w = (__42.w||__45.w);
                  int4 __48;
                    int4 v__37 = make_int4(1, 1, 1, 1);
                    __48.x = (__39.x-v__37.x);
                    __48.y = (__39.y-v__37.y);
                    __48.z = (__39.z-v__37.z);
                    __48.w = (__39.w-v__37.w);
                  __40.x = (bool(__41.x)?__39.x:__48.x);
                  __40.y = (bool(__41.y)?__39.y:__48.y);
                  __40.z = (bool(__41.z)?__39.z:__48.z);
                  __40.w = (bool(__41.w)?__39.w:__48.w);
                  __37.x = (v__26.x+__40.x);
                  __37.y = (v__26.y+__40.y);
                  __37.z = (v__26.z+__40.z);
                  __37.w = (v__26.w+__40.w);
                int4 v__38 = make_int4(32, 32, 32, 32);
                __36.x = (__37.x+v__38.x);
                __36.y = (__37.y+v__38.y);
                __36.z = (__37.z+v__38.z);
                __36.w = (__37.w+v__38.w);
              int v__39 = ((0x000000ff << 0) & (B[__36.x] << 0))|((0x000000ff << 8) & (B[__36.y] << 8))|((0x000000ff << 16) & (B[__36.z] << 16))|((0x000000ff << 24) & (B[__36.w] << 24));
              __35.x = (int)(((char)(v__39 >> 0)));
              __35.y = (int)(((char)(v__39 >> 8)));
              __35.z = (int)(((char)(v__39 >> 16)));
              __35.w = (int)(((char)(v__39 >> 24)));
              int4 __49;
                int4 __50;
                  int4 v__40 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                  int4 v__41 = make_int4(2, 2, 2, 2);
                  __50.x = (v__40.x%v__41.x);
                  __50.y = (v__40.y%v__41.y);
                  __50.z = (v__40.z%v__41.z);
                  __50.w = (v__40.w%v__41.w);
                int4 __51;
                ushort4 __52;
                  ushort4 __53;
                    ushort4 __54;
                      int4 v__42 = make_int4(2, 2, 2, 2);
                      int4 v__43 = make_int4(0, 0, 0, 0);
                      __54.x = (v__42.x>=v__43.x);
                      __54.y = (v__42.y>=v__43.y);
                      __54.z = (v__42.z>=v__43.z);
                      __54.w = (v__42.w>=v__43.w);
                    ushort4 __55;
                      int4 v__44 = make_int4(0, 0, 0, 0);
                      __55.x = (__50.x>=v__44.x);
                      __55.y = (__50.y>=v__44.y);
                      __55.z = (__50.z>=v__44.z);
                      __55.w = (__50.w>=v__44.w);
                    __53.x = (__54.x&&__55.x);
                    __53.y = (__54.y&&__55.y);
                    __53.z = (__54.z&&__55.z);
                    __53.w = (__54.w&&__55.w);
                  ushort4 __56;
                    ushort4 __57;
                      int4 v__45 = make_int4(2, 2, 2, 2);
                      int4 v__46 = make_int4(0, 0, 0, 0);
                      __57.x = (v__45.x<v__46.x);
                      __57.y = (v__45.y<v__46.y);
                      __57.z = (v__45.z<v__46.z);
                      __57.w = (v__45.w<v__46.w);
                    ushort4 __58;
                      int4 v__47 = make_int4(0, 0, 0, 0);
                      __58.x = (__50.x<=v__47.x);
                      __58.y = (__50.y<=v__47.y);
                      __58.z = (__50.z<=v__47.z);
                      __58.w = (__50.w<=v__47.w);
                    __56.x = (__57.x&&__58.x);
                    __56.y = (__57.y&&__58.y);
                    __56.z = (__57.z&&__58.z);
                    __56.w = (__57.w&&__58.w);
                  __52.x = (__53.x||__56.x);
                  __52.y = (__53.y||__56.y);
                  __52.z = (__53.z||__56.z);
                  __52.w = (__53.w||__56.w);
                int4 __59;
                  int4 v__48 = make_int4(2, 2, 2, 2);
                  __59.x = (__50.x+v__48.x);
                  __59.y = (__50.y+v__48.y);
                  __59.z = (__50.z+v__48.z);
                  __59.w = (__50.w+v__48.w);
                __51.x = (bool(__52.x)?__50.x:__59.x);
                __51.y = (bool(__52.y)?__50.y:__59.y);
                __51.z = (bool(__52.z)?__50.z:__59.z);
                __51.w = (bool(__52.w)?__50.w:__59.w);
                int4 v__49 = make_int4(4, 4, 4, 4);
                __49.x = (__51.x*v__49.x);
                __49.y = (__51.y*v__49.y);
                __49.z = (__51.z*v__49.z);
                __49.w = (__51.w*v__49.w);
              __34.x = (__35.x >> __49.x);
              __34.y = (__35.y >> __49.y);
              __34.z = (__35.z >> __49.z);
              __34.w = (__35.w >> __49.w);
            int4 v__50 = make_int4(15, 15, 15, 15);
            __33.x = (__34.x & v__50.x);
            __33.y = (__34.y & v__50.y);
            __33.z = (__34.z & v__50.z);
            __33.w = (__34.w & v__50.w);
          ((half2*)(&(__32.x)))->x = (half)(__33.x);
          ((half2*)(&(__32.x)))->y = (half)(__33.y);
          ((half2*)(&(__32.y)))->x = (half)(__33.z);
          ((half2*)(&(__32.y)))->y = (half)(__33.w);
          uint2 v__51 = make_uint2(__pack_half2(Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))], Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))]), __pack_half2(Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))], Scales[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))]));
          ((half2*)(&(__31.x)))->x = (((half2*)(&(__32.x)))->x*((half2*)(&(v__51.x)))->x);
          ((half2*)(&(__31.x)))->y = (((half2*)(&(__32.x)))->y*((half2*)(&(v__51.x)))->y);
          ((half2*)(&(__31.y)))->x = (((half2*)(&(__32.y)))->x*((half2*)(&(v__51.y)))->x);
          ((half2*)(&(__31.y)))->y = (((half2*)(&(__32.y)))->y*((half2*)(&(v__51.y)))->y);
        uint2 v__52 = make_uint2(__pack_half2(Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))], Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))]), __pack_half2(Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))], Zeros[((((((int)blockIdx.x) * 128) + (((int)threadIdx.z) * 16)) + (ax0_ax1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4))]));
        ((half2*)(&(__30.x)))->x = (((half2*)(&(__31.x)))->x-((half2*)(&(v__52.x)))->x);
        ((half2*)(&(__30.x)))->y = (((half2*)(&(__31.x)))->y-((half2*)(&(v__52.x)))->y);
        ((half2*)(&(__30.y)))->x = (((half2*)(&(__31.y)))->x-((half2*)(&(v__52.y)))->x);
        ((half2*)(&(__30.y)))->y = (((half2*)(&(__31.y)))->y-((half2*)(&(v__52.y)))->y);
      *(uint2*)(B_rescale_shared + (((((((k_0_0 + 1) & 1) * 9216) + (((int)threadIdx.z) * 1152)) + (ax0_ax1_fused_2_1 * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = __30;
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(APad_shared_wmma_matrix_a[ax0_0], (&(APad_shared[((((k_0_0 & 1) * 2304) + (ax0_0 * 1152)) + (k_0_1 * 16))])), 72);
      }
      nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[((((k_0_0 & 1) * 9216) + (((int)threadIdx.z) * 1152)) + (k_0_1 * 16))])), 72);
      for (int i_0_2 = 0; i_0_2 < 2; ++i_0_2) {
        nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2], APad_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2]);
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_0_1_1 = 0; k_0_1_1 < 4; ++k_0_1_1) {
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      nvcuda::wmma::load_matrix_sync(APad_shared_wmma_matrix_a[ax0_0_1], (&(APad_shared[(((ax0_0_1 * 1152) + (k_0_1_1 * 16)) + 2304)])), 72);
    }
    nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[0], (&(B_rescale_shared[(((((int)threadIdx.z) * 1152) + (k_0_1_1 * 16)) + 9216)])), 72);
    for (int i_0_2_1 = 0; i_0_2_1 < 2; ++i_0_2_1) {
      nvcuda::wmma::mma_sync(C_wmma_accumulator[i_0_2_1], APad_shared_wmma_matrix_a[i_0_2_1], B_rescale_shared_wmma_matrix_b[0], C_wmma_accumulator[i_0_2_1]);
    }
  }
  for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
    nvcuda::wmma::store_matrix_sync((&(C[(((ax0_0_2 * 286720) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 16))])), C_wmma_accumulator[ax0_0_2], 17920, nvcuda::wmma::mem_row_major);
  }
}

