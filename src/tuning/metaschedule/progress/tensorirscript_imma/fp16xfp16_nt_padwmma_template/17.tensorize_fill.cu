#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global float16), float16, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[16384, 16384])
    A_shared_wmma.matrix_a = alloc_buffer(float16[16384, 16384])
    B_shared = alloc_buffer(float16[16384, 16384])
    B_shared_wmma.matrix_b = alloc_buffer(float16[16384, 16384])
    C_wmma.accumulator = alloc_buffer(float16[16384, 16384])
    for (j_0_0_1: int32, 0, 16) "thread_binding" {
      for (i_0_0: int32, 0, 128) "thread_binding" {
        for (j_0_0_0: int32, 0, 4) "thread_binding" {
          for (i_0_1: int32, 0, 2) "thread_binding" {
            for (j_0_1: int32, 0, 2) "thread_binding" {
              for (i_0_2_init: int32, 0, 4) {
                for (j_0_2_init: int32, 0, 8) {
                  block([1024, 1024], "B_init_o") as [vi_o, vj_o] {
                    bind(vi_o, (((i_0_0*8) + (i_0_1*4)) + i_0_2_init))
                    bind(vj_o, ((((j_0_0_0*256) + (j_0_0_1*16)) + (j_0_1*8)) + j_0_2_init))
                    tir.reads([])
                    tir.writes([C_wmma.accumulator[(vi_o*16):((vi_o*16) + 16), (vj_o*16):((vj_o*16) + 16)]])
                    C_2 = match_buffer(C_wmma.accumulator[(vi_o*16):((vi_o*16) + 16), (vj_o*16):((vj_o*16) + 16)])
                    @tir.tvm_fill_fragment(C_3: Pointer(wmma.accumulator float16), 16, 16, 16, ((floordiv(floordiv(elem_offset: int32, C_s0: int32), 16)*floordiv(C_s0, 16)) + floordiv(floormod(elem_offset, C_s0), 16)), 0f32, dtype=handle)
                }
              }
              for (k_0_0: int32, 0, 512) {
                for (ax0_ax1_fused_0: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_1: int32, 0, 2) "thread_binding" {
                    for (ax0_ax1_fused_2: int32, 0, 4) {
                      for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_fused_4: int32, 0, 8) "vectorized" {
                          block([16384, 16384], "A_shared") as [v0, v1] {
                            bind(v0, ((i_0_0*128) + floordiv((((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*1024)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 32)))
                            bind(v1, ((k_0_0*32) + floormod((((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*1024)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 32)))
                            tir.reads([A[v0, v1]])
                            tir.writes([A_shared[v0, v1]])
                            A_shared[v0, v1] = A[v0, v1]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_fused_0_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_1_1: int32, 0, 2) "thread_binding" {
                    for (ax0_ax1_fused_2_1: int32, 0, 8) {
                      for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_fused_4_1: int32, 0, 8) "vectorized" {
                          block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                            bind(v0_1, (((j_0_0_0*4096) + (j_0_0_1*256)) + floordiv((((((ax0_ax1_fused_0_1*4096) + (ax0_ax1_fused_1_1*2048)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 32)))
                            bind(v1_1, ((k_0_0*32) + floormod((((((ax0_ax1_fused_0_1*4096) + (ax0_ax1_fused_1_1*2048)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 32)))
                            tir.reads([B[v0_1, v1_1]])
                            tir.writes([B_shared[v0_1, v1_1]])
                            B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                        }
                      }
                    }
                  }
                }
                for (k_0_1: int32, 0, 2) {
                  for (ax0_0: int32, 0, 4) {
                    for (ax1_0: int32, 0, 1) {
                      for (ax0_1: int32, 0, 16) {
                        for (ax1_1: int32, 0, 16) {
                          block([16384, 16384], "A_shared_wmma.matrix_a") as [v0_2, v1_2] {
                            bind(v0_2, ((((i_0_0*128) + (i_0_1*64)) + (ax0_0*16)) + ax0_1))
                            bind(v1_2, ((((k_0_0*32) + (k_0_1*16)) + (ax1_0*16)) + ax1_1))
                            tir.reads([A_shared[v0_2, v1_2]])
                            tir.writes([A_shared_wmma.matrix_a[v0_2, v1_2]])
                            A_shared_wmma.matrix_a[v0_2, v1_2] = A_shared[v0_2, v1_2]
                        }
                      }
                    }
                  }
                  for (ax0_0_1: int32, 0, 8) {
                    for (ax1_0_1: int32, 0, 1) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([16384, 16384], "B_shared_wmma.matrix_b") as [v0_3, v1_3] {
                            bind(v0_3, (((((j_0_0_0*4096) + (j_0_0_1*256)) + (j_0_1*128)) + (ax0_0_1*16)) + ax0_1_1))
                            bind(v1_3, ((((k_0_0*32) + (k_0_1*16)) + (ax1_0_1*16)) + ax1_1_1))
                            tir.reads([B_shared[v0_3, v1_3]])
                            tir.writes([B_shared_wmma.matrix_b[v0_3, v1_3]])
                            B_shared_wmma.matrix_b[v0_3, v1_3] = B_shared[v0_3, v1_3]
                        }
                      }
                    }
                  }
                  for (i_0_2: int32, 0, 4) {
                    for (j_0_2: int32, 0, 8) {
                      for (i_1: int32, 0, 16) {
                        for (j_1: int32, 0, 16) {
                          for (k_1: int32, 0, 16) {
                            block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi, vj, vk] {
                              bind(vi, ((((i_0_0*128) + (i_0_1*64)) + (i_0_2*16)) + i_1))
                              bind(vj, (((((j_0_0_0*4096) + (j_0_0_1*256)) + (j_0_1*128)) + (j_0_2*16)) + j_1))
                              bind(vk, (((k_0_0*32) + (k_0_1*16)) + k_1))
                              tir.reads([C_wmma.accumulator[vi, vj], A_shared_wmma.matrix_a[vi, vk], B_shared_wmma.matrix_b[vj, vk]])
                              tir.writes([C_wmma.accumulator[vi, vj]])
                              C_wmma.accumulator[vi, vj] = (C_wmma.accumulator[vi, vj] + (A_shared_wmma.matrix_a[vi, vk]*B_shared_wmma.matrix_b[vj, vk]))
                          }
                        }
                      }
                    }
                  }
                }
              }
              for (ax0_0_2: int32, 0, 4) {
                for (ax1_0_2: int32, 0, 8) {
                  for (ax0_1_2: int32, 0, 16) {
                    for (ax1_1_2: int32, 0, 16) {
                      block([16384, 16384], "C_wmma.accumulator") as [v0_4, v1_4] {
                        bind(v0_4, ((((i_0_0*128) + (i_0_1*64)) + (ax0_0_2*16)) + ax0_1_2))
                        bind(v1_4, (((((j_0_0_0*4096) + (j_0_0_1*256)) + (j_0_1*128)) + (ax1_0_2*16)) + ax1_1_2))
                        tir.reads([C_wmma.accumulator[v0_4, v1_4]])
                        tir.writes([C[v0_4, v1_4]])
                        C[v0_4, v1_4] = C_wmma.accumulator[v0_4, v1_4]
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
}

#[metadata]
{
  "root": 1, 
  "nodes": [
    {
      "type_key": ""
    }, 
    {
      "type_key": "Map", 
      "keys": [
        "IntImm"
      ], 
      "data": [2]
    }, 
    {
      "type_key": "Array", 
      "data": [3]
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }
  ], 
  "b64ndarrays": [], 
  "attrs": {"tvm_version": "0.11.dev0"}
}