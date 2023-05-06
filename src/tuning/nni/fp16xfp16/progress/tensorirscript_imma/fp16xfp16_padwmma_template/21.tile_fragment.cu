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
    B_shared = alloc_buffer(float16[16384, 16384])
    A_shared_wmma.matrix_a = alloc_buffer(float16[16384, 16384])
    B_shared_wmma.matrix_b = alloc_buffer(float16[16384, 16384])
    C_wmma.accumulator = alloc_buffer(float16[16384, 16384])
    for (i_0: int32, 0, 256) "thread_binding" {
      for (j_0: int32, 0, 256) "thread_binding" {
        for (i_1_0: int32, 0, 4) "thread_binding" {
          for (j_1_0: int32, 0, 16) "thread_binding" {
            for (k_0: int32, 0, 1024) {
              for (ax0_ax1_fused_0: int32, 0, 1) {
                for (ax0_ax1_fused_1: int32, 0, 4) "thread_binding" {
                  for (ax0_ax1_fused_2: int32, 0, 16) "thread_binding" {
                    for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4: int32, 0, 8) "vectorized" {
                        block([16384, 16384], "A_shared") as [v0, v1] {
                          where((((((((((ax0_ax1_fused_0*4) + ax0_ax1_fused_1)*16) + ax0_ax1_fused_2)*32) + ax0_ax1_fused_3)*8) + ax0_ax1_fused_4) < 1024))
                          bind(v0, ((i_0*64) + floordiv((((((ax0_ax1_fused_0*16384) + (ax0_ax1_fused_1*4096)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 16)))
                          bind(v1, ((k_0*16) + floormod((((((ax0_ax1_fused_0*16384) + (ax0_ax1_fused_1*4096)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 16)))
                          tir.reads([A[v0, v1]])
                          tir.writes([A_shared[v0, v1]])
                          tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                          A_shared[v0, v1] = A[v0, v1]
                      }
                    }
                  }
                }
              }
              for (ax0_ax1_fused_0_1: int32, 0, 1) {
                for (ax0_ax1_fused_1_1: int32, 0, 4) "thread_binding" {
                  for (ax0_ax1_fused_2_1: int32, 0, 16) "thread_binding" {
                    for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4_1: int32, 0, 8) "vectorized" {
                        block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                          where((((((((((ax0_ax1_fused_0_1*4) + ax0_ax1_fused_1_1)*16) + ax0_ax1_fused_2_1)*32) + ax0_ax1_fused_3_1)*8) + ax0_ax1_fused_4_1) < 1024))
                          bind(v0_1, ((j_0*64) + floordiv((((((ax0_ax1_fused_0_1*16384) + (ax0_ax1_fused_1_1*4096)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 16)))
                          bind(v1_1, ((k_0*16) + floormod((((((ax0_ax1_fused_0_1*16384) + (ax0_ax1_fused_1_1*4096)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 16)))
                          tir.reads([B[v0_1, v1_1]])
                          tir.writes([B_shared[v0_1, v1_1]])
                          tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                          B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                      }
                    }
                  }
                }
              }
              for (k_1_0: int32, 0, 1) {
                for (ax0_0: int32, 0, 1) {
                  for (ax1_0: int32, 0, 1) {
                    for (ax0_1: int32, 0, 16) {
                      for (ax1_1: int32, 0, 16) {
                        block([16384, 16384], "A_shared_wmma.matrix_a") as [v0_2, v1_2] {
                          bind(v0_2, ((((i_0*64) + (i_1_0*16)) + (ax0_0*16)) + ax0_1))
                          bind(v1_2, (((k_0*16) + (ax1_0*16)) + ax1_1))
                          tir.reads([A_shared[v0_2, v1_2]])
                          tir.writes([A_shared_wmma.matrix_a[v0_2, v1_2]])
                          A_shared_wmma.matrix_a[v0_2, v1_2] = A_shared[v0_2, v1_2]
                      }
                    }
                  }
                }
                for (ax0_0_1: int32, 0, 1) {
                  for (ax1_0_1: int32, 0, 1) {
                    for (ax0_1_1: int32, 0, 16) {
                      for (ax1_1_1: int32, 0, 16) {
                        block([16384, 16384], "B_shared_wmma.matrix_b") as [v0_3, v1_3] {
                          where(((((j_0*64) + (j_1_0*4)) + ((ax0_0_1*16) + ax0_1_1)) < 16384))
                          bind(v0_3, ((((j_0*64) + (j_1_0*4)) + (ax0_0_1*16)) + ax0_1_1))
                          bind(v1_3, (((k_0*16) + (ax1_0_1*16)) + ax1_1_1))
                          tir.reads([B_shared[v0_3, v1_3]])
                          tir.writes([B_shared_wmma.matrix_b[v0_3, v1_3]])
                          B_shared_wmma.matrix_b[v0_3, v1_3] = B_shared[v0_3, v1_3]
                      }
                    }
                  }
                }
                for (i_1_1_0: int32, 0, 1) {
                  for (j_1_1_0: int32, 0, 1) {
                    for (i_1_1_1: int32, 0, 16) {
                      for (j_1_1_1: int32, 0, 16) {
                        for (k_1_1: int32, 0, 16) {
                          block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
                            where((((j_1_1_0*16) + j_1_1_1) < 4))
                            bind(vi, ((((i_0*64) + (i_1_0*16)) + (i_1_1_0*16)) + i_1_1_1))
                            bind(vj, (((j_0*64) + (j_1_0*4)) + ((j_1_1_0*16) + j_1_1_1)))
                            bind(vk, (((k_0*16) + (k_1_0*16)) + k_1_1))
                            tir.reads([A_shared_wmma.matrix_a[vi, vk], B_shared_wmma.matrix_b[vj, vk]])
                            tir.writes([C_wmma.accumulator[vi, vj]])
                            with init() {
                              C_wmma.accumulator[vi, vj] = 0f16
                            }
                            C_wmma.accumulator[vi, vj] = (C_wmma.accumulator[vi, vj] + (A_shared_wmma.matrix_a[vi, vk]*B_shared_wmma.matrix_b[vj, vk]))
                        }
                      }
                    }
                  }
                }
              }
            }
            for (ax0_0_2: int32, 0, 1) {
              for (ax1_0_2: int32, 0, 1) {
                for (ax0_1_2: int32, 0, 16) {
                  for (ax1_1_2: int32, 0, 16) {
                    block([16384, 16384], "C_wmma.accumulator") as [v0_4, v1_4] {
                      where(((((j_0*64) + (j_1_0*4)) + ((ax1_0_2*16) + ax1_1_2)) < 16384))
                      bind(v0_4, ((((i_0*64) + (i_1_0*16)) + (ax0_0_2*16)) + ax0_1_2))
                      bind(v1_4, ((((j_0*64) + (j_1_0*4)) + (ax1_0_2*16)) + ax1_1_2))
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