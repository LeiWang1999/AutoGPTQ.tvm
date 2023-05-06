#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [128, 8192], []),
             B: Buffer(B_1: Pointer(global float16), float16, [8192, 8192], []),
             C: Buffer(C_1: Pointer(global float16), float16, [128, 8192], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[128, 8192])
    A_shared_wmma.matrix_a = alloc_buffer(float16[128, 8192])
    B_shared = alloc_buffer(float16[8192, 8192])
    B_shared_wmma.matrix_b = alloc_buffer(float16[8192, 8192])
    C_wmma.accumulator = alloc_buffer(float16[128, 8192])
     {
      for (ax0: int32, 0, 8192) {
        for (ax1: int32, 0, 8192) {
          block([8192, 8192], "B_shared") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_shared[v0, v1]])
            B_shared[v0, v1] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 128) {
        for (ax1_1: int32, 0, 8192) {
          block([128, 8192], "A_shared") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_shared[v0_1, v1_1]])
            A_shared[v0_1, v1_1] = A[v0_1, v1_1]
        }
      }
      for (ax0_2: int32, 0, 128) {
        for (ax1_2: int32, 0, 8192) {
          block([128, 8192], "A_shared_wmma.matrix_a") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([A_shared[v0_2, v1_2]])
            tir.writes([A_shared_wmma.matrix_a[v0_2, v1_2]])
            A_shared_wmma.matrix_a[v0_2, v1_2] = A_shared[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 8192) {
        for (ax1_3: int32, 0, 8192) {
          block([8192, 8192], "B_shared_wmma.matrix_b") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([B_shared[v0_3, v1_3]])
            tir.writes([B_shared_wmma.matrix_b[v0_3, v1_3]])
            B_shared_wmma.matrix_b[v0_3, v1_3] = B_shared[v0_3, v1_3]
        }
      }
      for (i: int32, 0, 128) {
        for (j: int32, 0, 8192) {
          for (k: int32, 0, 8192) {
            block([128, 8192, tir.reduce_axis(0, 8192)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_shared_wmma.matrix_a[vi, vk], B_shared_wmma.matrix_b[vj, vk]])
              tir.writes([C_wmma.accumulator[vi, vj]])
              with init() {
                C_wmma.accumulator[vi, vj] = 0f16
              }
              C_wmma.accumulator[vi, vj] = (C_wmma.accumulator[vi, vj] + (A_shared_wmma.matrix_a[vi, vk]*B_shared_wmma.matrix_b[vj, vk]))
          }
        }
      }
      for (ax0_4: int32, 0, 128) {
        for (ax1_4: int32, 0, 8192) {
          block([128, 8192], "C_wmma.accumulator") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([C_wmma.accumulator[v0_4, v1_4]])
            tir.writes([C[v0_4, v1_4]])
            C[v0_4, v1_4] = C_wmma.accumulator[v0_4, v1_4]
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