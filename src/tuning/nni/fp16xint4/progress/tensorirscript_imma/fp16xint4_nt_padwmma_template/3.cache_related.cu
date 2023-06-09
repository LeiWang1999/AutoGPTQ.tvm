#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 8192], []),
             C: Buffer(C_1: Pointer(global float16), float16, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[16384, 16384])
    A_shared_wmma.matrix_a = alloc_buffer(float16[16384, 16384])
    B_decompress_shared = alloc_buffer(float16[16384, 16384])
    B_decompress_shared_wmma.matrix_b = alloc_buffer(float16[16384, 16384])
    C_wmma.accumulator = alloc_buffer(float16[16384, 16384])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "A_shared") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([A[v0, v1]])
            tir.writes([A_shared[v0, v1]])
            A_shared[v0, v1] = A[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 16384) {
        for (ax1_1: int32, 0, 16384) {
          block([16384, 16384], "A_shared_wmma.matrix_a") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A_shared[v0_1, v1_1]])
            tir.writes([A_shared_wmma.matrix_a[v0_1, v1_1]])
            A_shared_wmma.matrix_a[v0_1, v1_1] = A_shared[v0_1, v1_1]
        }
      }
      for (ax0_2: int32, 0, 16384) {
        for (ax1_2: int32, 0, 16384) {
          block([16384, 16384], "B_decompress_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([B[v0_2, floordiv(v1_2, 2):(floordiv(v1_2, 2) + 2)]])
            tir.writes([B_decompress_shared[v0_2, v1_2]])
            B_decompress_shared[v0_2, v1_2] = select((floormod((floormod(v1_2, 32)*4), 8) <= 5), cast(float16, @tir.bitwise_and(@tir.shift_right(cast(int32, B[v0_2, ((floordiv(v1_2, 32)*16) + floordiv((floormod(v1_2, 32)*4), 8))]), floormod((floormod(v1_2, 32)*4), 8), dtype=int32), 15, dtype=int32)), cast(float16, @tir.bitwise_or(cast(int8, @tir.bitwise_and(@tir.shift_right(cast(int32, B[v0_2, ((floordiv(v1_2, 32)*16) + floordiv((floormod(v1_2, 32)*4), 8))]), floormod((floormod(v1_2, 32)*4), 8), dtype=int32), (@tir.shift_left(1, (8 - floormod((floormod(v1_2, 32)*4), 8)), dtype=int32) - 1), dtype=int32)), cast(int8, @tir.bitwise_and(@tir.bitwise_and(@tir.shift_left(cast(int32, B[v0_2, (((floordiv(v1_2, 32)*16) + floordiv((floormod(v1_2, 32)*4), 8)) + 1)]), (8 - floormod((floormod(v1_2, 32)*4), 8)), dtype=int32), @tir.shift_left(15, (8 - floormod((floormod(v1_2, 32)*4), 8)), dtype=int32), dtype=int32), 15, dtype=int32)), dtype=int8)))
        }
      }
      for (ax0_3: int32, 0, 16384) {
        for (ax1_3: int32, 0, 16384) {
          block([16384, 16384], "B_decompress_shared_wmma.matrix_b") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([B_decompress_shared[v0_3, v1_3]])
            tir.writes([B_decompress_shared_wmma.matrix_b[v0_3, v1_3]])
            B_decompress_shared_wmma.matrix_b[v0_3, v1_3] = B_decompress_shared[v0_3, v1_3]
        }
      }
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_shared_wmma.matrix_a[vi, vk], B_decompress_shared_wmma.matrix_b[vj, vk]])
              tir.writes([C_wmma.accumulator[vi, vj]])
              with init() {
                C_wmma.accumulator[vi, vj] = 0f16
              }
              C_wmma.accumulator[vi, vj] = (C_wmma.accumulator[vi, vj] + (A_shared_wmma.matrix_a[vi, vk]*B_decompress_shared_wmma.matrix_b[vj, vk]))
          }
        }
      }
      for (ax0_4: int32, 0, 16384) {
        for (ax1_4: int32, 0, 16384) {
          block([16384, 16384], "C_wmma.accumulator") as [v0_4, v1_4] {
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