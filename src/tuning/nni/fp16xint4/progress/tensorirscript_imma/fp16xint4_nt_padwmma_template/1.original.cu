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
    B_decompress = alloc_buffer(float16[16384, 16384])
     {
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          block([16384, 16384], "B_decompress") as [vi, vj] {
            bind(vi, i)
            bind(vj, j)
            tir.reads([B[vi, floordiv(vj, 2):(floordiv(vj, 2) + 2)]])
            tir.writes([B_decompress[vi, vj]])
            B_decompress[vi, vj] = select((floormod((floormod(vj, 32)*4), 8) <= 5), cast(float16, @tir.bitwise_and(@tir.shift_right(cast(int32, B[vi, ((floordiv(vj, 32)*16) + floordiv((floormod(vj, 32)*4), 8))]), floormod((floormod(vj, 32)*4), 8), dtype=int32), 15, dtype=int32)), cast(float16, @tir.bitwise_or(cast(int8, @tir.bitwise_and(@tir.shift_right(cast(int32, B[vi, ((floordiv(vj, 32)*16) + floordiv((floormod(vj, 32)*4), 8))]), floormod((floormod(vj, 32)*4), 8), dtype=int32), (@tir.shift_left(1, (8 - floormod((floormod(vj, 32)*4), 8)), dtype=int32) - 1), dtype=int32)), cast(int8, @tir.bitwise_and(@tir.bitwise_and(@tir.shift_left(cast(int32, B[vi, (((floordiv(vj, 32)*16) + floordiv((floormod(vj, 32)*4), 8)) + 1)]), (8 - floormod((floormod(vj, 32)*4), 8)), dtype=int32), @tir.shift_left(15, (8 - floormod((floormod(vj, 32)*4), 8)), dtype=int32), dtype=int32), 15, dtype=int32)), dtype=int8)))
        }
      }
      for (i_1: int32, 0, 16384) {
        for (j_1: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi_1, vj_1, vk] {
              bind(vi_1, i_1)
              bind(vj_1, j_1)
              bind(vk, k)
              tir.reads([A[vi_1, vk], B_decompress[vj_1, vk]])
              tir.writes([C[vi_1, vj_1]])
              with init() {
                C[vi_1, vj_1] = 0f16
              }
              C[vi_1, vj_1] = (C[vi_1, vj_1] + (A[vi_1, vk]*B_decompress[vj_1, vk]))
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