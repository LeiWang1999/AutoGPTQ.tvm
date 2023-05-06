# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 8192), "float16"), B: T.Buffer((8192, 8192), "float16"), C: T.Buffer((128, 8192), "float16")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    A_shared = T.alloc_buffer((128, 8192), "float16", scope="shared")
    A_shared_wmma_matrix_a = T.alloc_buffer((128, 8192), "float16", scope="wmma.matrix_a")
    B_shared = T.alloc_buffer((8192, 8192), "float16", scope="shared")
    B_shared_wmma_matrix_b = T.alloc_buffer((8192, 8192), "float16", scope="wmma.matrix_b")
    C_wmma_accumulator = T.alloc_buffer((128, 8192), "float16", scope="wmma.accumulator")
    for j_0_0_1 in T.thread_binding(8, thread="blockIdx.z"):
        for i_0_0 in T.thread_binding(1, thread="blockIdx.y"):
            for j_0_0_0 in T.thread_binding(4, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init in T.grid(2, 16):
                            with T.block("B_init_o"):
                                vi_o = T.axis.spatial(8, i_0_0 * 8 + i_0_1 * 2 + i_0_2_init)
                                vj_o = T.axis.spatial(512, j_0_0_0 * 128 + j_0_0_1 * 16 + j_0_1 * 16 + j_0_2_init)
                                T.reads()
                                T.writes(C_wmma_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                                C_s0 = T.int32()
                                C_s1 = T.int32()
                                C_1 = T.match_buffer(C_wmma_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16], (16, 16), "float16", strides=(C_s0, C_s1), scope="wmma.accumulator", offset_factor=16)
                                T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.float32(0))
                        for k_0_0 in T.serial(256, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 3, 2, 4], "software_pipeline_stage": [0, 0, 0, 2, 0]}):
                            for ax0_ax1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(4):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(8):
                                                with T.block("A_shared"):
                                                    v0 = T.axis.spatial(128, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 32)
                                                    v1 = T.axis.spatial(8192, k_0_0 * 32 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 1024 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 32)
                                                    T.reads(A[v0, v1])
                                                    T.writes(A_shared[v0, v1])
                                                    T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                    A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.z"):
                                    for ax0_ax1_fused_2 in range(8):
                                        for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_4 in T.vectorized(8):
                                                with T.block("B_shared"):
                                                    v0 = T.axis.spatial(8192, j_0_0_0 * 2048 + j_0_0_1 * 256 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) // 32)
                                                    v1 = T.axis.spatial(8192, k_0_0 * 32 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 2048 + ax0_ax1_fused_2 * 256 + ax0_ax1_fused_3 * 8 + ax0_ax1_fused_4) % 32)
                                                    T.reads(B[v0, v1])
                                                    T.writes(B_shared[v0, v1])
                                                    T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                    B_shared[v0, v1] = B[v0, v1]
                            for k_0_1 in T.serial(2, annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 0]}):
                                for ax0_0, ax1_0 in T.grid(2, 1):
                                    with T.block("A_shared_wmma.matrix_a_o"):
                                        v0_o = T.axis.spatial(8, i_0_1 * 2 + ax0_0)
                                        v1_o = T.axis.spatial(512, k_0_0 * 2 + k_0_1 + ax1_0)
                                        T.reads(A_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        T.writes(A_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(A_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(A_s0, A_s1), scope="shared", offset_factor=16)
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C_1 = T.match_buffer(A_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(C_s0, C_s1), scope="wmma.matrix_a", offset_factor=16)
                                        T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_s0 * 16, 1), A_s0, "row_major")
                                for ax0_0, ax1_0 in T.grid(16, 1):
                                    with T.block("B_shared_wmma.matrix_b_o"):
                                        v0_o = T.axis.spatial(512, j_0_0_0 * 128 + j_0_0_1 * 16 + ax0_0)
                                        v1_o = T.axis.spatial(512, k_0_0 * 2 + k_0_1 + ax1_0)
                                        T.reads(B_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        T.writes(B_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(B_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(A_s0, A_s1), scope="shared", offset_factor=16)
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C_1 = T.match_buffer(B_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(C_s0, C_s1), scope="wmma.matrix_b", offset_factor=16)
                                        T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_s0 * 16, 1), A_s0, "col_major")
                                for i_0_2, j_0_2 in T.grid(2, 16):
                                    with T.block("B_update_o"):
                                        vi_o = T.axis.spatial(8, i_0_0 * 8 + i_0_1 * 2 + i_0_2)
                                        vj_o = T.axis.spatial(512, j_0_0_0 * 128 + j_0_0_1 * 16 + j_0_1 * 16 + j_0_2)
                                        vk_o = T.axis.reduce(512, k_0_0 * 2 + k_0_1)
                                        T.reads(C_wmma_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16], A_shared_wmma_matrix_a[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_shared_wmma_matrix_b[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                                        T.writes(C_wmma_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                                        A_s0 = T.int32()
                                        A_s1 = T.int32()
                                        A_1 = T.match_buffer(A_shared_wmma_matrix_a[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], (16, 16), "float16", strides=(A_s0, A_s1), scope="wmma.matrix_a", offset_factor=16)
                                        B_s0 = T.int32()
                                        B_s1 = T.int32()
                                        B_1 = T.match_buffer(B_shared_wmma_matrix_b[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], (16, 16), "float16", strides=(B_s0, B_s1), scope="wmma.matrix_b", offset_factor=16)
                                        C_s0 = T.int32()
                                        C_s1 = T.int32()
                                        C_1 = T.match_buffer(C_wmma_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16], (16, 16), "float16", strides=(C_s0, C_s1), scope="wmma.accumulator", offset_factor=16)
                                        T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16, A_1.data, A_1.elem_offset // A_s0 // 16 * (A_s0 // 16) + A_1.elem_offset % A_s0 // 16, B_1.data, B_1.elem_offset // B_s0 // 16 * (B_s0 // 16) + B_1.elem_offset % B_s0 // 16, C_1.data, C_1.elem_offset // C_s0 // 16 * (C_s0 // 16) + C_1.elem_offset % C_s0 // 16)
                        for ax0_0, ax1_0 in T.grid(2, 16):
                            with T.block("C_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(8, i_0_1 * 2 + ax0_0)
                                v1_o = T.axis.spatial(512, j_0_0_0 * 128 + j_0_0_1 * 16 + ax1_0)
                                T.reads(C_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                T.writes(C[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                A_s0 = T.int32()
                                A_s1 = T.int32()
                                A_1 = T.match_buffer(C_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(A_s0, A_s1), scope="wmma.accumulator", offset_factor=16)
                                C_s0 = T.int32()
                                C_s1 = T.int32()
                                C_1 = T.match_buffer(C[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=(C_s0, C_s1), offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_s0 // 16 * (A_s0 // 16) + A_1.elem_offset % A_s0 // 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_s0 * 16, 2), C_s0, "row_major")