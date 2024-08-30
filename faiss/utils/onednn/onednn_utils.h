/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <pthread.h>
#include "oneapi/dnnl/dnnl.hpp"
#include <faiss/impl/ResultHandler.h>
#include <mm_malloc.h>

#include <sys/syscall.h>
#include <unistd.h>
#include <immintrin.h>

#include <iostream>

namespace faiss {
static dnnl::engine cpu_engine;
static dnnl::stream engine_stream;
static bool is_onednn_init = false;
static std::mutex init_mutex;

enum DNNL_STATE {
  DNNL_UNSUPPORTED = false,
  DNNL_SUPPORTED = true,  
  DNNL_UNKOWN = 99
};

static DNNL_STATE dnnl_state = DNNL_STATE::DNNL_UNKOWN;
static bool is_dnnl_enabled() {
  if (dnnl_state == DNNL_STATE::DNNL_UNKOWN) [[unlikely]]  {
    char* env = getenv("DNNL_ENABLE");

	std::cout << "is_dnnl_enabled DNNL_ENABLE=" << env << "\n";

    if (env != NULL && strcmp(env, "1") == 0) {
      dnnl_state = DNNL_STATE::DNNL_SUPPORTED;
    } else {
      dnnl_state = DNNL_STATE::DNNL_UNSUPPORTED;
    } 
  } 
  return dnnl_state;  
}

static void init_onednn() {
  // printf("try init init_onednn\n");
  std::unique_lock<std::mutex> lock(init_mutex);
  
  if (is_onednn_init) {
    return;
  }

  printf("init onednn\n");

  // init onednn env 
  cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  engine_stream = dnnl::stream(cpu_engine);

  is_onednn_init = true;
}

/**
 * @brief Compute float32 matrix inner product with bf16 intermediate results to accelerate
 * @details The main idea is:  
 * 1. Define float32 memory layout for input and output
 * 2. Create low precision bf16 memory descriptors as inner product input
 * 3. Generate inner product primitive descriptor
 * 4. Execute float32 => (reorder) => bf16 => (inner product) => float32
 *    chain operation, isolate different precision data, accelerate inner product 
 * 5. Pipeline execution via streams for asynchronous scheduling
 *
 * @param xrow Row number of input matrix X  
 * @param xcol Column number of input matrix X
 * @param yrow Row number of weight matrix Y
 * @param ycol Column number of weight matrix Y
 * @param in_f32_1 Input matrix pointer in float32 type 
 * @param in_f32_2 Weight matrix pointer in float32 type
 * @param out_f32 Output matrix pointer for result in float32 type
 * @return None
 */

static void comput_f32bf16f32_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    float* in_f32_1, float* in_f32_2, float** out_f32) {
	dnnl::inner_product_forward::primitive_desc inner_product_pd;
    dnnl::inner_product_forward inner_product_prim;

    dnnl::memory::desc f32_md1;
    dnnl::memory::desc f32_md2;
    dnnl::memory::desc f32_dst_md2;
    dnnl::memory f32_mem1;
    dnnl::memory f32_mem2;
    dnnl::memory f32_dst_mem;

    dnnl::memory::desc bf16_md1;
    dnnl::memory::desc bf16_md2;
    dnnl::memory bf16_mem1;
    dnnl::memory bf16_mem2;

	std::cout << "comput_f32bf16f32_inner_product xrow=" << xrow << "  xcol=" << xcol << "  yrow=" << yrow << "  ycol=" << ycol << "\n";

    f32_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    f32_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

        f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
    f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
    f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);

    // inner memory bf16
    bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
    bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);

    inner_product_pd = dnnl::inner_product_forward::primitive_desc(
          cpu_engine, dnnl::prop_kind::forward_training,
          bf16_md1, bf16_md2, f32_dst_md2);

    inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

    bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine);
    dnnl::reorder(f32_mem1, bf16_mem1).execute(engine_stream, f32_mem1, bf16_mem1);

    bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
    dnnl::reorder(f32_mem2, bf16_mem2).execute(engine_stream, f32_mem2, bf16_mem2);

    inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
            {DNNL_ARG_WEIGHTS, bf16_mem2},
            {DNNL_ARG_DST, f32_dst_mem}});

}

static void comput_f32bf16f32_inner_product_blas(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    float* in_f32_1, float* in_f32_2, float* out_f32) {
	//std::cout << "comput_f32bf16f32_inner_product_blas  xrow=" << xrow << "  xcol=" << xcol << "  yrow=" << yrow << "  ycol=" << ycol << "\n";

  dnnl::memory::desc f32_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  dnnl::memory f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
  dnnl::memory f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
  dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32); 

  // inner memory bf16
  dnnl::memory::desc bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
  dnnl::memory::desc bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any); 

  
  dnnl::inner_product_forward::primitive_desc inner_product_pd = dnnl::inner_product_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_training,
      bf16_md1, bf16_md2, f32_dst_md2);

  dnnl::inner_product_forward inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

  dnnl::memory bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine);
  dnnl::reorder(f32_mem1, bf16_mem1).execute(engine_stream, f32_mem1, bf16_mem1);
 
  dnnl::memory bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
  dnnl::reorder(f32_mem2, bf16_mem2).execute(engine_stream, f32_mem2, bf16_mem2);

  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                             {DNNL_ARG_WEIGHTS, bf16_mem2},
                                             {DNNL_ARG_DST, f32_dst_mem}});
 
 
  // printf("comput_f32bf16f32_inner_product_blas finished#######>\n");
}

static void comput_f16f16f32_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    uint16_t* in_f16_1, uint16_t* in_f16_2, float* out_f32) {

  dnnl::memory::desc f16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::any);
  dnnl::memory::desc f16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::any);
  dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  // dnnl::memory f16_mem1 = dnnl::memory(f16_md1, cpu_engine, in_f32_1);
  // dnnl::memory f16_mem2 = dnnl::memory(f16_md2, cpu_engine, in_f32_2);
  dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32); 

  // // inner memory bf16
  // dnnl::memory::desc f16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::any);
  // dnnl::memory::desc f16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::any); 

  
  dnnl::inner_product_forward::primitive_desc inner_product_pd = dnnl::inner_product_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_training,
      f16_md1, f16_md2, f32_dst_md2);

  dnnl::inner_product_forward inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

  dnnl::memory f16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine, in_f16_1);
  dnnl::memory f16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine, in_f16_2);

  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, f16_mem1},
                                             {DNNL_ARG_WEIGHTS, f16_mem2},
                                             {DNNL_ARG_DST, f32_dst_mem}});
 

 
  // printf("comput_f32bf16f32_inner_product finished#######>\n");
}

static void comput_bf16bf16f32_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    uint16_t* in_bf16_1, uint16_t* in_bf16_2, float* out_f32) {

  dnnl::memory::desc bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
  dnnl::memory::desc bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
  dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32); 
  
  dnnl::inner_product_forward::primitive_desc inner_product_pd = dnnl::inner_product_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_training,
      bf16_md1, bf16_md2, f32_dst_md2);

  dnnl::inner_product_forward inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

  dnnl::memory bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine, in_bf16_1);
  dnnl::memory bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine, in_bf16_2);

  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                             {DNNL_ARG_WEIGHTS, bf16_mem2},
                                             {DNNL_ARG_DST, f32_dst_mem}});
 
 
  // printf("comput_f32bf16f32_inner_product finished#######>\n");
}

// #define MAX_ROWS 16
// #define MAX_COLS 64
// uint64_t STRIDE = 64;
// // #define STRIDE 64
// #define ARCH_GET_XCOMP_PERM 0x1022
// #define ARCH_REQ_XCOMP_PERM 0x1023
// #define XFEATURE_XTILECFG 17
// #define XFEATURE_XTILEDATA 18

// enum AMX_STATE {
//   AMX_UNSUPPORTED = false,
//   AMX_SUPPORTED = true,  
//   AMX_UNKOWN = 99
// };

// // Define tile config data structure
// typedef struct __tile_config
// {
//     uint8_t palette_id;
//     uint8_t start_row;
//     uint8_t reserved_0[14];
//     uint16_t colsb[16];
//     uint8_t rows[16];
// } __tilecfg;

// static AMX_STATE amx_state = AMX_STATE::AMX_UNKOWN;

// static bool is_amx_enabled() {
//   if (amx_state == AMX_STATE::AMX_UNKOWN) [[unlikely]]  {
//     char* env = getenv("AMX_ENABLE");
//     if (env != NULL && strcmp(env, "1") == 0) {
//       amx_state = AMX_STATE::AMX_SUPPORTED;
//     } else {
//       amx_state = AMX_STATE::AMX_UNSUPPORTED;
//     } 
//   } 
//   return amx_state;  
// }

// static __tilecfg tile_data = {0}; 

// /* Initialize tile config */
// static void init_tile_config(__tilecfg *tileinfo, uint16_t rows, uint16_t cols)
// {
//     int i;
//     tileinfo->palette_id = 1;
//     tileinfo->start_row = 0;

//     for (i = 0; i < 1; ++i)
//     {
//         tileinfo->colsb[i] = rows;
//         tileinfo->rows[i] = rows;
//     }

//     for (i = 1; i < 4; ++i)
//     {
//         tileinfo->colsb[i] = cols;
//         tileinfo->rows[i] = rows;
//     }

//     _tile_loadconfig(tileinfo);
// }

// /* Initialize int8_t buffer */
// static void init_buffer(float *buf, float value, uint32_t size)
// {
//     int i;

//     for (i = 0; i < size; i++)
//         buf[i] = value;
// }

// /* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
// static bool set_tiledata_use()
// {
//     if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
//     {
//         printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
//         return false;
//     }
//     else
//     {
//         printf("\n TILE DATA USE SET - OK \n\n");
//         return true;
//     }

//     return true;
// }

// static void init_AMX_tile_() {
//     if (!set_tiledata_use())
//         exit(-1);

//     // Load tile configuration
//     init_tile_config(&tile_data, MAX_ROWS, MAX_COLS);  
// }

// static void print_buffer(const float *buf, int32_t rows, int32_t colsb)
// {
//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < (colsb); j++)
//         {
//             printf("%.2f ", buf[i * colsb + j]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

// // #define DIM_T 1024

// float amx_inner_product(const float *src1, const float *src2, size_t DIM) {
//     printf("src1:\n");
//     print_buffer(src1, 1, DIM);
//     printf("src2:\n");
//     print_buffer(src2, 1, DIM);
    
//     uint16_t *src1_u16 = new uint16_t[DIM];
//     uint16_t *src2_u16 = new uint16_t[DIM];    

//     float res_tmp[256];

//     cvt_float_to_bfloat16(src1,src1_u16, DIM);
//     cvt_float_to_bfloat16(src2,src2_u16, DIM);

//     init_buffer(res_tmp, (float)0.0, 256);
//     _tile_zero(1);
    
//     for (int i = 0; i < DIM; i += 32) {
//         _tile_loadd(2, src1_u16, STRIDE);
//         _tile_loadd(3, src2_u16, STRIDE);
//         _tile_dpbf16ps(1, 2, 3);
//         _tile_stored(1, res_tmp, STRIDE);
//     }

//     print_buffer(res_tmp, 16 ,16);
        
//     return res_tmp[0];   
// }

__attribute__((constructor))
static void library_load() {
    // 这个函数会在库加载时自动调用
    // printf("Library loaded.\n");
    init_onednn();
    // init_AMX_tile_();
    is_dnnl_enabled();
}

}//namespace faiss

