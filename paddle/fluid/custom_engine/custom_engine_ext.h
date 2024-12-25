// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#if !defined(_WIN32)
#include <cstddef>
#include <cstring>

#include "paddle/phi/backends/device_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct C_Operation_st* C_Operation;

struct C_CustomEngineInterface {
  size_t size;
  C_Status (*graph_engine_build)();
  C_Status (*graph_engine_execute)();
  C_Status (*custom_engine_op_lower)();
};

struct CustomEngineParams {
  size_t size;
  C_CustomEngineInterface* interface;
};

// Plugin implement it and fill CustomEngineParams
void InitPluginCustomEngine(CustomEngineParams*);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
