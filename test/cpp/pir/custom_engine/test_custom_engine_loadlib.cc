// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <array>
#include <string>

// #include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/custom_engine/fake_cpu_engine.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/backends/custom/fake_cpu_device.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/platform/device_context.h"

void RegisterDevice() {
  CustomRuntimeParams runtime_params;
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  InitFakeCPUDevice(&runtime_params);
  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), "", nullptr);
}

void RegisterEngine() {
  CustomEngineParams engine_params;
  std::memset(&engine_params, 0, sizeof(CustomEngineParams));
  engine_params.size = sizeof(CustomEngineParams);
  auto engine_interface = new (C_CustomEngineInterface);
  engine_params.interface = engine_interface;
  std::memset(engine_params.interface, 0, sizeof(C_CustomEngineInterface));
  engine_params.interface->size = sizeof(C_CustomEngineInterface);

  InitFakeCPUEngine(&engine_params);
  paddle::custom_engine::LoadCustomEngineLib("", &engine_params);
}

void InitCustom() {
  RegisterDevice();
  EXPECT_GT(static_cast<int>(phi::DeviceManager::GetAllDeviceTypes().size()),
            0);
  auto place = phi::CustomPlace(DEVICE_TYPE, 0);
  auto device = phi::DeviceManager::GetDeviceWithPlace(place);
  EXPECT_NE(device, nullptr);

  std::vector<phi::Place> places;
  auto device_types = phi::DeviceManager::GetAllDeviceTypes();
  for (auto dev_type : device_types) {
    auto devices = phi::DeviceManager::GetDeviceList(dev_type);
    for (auto dev_id : devices) {
      places.push_back(phi::PlaceHelper::CreatePlace(dev_type, dev_id));
    }
  }
  EXPECT_GT(static_cast<int>(places.size()), 0);

  phi::DeviceContextPool::Init(places);
  RegisterEngine();
}

TEST(CustomDevice, Tensor) {
  paddle::framework::InitMemoryMethod();
  InitCustom();

  auto* interface = paddle::custom_engine::CustomEngineManager::Instance()
                        ->GetCustomEngineInterface();
  if (interface->graph_engine_build) {
    EXPECT_EQ(interface->graph_engine_build(), C_SUCCESS);
  } else {
    FAIL() << "graph_engine_build has not register";
  }
  if (interface->graph_engine_execute) {
    EXPECT_EQ(interface->graph_engine_execute(), C_SUCCESS);
  } else {
    FAIL() << "graph_engine_execute has not register";
  }
  if (interface->custom_engine_op_lower) {
    EXPECT_EQ(interface->custom_engine_op_lower(), C_SUCCESS);
  } else {
    FAIL() << "custom_engine_op_lower has not register";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
