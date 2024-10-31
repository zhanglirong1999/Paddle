#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.nn import Layer
from paddle.optimizer import Optimizer


class ParallelOptimizer(Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.is_initialized = False

    def __getattr__(self, item):
        return getattr(self.optimizer, item)


class ParallelBase(Layer):
    def __init__(self, model, optimizer=None):
        super().__init__()
        self.pp_parallelizer = None
        self.tp_parallelizer = None
        self.sharding_parallelizer = None

        if isinstance(model, ParallelBase):
            self.pp_parallelizer = model.pp_parallelizer
            self.tp_parallelizer = model.tp_parallelizer
            self.sharding_parallelizer = model.sharding_parallelizer
            self.model = model.model
            self.optimizer = model.optimizer
        else:
            self.model = model
            assert isinstance(optimizer, Optimizer)
            self.optimizer = ParallelOptimizer(optimizer)

        self.is_parallelized = False

    def __getattr__(self, item):
        return getattr(self.model, item)

    def parallelize_model_and_optimizer(self):
        if self.pp_parallelizer is not None:
            assert callable(self.pp_parallelizer)
            self.model = self.pp_parallelizer(self.model)

        if self.tp_parallelizer is not None:
            assert callable(self.tp_parallelizer)
            self.model = self.tp_parallelizer(self.model)

        if self.sharding_parallelizer is not None:
            assert callable(self.sharding_parallelizer)
            self.model = self.sharding_parallelizer(self.model)

        assert isinstance(self.optimizer, ParallelOptimizer)
        assert not self.optimizer.is_initialized

        # call shard optimizer here
        # self.optimizer.optimizer = ShardOptimizer()
        self.optimizer.is_initialized = True

    def forward(self, *args):
        if not self.is_parallelized:
            self.parallelize_model_and_optimizer()
            self.is_parallelized = True
        self.model(*args)
