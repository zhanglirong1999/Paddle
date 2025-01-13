# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

from paddle.base import core

from .pass_base import PassBase, register_pass

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.DENSE_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


def is_reshard_op(op):
    return op.has_attr('op_namescope') and "/auto_parallel/reshard" in op.attr(
        'op_namescope'
    )


@register_pass("auto_parallel_pipeline")
class PipelinePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self._acc_steps = self.get_attr("accumulate_steps")
        self._mode = self.get_attr("schedule_mode")
        self._gen_bsz = self.get_attr("generation_batch_size")
        self._program = main_program

        self._cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        self._nrank = len(trainer_endpoints)

        # compute current pp stage
        self._pp_stages = len(self._dist_context.process_meshes)
        self._cur_pp_stage = self._get_pp_stage(self._cur_rank)

        if self._mode == "F-Then-B":
            raise NotImplementedError("F-Then-B has not been implemented")
        else:
            raise ValueError(
                "Now only 'F-Then-B' is supported."
                f"The given value is {self._mode}."
            )

    def _insert_sync_ops_for_stream(self):
        for block in self._program.blocks:
            offset = 0
            send_vars = []
            # insert sync ops
            for index, op in enumerate(list(block.ops)):
                # NOTE: pipeline might hang when dynamic_shape is True
                if op.type in ['send_v2', 'recv_v2']:
                    op._set_attr("dynamic_shape", False)
                # set send op on comm stream
                if op.type == 'send_v2':
                    # step1: set 'use_calc_stream' False
                    op._set_attr("use_calc_stream", False)
                    op_role = op.attr('op_role')
                    # step2: insert 'c_sync_calc_stream' op before 'send_v2' op
                    var_name = op.input_arg_names[0]
                    var = block.var(var_name)
                    block._insert_op_without_sync(
                        index=index + offset,
                        type="c_sync_calc_stream",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': op_role},
                    )
                    offset += 1
                    send_vars.append(var_name)

            for var_name in send_vars:
                nop_op = block.append_op(type='nop')
                nop_op.desc.set_input('X', [var_name])
                nop_op.desc.set_output('Out', [var_name])

            block._sync_with_cpp()

    def _get_pp_stage(self, rank):
        pp_idx = None
        for idx, process_mesh in enumerate(self._dist_context.process_meshes):
            if rank in process_mesh.process_ids:
                pp_idx = idx
                break
        return pp_idx
