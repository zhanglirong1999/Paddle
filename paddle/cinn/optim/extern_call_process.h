// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace the external call statements that are Store operations involving
 * multiple output with the external call themselves.
 *
 * This pass is applicable in scenarios where the external call statements
 * containing Store operations that have multiple output arguments. Replacing
 * them can simplify the IR.
 *
 * When applied, this pass will traverse the statements within block and
 identify
 * the external call statements which are Store operations with multiple output.
 * For each matching Store operation, this pass replaces the entire Store
 expression
 * with the external call expression itself.

 * Examples:
 * 1. Multi-Output External Call:
 *    Input IR:
 *      Store(target, Call(extern_func, args, write_args))
 *    Output IR:
 *      Call(extern_func, args, write_args)
 *
 * 2. Single-Output External Call (No Change):
 *    Input IR:
 *      Store(target, Call(extern_func, args, {}))
 *    Output IR:
 *      Store(target, Call(extern_func, args, {}))
 */
void ExternCallMultiOutputShallowStore(Expr* e);

/*
 * Remove external call statements that are TupleGet.
 *
 * This pass is applicable in scenarios where the external call statements are
 * TupleGet.
 *
 * When applied, this pass will traverse the external call statements in the
 * block and remove the statements that are TupleGet.
 */
void ExternCallRemoveTupleGetStatements(Expr* e);

}  // namespace optim
}  // namespace cinn
