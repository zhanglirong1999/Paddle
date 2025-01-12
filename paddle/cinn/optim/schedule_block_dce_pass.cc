// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/schedule_block_dce_pass.h"

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/stmt_visitors.h"

namespace cinn {
namespace optim {
using ir::stmt::Alloc;
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::Free;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

class DeadStoreCollector : public ir::IRMutator<>,
                           public ir::stmt::StmtVisitor<> {
 public:
  explicit DeadStoreCollector(
      std::unordered_set<std::string>* dead_schedule_block_names,
      std::unordered_set<std::string>* output_names)
      : dead_schedule_block_names_(dead_schedule_block_names),
        output_names_(output_names) {}

  void Visit(const ir::Expr& expr) {
    ir::Expr _expr = expr;
    ir::IRMutator<>::Visit(&_expr, &_expr);
  }

  void operator()(const BlockRef& block) {
    // 1. Collecting names in load expressions.
    global_load_tensor_names.clear();
    global_load_buffer_names.clear();
    ir::stmt::StmtVisitor<>::VisitBlock(block);
    is_load_collected = true;

    // 2. Collecting dead store expressions.
    dead_schedule_block_names_->clear();
    ir::stmt::StmtVisitor<>::VisitBlock(block);
  }

 private:
  void VisitStmt(const IfThenElse& stmt) override {
    Visit(stmt->condition());
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(const For& stmt) override {
    Visit(stmt->min());
    Visit(stmt->extent());
    VisitBlock(stmt->body());
  }

  void VisitStmt(const Schedule& stmt) override {
    for (const auto& iter_value : stmt->iter_values()) {
      Visit(iter_value);
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(const Let& stmt) override {
    if (stmt->body().defined()) {
      Visit(stmt->body());
    }
  }

  void VisitStmt(const ir::stmt::Store& stmt) override {
    if (!is_load_collected) {
      Visit(stmt->value());
      return;
    }

    auto IsShareBufferWithLoadedTensor =
        [&](const ir::_Tensor_* tensor) -> bool {
      return global_load_buffer_names.count(tensor->buffer->name) > 0;
    };
    auto IsLoadedTensor = [&](const ir::_Tensor_* tensor) -> bool {
      return global_load_tensor_names.count(tensor->name) > 0;
    };
    auto IsOutputTensor = [&](const ir::_Tensor_* tensor) -> bool {
      return output_names_->count(tensor->name) > 0;
    };
    auto IsDeadStore = [&](const ir::stmt::Store& store) -> bool {
      const ir::_Tensor_* tensor = store->tensor().as_tensor();
      return !IsOutputTensor(tensor) && !IsLoadedTensor(tensor) &&
             !IsShareBufferWithLoadedTensor(tensor);
    };
    auto InsertDeadStoreName = [&](const ir::stmt::Store& store) -> void {
      if (stmt.defined() && IsDeadStore(store)) {
        VLOG(6) << "Find dead schedule block names: \n"
                << store->tensor().as_tensor()->name;
        dead_schedule_block_names_->insert(store->tensor().as_tensor()->name);
      }
    };
    InsertDeadStoreName(stmt);
  }

  void VisitStmt(const Evaluate& stmt) override {}

  void VisitStmt(const Alloc& stmt) override {}

  void VisitStmt(const Free& stmt) override {}

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    global_load_buffer_names.insert(op->tensor.as_tensor()->buffer->name);
    global_load_tensor_names.insert(op->tensor.as_tensor()->name);
  }

 private:
  bool is_load_collected = false;
  std::unordered_set<std::string> global_load_tensor_names;
  std::unordered_set<std::string> global_load_buffer_names;
  std::unordered_set<std::string>* dead_schedule_block_names_;
  std::unordered_set<std::string>* output_names_;
};

class ScheduleBlockDCE : public ir::stmt::StmtMutator<bool, bool> {
 public:
  explicit ScheduleBlockDCE(const std::vector<std::string>& output_names)
      : output_names_(output_names.begin(), output_names.end()) {}

  void operator()(BlockRef block) {
    DeadStoreCollector collector(&dead_schedule_block_names_, &output_names_);
    collector(block);
    while (!dead_schedule_block_names_.empty()) {
      VisitBlock(block);
      collector(block);
    }
  }

 private:
  bool VisitBlock(BlockRef block) override {
    const auto& stmts = block->stmts();
    std::unordered_set<int> need_remove_ids;
    for (int i = 0; i < block->stmts().size(); ++i) {
      switch (stmts[i]->stmt_type()) {
        case ir::StmtNodeTy::Schedule:
          if (VisitStmt(stmts[i].as<Schedule>())) {
            need_remove_ids.insert(i);
          }
          break;
        case ir::StmtNodeTy::IfThenElse:
          if (VisitStmt(stmts[i].as<IfThenElse>())) {
            need_remove_ids.insert(i);
          }
          break;
        case ir::StmtNodeTy::For:
          if (VisitStmt(stmts[i].as<For>())) {
            need_remove_ids.insert(i);
          }
          break;
        default:
          ir::stmt::StmtMutator<bool, bool>::VisitStmt(stmts[i]);
          break;
      }
    }

    if (!need_remove_ids.empty()) {
      std::vector<StmtRef> new_stmts;
      for (int i = 0; i < block->stmts().size(); ++i) {
        VLOG(6) << "[TEST] Remove dead schedule block: \n" << i << "\n";
        if (need_remove_ids.count(i) == 0) {
          new_stmts.push_back(block->stmts()[i]);
        }
      }
      block->set_stmts(new_stmts);
    }
  }

  bool VisitStmt(Schedule stmt) override {
    VisitBlock(stmt->body());
    return !stmt->block_fields().empty() &&
           dead_schedule_block_names_.count(stmt->name()) > 0;
  }

  bool VisitStmt(For stmt) override {
    VisitBlock(stmt->body());
    return (IsEmptyBlock(stmt->body()));
  }

  bool VisitStmt(IfThenElse stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
    return (IsEmptyIf(stmt));
  }

  bool VisitStmt(Let stmt) override {}

  bool VisitStmt(Store stmt) override {}

  bool VisitStmt(Evaluate stmt) override {}

  bool VisitStmt(Alloc stmt) override {}

  bool VisitStmt(Free stmt) override {}

  bool IsEmptyStmt(const StmtRef& stmt) {
    if (stmt->block_fields().empty()) return false;
    for (const BlockRef& block : stmt->block_fields()) {
      if (!IsEmptyBlock(block)) return false;
    }
    return true;
  }

  bool IsEmptyBlock(const BlockRef& block) {
    if (block->stmts().empty()) return false;
    for (const StmtRef& stmt : block->stmts()) {
      if (!IsEmptyStmt(stmt)) return false;
    }
    return true;
  }

  bool IsEmptyIf(const IfThenElse& stmt) {
    if (stmt->false_case().defined()) {
      return IsEmptyBlock(stmt->true_case()) &&
             IsEmptyBlock(stmt->false_case());
    }
    return IsEmptyBlock(stmt->true_case());
  }

 private:
  std::unordered_set<std::string> dead_schedule_block_names_;
  std::unordered_set<std::string> output_names_;
};

void EliminateDeadScheduleBlock(std::vector<std::string> output_names_,
                                ir::stmt::BlockRef block) {
  ScheduleBlockDCE eliminator(output_names_);
  eliminator(block);
}

}  // namespace optim
}  // namespace cinn
