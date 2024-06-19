// REGISTER_IR_PASS(cpu_bfloat16_placement_pass, OneDNNPlacementBf16Pass);
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

#include "paddle/fluid/pir/transforms/onednn/onednn_placement_pass.h"

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

template <typename OpType>
class OneDNNBf16PlacementPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    // for (auto& value : op->operands_source()) {
    //   pir::Type op_dtype = pir::GetDataTypeFromValue(value);
    //    // Only float input can be converted to bfloat16
    //   if (!op_dtype.isa<pir::Float32Type>()) {
    //       return false;
    //   }
    // }
    // The pass use HasOpINT8DataType to skip int8 op
    auto op_attr = op->attributes();

    if (op_attr.find("mkldnn_data_type") != op_attr.end() &&
        op_attr.find("mkldnn_data_type")->second == "int8") {
      return false;
    }
    if (op_attr.find("use_quantizer") != op_attr.end() &&
        op_attr.find("use_quantizer")->second) {
      return false;
    }

    std::string target_op_name = op->name();
    target_op_name.replace(0, 5, "onednn_op");

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      auto yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      paddle::dialect::OpRunTimeInfo runtime_info =
          std::get<3>(yaml_interface->get_op_info_(target_op_name));
      for (auto &attr : runtime_info.extra_args_default_value) {
        if (attr.first == "mkldnn_data_type") {
          attributes[attr.first] =
              pir::StrAttribute::get(pir::IrContext::Instance(), "bfloat16");
        } else {
          attributes[attr.first] = attr.second;
        }
      }

      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

template <typename OpType>
class RemoveOrphanedPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  // find orphaned bfloat16 operator that is between two float32 operators
  // revert mkldnn_data_type attr to float32
  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    bool prev_fp32 = false;
    bool next_fp32 = false;
    if (op->num_operands()) {
      for (uint32_t i = 0; i < op->num_operands(); i++) {
        if (!op->operand_source(i) || !op->operand_source(i).type()) {
          continue;
        }
        auto *prev_op = pir::GetDefiningOpForInput(op, i);

        auto op_attr = prev_op->attributes();
        if (op_attr.find("mkldnn_data_type") != op_attr.end() &&
            op_attr.find("mkldnn_data_type")->second == "float32") {
          prev_fp32 = true;
          break;
        }
      }
    } else {
      // The first op in graph
      return false;
    }

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      auto *next_op = pir::GetDefiningOpForInput(op, i);

      auto op_next_attr = next_op->attributes();
      if (op_next_attr.find("mkldnn_data_type") != op_next_attr.end() &&
          op_next_attr.find("mkldnn_data_type")->second == "float32") {
        next_fp32 = true;
        break;
      }
    }

    if (prev_fp32 && next_fp32) {
      VLOG(10) << "RemoveOrphanedOperators";
    } else {
      return false;
    }

    std::string target_op_name = op->name();
    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();

      if (attributes.find("mkldnn_data_type") != attributes.end()) {
        attributes["mkldnn_data_type"] =
            pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
      }

      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

template <typename OpType>
class RemoveUnsupportedOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    bool unsupported_op = false;
    for (auto &value : op->operands_source()) {
      pir::Type op_dtype = pir::GetDataTypeFromValue(value);
      // Only float input can be converted to bfloat16
      if (!op_dtype.isa<pir::Float32Type>()) {
        unsupported_op = true;
      }
    }
    if (!unsupported_op) {
      return false;
    }

    std::string target_op_name = op->name();
    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      if (attributes.find("mkldnn_data_type") != attributes.end()) {
        attributes["mkldnn_data_type"] =
            pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
      }
      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

class PatternCreator {
 public:
  explicit PatternCreator(pir::IrContext *context) : context(context) {}

  template <typename Op>
  void CreateBf16PlacementPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<OneDNNBf16PlacementPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  template <typename Op>
  void CreateRemoveOrphanedPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<RemoveOrphanedPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  template <typename Op>
  void CreateRUnsupportedOpPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<RemoveUnsupportedOpPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  void ClearBenefit() { benefit = 1; }

 private:
  pir::IrContext *context;
  int benefit = 1;
};

class OneDNNPlacementBf16Pass : public pir::PatternRewritePass {
 public:
  OneDNNPlacementBf16Pass()
      : pir::PatternRewritePass("cpu_bfloat16_placement_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    PatternCreator patternCreator(context);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AbsOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Abs_Op>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::BilinearInterpOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ClipOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Clip_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ConcatOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Conv2dOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Conv3dOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::DepthwiseConv2dOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::EluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Elu_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ExpOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Exp_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FlattenOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Flatten_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::GeluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LayerNormOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LeakyReluOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LeakyRelu_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LogSoftmaxOp>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::NearestInterpOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Pad3dOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::PreluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::PriorBoxOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Relu_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Relu6Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::RoundOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Round_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ScaleOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ScaleSrOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Scale_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ScaleSr_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Sgd_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<
        paddle::dialect::SgdDenseParamSparseGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<
        paddle::dialect::SgdSparseParamSparseGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ShapeOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ShapeSrOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SigmoidOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Sigmoid_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SoftplusOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqrtOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqrtSrOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Sqrt_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqrtSr_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqueezeOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Squeeze_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::StackOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::TanhOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Tanh_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AbsGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ClipGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ClipGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ConcatGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Conv2dGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Conv3dGradOp>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::DepthwiseConv2dGradOp>(
            ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::EluGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::EluGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ExpGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ExpGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ExpandGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FlattenGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FlattenGrad_Op>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::LeakyReluGradOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::LeakyReluGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::PreluGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Relu6GradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Relu6Grad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReluGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SigmoidGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SigmoidGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqrtGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqrtGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqueezeGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqueezeGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::TanhGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::TanhGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FcOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FusionGruOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AddNOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Cast_Op>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::Conv2dTransposeOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::Conv2dTransposeBiasOp>(
            ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::DivideOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Divide_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::GaussianOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::HardswishOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LrnOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::MatmulWithFlattenOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MaxOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MeanOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MinOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MishOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MultiplyOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MultiplySrOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Multiply_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MultiplySr_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::PadOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Pool2dOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Reshape_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SliceOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Softmax_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SplitOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SplitWithNumOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SubtractOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Subtract_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SumOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SwishOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Transpose_Op>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::AddDoubleGradOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::AddDoubleGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AddGrad_Op>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::AddTripleGradOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::AddTripleGrad_Op>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::BatchNormGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::DivideGradOp>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::HardswishGradOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::HardswishGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LrnGradOp>(ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::MatmulWithFlattenGradOp>(
            ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MeanGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MishGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MishGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MultiplyGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Pool2dGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReshapeGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReshapeGrad_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SliceGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SoftmaxGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SubtractGradOp>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::SubtractGrad_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SumGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SwishGradOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SwishGrad_Op>(
        ps);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::TransposeGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::GeluGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReluGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AddOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Add_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::BatchNormOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::BatchNorm_Op>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::CastOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FullOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReshapeOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SoftmaxOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::TransposeOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AddGradOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MatmulGradOp>(
        ps);

    patternCreator.ClearBenefit();

    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AbsOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Abs_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::BilinearInterpOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ClipOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Clip_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ConcatOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Conv2dOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Conv3dOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::DepthwiseConv2dOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::EluOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Elu_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ExpOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Exp_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FlattenOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Flatten_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::GeluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LayerNormOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LeakyReluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LeakyRelu_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LogSoftmaxOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::NearestInterpOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Pad3dOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::PreluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::PriorBoxOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Relu_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Relu6Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::RoundOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Round_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ScaleOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ScaleSrOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Scale_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ScaleSr_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Sgd_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::SgdDenseParamSparseGrad_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::SgdSparseParamSparseGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ShapeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ShapeSrOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SigmoidOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Sigmoid_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SoftplusOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqrtOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqrtSrOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Sqrt_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqrtSr_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqueezeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Squeeze_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::StackOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TanhOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Tanh_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AbsGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ClipGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ClipGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ConcatGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Conv2dGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Conv3dGradOp>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::DepthwiseConv2dGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::EluGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::EluGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ExpGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ExpGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ExpandGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FlattenGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FlattenGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LeakyReluGradOp>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::LeakyReluGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::PreluGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Relu6GradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Relu6Grad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReluGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SigmoidGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SigmoidGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqrtGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqrtGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqueezeGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqueezeGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TanhGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TanhGrad_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FcOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FusionGruOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddNOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Cast_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::Conv2dTransposeBiasOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::DivideOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Divide_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::GaussianOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::HardswishOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LrnOp>(
        ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::MatmulWithFlattenOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MaxOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MeanOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MinOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MishOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MultiplyOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MultiplySrOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Multiply_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MultiplySr_Op>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::PadOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Pool2dOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Reshape_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SliceOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Softmax_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SplitOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SplitWithNumOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SubtractOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Subtract_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SumOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SwishOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Transpose_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddDoubleGradOp>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::AddDoubleGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddTripleGradOp>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::AddTripleGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::BatchNormGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::DivideGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::HardswishGradOp>(
            ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::HardswishGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LrnGradOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::MatmulWithFlattenGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MeanGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MishGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MishGrad_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MultiplyGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Pool2dGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReshapeGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReshapeGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SliceGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SoftmaxGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SubtractGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SubtractGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SumGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SwishGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SwishGrad_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TransposeGradOp>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::GeluGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReluGradOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Add_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::BatchNormOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::BatchNorm_Op>(
            ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::CastOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FullOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReshapeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SoftmaxOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TransposeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddGradOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MatmulGradOp>(
            ps);

    patternCreator.ClearBenefit();

    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AbsOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Abs_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::BilinearInterpOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ClipOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Clip_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ConcatOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Conv2dOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Conv3dOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::DepthwiseConv2dOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::EluOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Elu_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ExpOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Exp_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FlattenOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Flatten_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::GeluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LayerNormOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LeakyReluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LeakyRelu_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LogSoftmaxOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::NearestInterpOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Pad3dOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::PreluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::PriorBoxOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Relu_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Relu6Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::RoundOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Round_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ScaleOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ScaleSrOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Scale_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ScaleSr_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Sgd_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::SgdDenseParamSparseGrad_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::SgdSparseParamSparseGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ShapeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ShapeSrOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SigmoidOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Sigmoid_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SoftplusOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqrtOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqrtSrOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Sqrt_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqrtSr_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqueezeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Squeeze_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::StackOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TanhOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Tanh_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AbsGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ClipGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ClipGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ConcatGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Conv2dGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Conv3dGradOp>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::DepthwiseConv2dGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::EluGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::EluGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ExpGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ExpGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ExpandGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FlattenGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FlattenGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LeakyReluGradOp>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::LeakyReluGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::PreluGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Relu6GradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Relu6Grad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReluGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SigmoidGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SigmoidGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqrtGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqrtGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqueezeGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqueezeGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TanhGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TanhGrad_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FcOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FusionGruOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddNOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Cast_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::Conv2dTransposeBiasOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::DivideOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Divide_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::GaussianOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::HardswishOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LrnOp>(
        ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::MatmulWithFlattenOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MaxOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MeanOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MinOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MishOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MultiplyOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MultiplySrOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Multiply_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MultiplySr_Op>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::PadOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Pool2dOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Reshape_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SliceOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Softmax_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SplitOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SplitWithNumOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SubtractOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Subtract_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SumOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SwishOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Transpose_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddDoubleGradOp>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::AddDoubleGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddTripleGradOp>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::AddTripleGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::BatchNormGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::DivideGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::HardswishGradOp>(
            ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::HardswishGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LrnGradOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::MatmulWithFlattenGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MeanGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MishGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MishGrad_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MultiplyGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Pool2dGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReshapeGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReshapeGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SliceGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SoftmaxGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SubtractGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SubtractGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SumGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SwishGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SwishGrad_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TransposeGradOp>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::GeluGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReluGradOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Add_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::BatchNormOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::BatchNorm_Op>(
            ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::CastOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FullOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReshapeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SoftmaxOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TransposeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddGradOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MatmulGradOp>(
            ps);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16PlacementPass() {
  return std::make_unique<OneDNNPlacementBf16Pass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_placement_pass, OneDNNPlacementBf16Pass);
