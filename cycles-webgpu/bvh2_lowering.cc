/* SPDX-License-Identifier: Apache-2.0 */

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

/* Stock film kernels address a float array. Translate its tagged pointers to
 * 32-bit word indices in the HLSL storage buffer before SPIR-V lowering. */
struct LowerFilmStorage : PassInfoMixin<LowerFilmStorage> {
  static constexpr unsigned film_space = 31;

  static Value *word_index(Value *pointer, const DataLayout &layout, IRBuilder<> &builder)
  {
    if (const auto *global = dyn_cast<GlobalVariable>(pointer)) {
      if (global->getName() == "cycles_adaptive_film") {
        return builder.getInt32(0);
      }
    }
    if (auto *gep = dyn_cast<GEPOperator>(pointer)) {
      SmallMapVector<Value *, APInt, 4> offsets;
      APInt fixed(32, 0);
      if (!gep->collectOffset(layout, 32, offsets, fixed) || fixed.urem(4) != 0) {
        report_fatal_error("Cannot lower unaligned stock film address");
      }

      Value *index = word_index(gep->getPointerOperand(), layout, builder);
      index = builder.CreateAdd(index, builder.getInt32(fixed.sdiv(4).getZExtValue()));
      for (const auto &[offset, stride] : offsets) {
        if (stride.urem(4) != 0) {
          report_fatal_error("Cannot lower unaligned stock film stride");
        }
        Value *word = builder.CreateIntCast(offset, builder.getInt32Ty(), true);
        word = builder.CreateMul(word, builder.getInt32(stride.sdiv(4).getZExtValue()));
        index = builder.CreateAdd(index, word);
      }
      return index;
    }
    if (auto *select = dyn_cast<SelectInst>(pointer)) {
      return builder.CreateSelect(select->getCondition(),
                                  word_index(select->getTrueValue(), layout, builder),
                                  word_index(select->getFalseValue(), layout, builder));
    }
    report_fatal_error("Unexpected pointer in stock film storage");
  }

  PreservedAnalyses run(Module &module, ModuleAnalysisManager &)
  {
    Function *read = module.getFunction("_Z25cycles_adaptive_film_readj");
    Function *write = module.getFunction("_Z26cycles_adaptive_film_writejj");
    if (!read || !write) {
      report_fatal_error("Stock film storage accessors are missing");
    }

    for (Function &function : module) {
      for (BasicBlock &block : function) {
        for (Instruction &instruction : llvm::make_early_inc_range(block)) {
          auto *load = dyn_cast<LoadInst>(&instruction);
          auto *store = dyn_cast<StoreInst>(&instruction);
          Value *pointer = load ? load->getPointerOperand() :
                           store ? store->getPointerOperand() :
                                   nullptr;
          if (!pointer || pointer->getType()->getPointerAddressSpace() != film_space) {
            continue;
          }

          IRBuilder<> builder(&instruction);
          Value *index = word_index(pointer, module.getDataLayout(), builder);
          Type *word = builder.getInt32Ty();
          if (load) {
            CallInst *value = builder.CreateCall(read, {index});
            value->setCallingConv(read->getCallingConv());
            instruction.replaceAllUsesWith(builder.CreateBitCast(value, load->getType()));
          }
          else {
            Value *value = builder.CreateBitCast(store->getValueOperand(), word);
            builder.CreateCall(write, {index, value})->setCallingConv(write->getCallingConv());
          }
          instruction.eraseFromParent();
        }
      }
    }
    read->setLinkage(GlobalValue::InternalLinkage);
    write->setLinkage(GlobalValue::InternalLinkage);
    return PreservedAnalyses::none();
  }
};

CallInst *resource_pointer(Value *value)
{
  auto *call = dyn_cast<CallInst>(value);
  return call && call->getCalledFunction() &&
                 call->getCalledFunction()->getName().starts_with("llvm.spv.resource.getpointer.") ?
             call :
             nullptr;
}

/* Carry a binding index instead of a storage pointer across control flow.
 * The dispatch loop may store the index, but WGSL cannot store a pointer. */
bool rematerialize_resource_pointers(Function &function)
{
  SmallVector<PHINode *, 8> phis;
  for (BasicBlock &block : function) {
    for (PHINode &phi : block.phis()) {
      if (phi.getType()->isPointerTy()) {
        phis.push_back(&phi);
      }
    }
  }

  bool changed = false;
  for (PHINode *phi : phis) {
    auto *base = resource_pointer(phi->getIncomingValue(0));
    if (!base || !llvm::all_of(phi->incoming_values(), [&](Value *value) {
          auto *call = resource_pointer(value);
          return call && call->getCalledFunction() == base->getCalledFunction() &&
                 call->getArgOperand(0) == base->getArgOperand(0);
        }))
    {
      continue;
    }
    auto *index = PHINode::Create(base->getArgOperand(1)->getType(),
                                  phi->getNumIncomingValues(),
                                  "binding.index",
                                  phi->getIterator());
    for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
      index->addIncoming(resource_pointer(phi->getIncomingValue(i))->getArgOperand(1),
                         phi->getIncomingBlock(i));
    }
    IRBuilder<> builder(phi->getParent(), phi->getParent()->getFirstNonPHIIt());
    Value *pointer = builder.CreateCall(base->getCalledFunction(), {base->getArgOperand(0), index});
    phi->replaceAllUsesWith(pointer);
    phi->eraseFromParent();
    changed = true;
  }

  SmallVector<CallInst *, 32> pointers;
  for (BasicBlock &block : function) {
    for (Instruction &instruction : block) {
      if (auto *call = resource_pointer(&instruction)) {
        pointers.push_back(call);
      }
    }
  }
  for (CallInst *pointer : pointers) {
    for (Use &use : llvm::make_early_inc_range(pointer->uses())) {
      auto *user = cast<Instruction>(use.getUser());
      if (isa<PHINode>(user) || user->getParent() == pointer->getParent()) {
        continue;
      }
      Instruction *local = pointer->clone();
      local->insertBefore(user->getIterator());
      use.set(local);
      changed = true;
    }
    if (pointer->use_empty()) {
      pointer->eraseFromParent();
      changed = true;
    }
  }
  return changed;
}

/* HLSL lazily acquires bindings where they are first used. Constant handles
 * must dominate the dispatch loop to keep opaque resources out of its state. */
struct HoistResourceHandles : PassInfoMixin<HoistResourceHandles> {
  PreservedAnalyses run(Function &function, FunctionAnalysisManager &)
  {
    BasicBlock &entry = function.getEntryBlock();
    SmallVector<CallInst *, 4> handles;
    for (BasicBlock &block : function) {
      if (&block == &entry) {
        continue;
      }
      for (Instruction &instruction : block) {
        auto *call = dyn_cast<CallInst>(&instruction);
        if (call && call->getCalledFunction() &&
            call->getCalledFunction()->getName().starts_with(
                "llvm.spv.resource.handlefrombinding.") &&
            llvm::all_of(call->args(), [](const Use &arg) { return isa<Constant>(arg.get()); }))
        {
          handles.push_back(call);
        }
      }
    }
    for (auto *handle : handles) {
      handle->moveBefore(entry.getTerminator()->getIterator());
    }
    const bool rematerialized = rematerialize_resource_pointers(function);
    return handles.empty() && !rematerialized ? PreservedAnalyses::all() :
                                               PreservedAnalyses::none();
  }
};

bool type_array_accesses(Function &function)
{
  SmallVector<Instruction *, 16> accesses;
  for (BasicBlock &block : function) {
    for (Instruction &instruction : block) {
      if (isa<GetElementPtrInst, LoadInst, StoreInst>(instruction)) {
        accesses.push_back(&instruction);
      }
    }
  }
  bool changed = false;
  for (Instruction *instruction : accesses) {
    IRBuilder<> builder(instruction);
    Value *pointer;
    Type *element;
    Value *offset = builder.getInt32(0);
    if (auto *gep = dyn_cast<GetElementPtrInst>(instruction)) {
      if (gep->getNumIndices() != 1)
        continue;
      pointer = gep->getPointerOperand();
      element = gep->getSourceElementType();
      offset = *gep->idx_begin();
    }
    else if (auto *load = dyn_cast<LoadInst>(instruction)) {
      pointer = load->getPointerOperand();
      element = load->getType();
    }
    else {
      auto *store = cast<StoreInst>(instruction);
      pointer = store->getPointerOperand();
      element = store->getValueOperand()->getType();
    }
    if (auto *extension = dyn_cast<CastInst>(offset);
        extension && isa<SExtInst, ZExtInst>(extension))
    {
      offset = extension->getOperand(0);
    }
    if (auto *constant = dyn_cast<ConstantInt>(offset)) {
      if (constant->getValue().isSignedIntN(32)) {
        offset = builder.getInt32(constant->getSExtValue());
      }
    }
    Value *array_base = pointer;
    SmallVector<Value *, 3> indices{builder.getInt32(0)};
    ArrayType *array = nullptr;
    if (auto *allocation = dyn_cast<AllocaInst>(pointer)) {
      array = dyn_cast<ArrayType>(allocation->getAllocatedType());
      if (!array || array->getElementType() != element)
        continue;
    }
    else {
      /* LLVM folds a constant row in a two-dimensional lookup table into a byte
       * offset. Restore typed indices so SPIR-V can read the private array. */
      auto *row = dyn_cast<GEPOperator>(pointer);
      auto *global = row ? dyn_cast<GlobalVariable>(row->getPointerOperand()) : nullptr;
      if (!global || !global->isConstant() || !offset->getType()->isIntegerTy(32))
        continue;
      array = dyn_cast<ArrayType>(global->getValueType());
      auto *row_type = array ? dyn_cast<ArrayType>(array->getElementType()) : nullptr;
      if (!row_type || row_type->getElementType() != element)
        continue;

      const auto &layout = function.getParent()->getDataLayout();
      APInt bytes(layout.getIndexSizeInBits(row->getPointerAddressSpace()), 0);
      const uint64_t row_size = layout.getTypeAllocSize(row_type).getFixedValue();
      if (!row_size || !row->accumulateConstantOffset(layout, bytes) || bytes.isNegative() ||
          bytes.getZExtValue() % row_size ||
          bytes.getZExtValue() / row_size >= array->getNumElements())
      {
        continue;
      }

      array_base = global;
      indices.push_back(builder.getInt32(bytes.getZExtValue() / row_size));
    }
    indices.push_back(offset);
    auto *typed = builder.Insert(GetElementPtrInst::Create(array, array_base, indices),
                                 "typed.array.element");
    if (auto *gep = dyn_cast<GetElementPtrInst>(instruction)) {
      gep->replaceAllUsesWith(typed);
      gep->eraseFromParent();
    }
    else if (auto *load = dyn_cast<LoadInst>(instruction)) {
      load->setOperand(load->getPointerOperandIndex(), typed);
    }
    else {
      auto *store = cast<StoreInst>(instruction);
      store->setOperand(store->getPointerOperandIndex(), typed);
    }
    changed = true;
  }
  return changed;
}

ArrayType *local_array_element_address(Value *pointer)
{
  auto *address = dyn_cast<GetElementPtrInst>(pointer);
  if (!address || address->getNumIndices() != 2 ||
      !isa<AllocaInst>(address->getPointerOperand()))
  {
    return nullptr;
  }
  return dyn_cast<ArrayType>(address->getSourceElementType());
}

bool rebase_array_byte_offsets(Function &function)
{
  SmallVector<GetElementPtrInst *, 8> offsets;
  for (BasicBlock &block : function) {
    for (Instruction &instruction : block) {
      auto *address = dyn_cast<GetElementPtrInst>(&instruction);
      if (address && address->getSourceElementType()->isIntegerTy(8) &&
          address->getNumIndices() == 1 &&
          local_array_element_address(address->getPointerOperand()))
      {
        offsets.push_back(address);
      }
    }
  }

  bool changed = false;
  const auto &layout = function.getParent()->getDataLayout();
  for (auto *address : offsets) {
    auto *bytes = dyn_cast<ConstantInt>(*address->idx_begin());
    auto *element = cast<GetElementPtrInst>(address->getPointerOperand());
    auto *array = cast<ArrayType>(element->getSourceElementType());
    Value *first = element->getOperand(1);
    Value *index = element->getOperand(2);
    const auto element_size = layout.getTypeAllocSize(array->getElementType());
    if (!bytes || !bytes->getValue().isSignedIntN(32) || element_size.isScalable() ||
        !element_size.getFixedValue() || !index->getType()->isIntegerTy(32))
    {
      continue;
    }
    const int64_t stride = element_size.getFixedValue();
    const int64_t byte_offset = bytes->getSExtValue();
    if (byte_offset % stride) {
      continue;
    }
    /* InstCombine may express adjacent stack words as a byte offset from an
     * element. SPIR-V requires the full typed access chain to the array. */
    IRBuilder<> builder(address);
    Value *adjusted = builder.CreateAdd(index, builder.getInt32(byte_offset / stride));
    Value *typed = builder.CreateGEP(array, element->getPointerOperand(), {first, adjusted});
    address->replaceAllUsesWith(typed);
    address->eraseFromParent();
    changed = true;
  }
  return changed;
}

bool is_word_buffer_pointer(Value *pointer)
{
  auto *call = dyn_cast<CallInst>(pointer);
  if (!call || !call->getCalledFunction() ||
      !call->getCalledFunction()->getName().starts_with("llvm.spv.resource.getpointer."))
  {
    return false;
  }
  auto *storage = dyn_cast<TargetExtType>(call->getArgOperand(0)->getType());
  auto *array = storage && storage->getName() == "spirv.VulkanBuffer" ?
                    dyn_cast<ArrayType>(storage->getTypeParameter(0)) :
                    nullptr;
  return array && array->getElementType()->isIntegerTy(32);
}

Type *scalar_word_address_type(Value *pointer)
{
  if (is_word_buffer_pointer(pointer)) {
    return Type::getInt32Ty(pointer->getContext());
  }
  auto *array = local_array_element_address(pointer);
  Type *element = array ? array->getElementType() : nullptr;
  return element && (element->isFloatTy() || element->isIntegerTy(32)) ? element : nullptr;
}

bool needs_scalar_bitcast(Value *pointer, Type *access_type)
{
  if (access_type->isFloatTy() || access_type->isIntegerTy(32)) {
    Type *expected_type = scalar_word_address_type(pointer);
    return expected_type && expected_type != access_type;
  }
  return false;
}

bool can_widen_int_compare(const ICmpInst &compare)
{
  const Type *type = compare.getOperand(0)->getType();
  if (!type->isIntegerTy(8) && !type->isIntegerTy(16)) {
    return false;
  }
  return llvm::all_of(compare.operands(), [](const Use &operand) {
    const auto *truncate = dyn_cast<TruncInst>(operand.get());
    return isa<ConstantInt>(operand.get()) ||
           (truncate && truncate->getSrcTy()->isIntegerTy(32));
  });
}

Value *widen_int_compare(IRBuilder<> &builder, const ICmpInst &compare)
{
  const bool is_signed = compare.isSigned();
  auto widen = [&](Value *operand) -> Value * {
    if (const auto *constant = dyn_cast<ConstantInt>(operand)) {
      return builder.getInt32(is_signed ? constant->getSExtValue() : constant->getZExtValue());
    }
    auto *truncate = cast<TruncInst>(operand);
    const uint32_t bits = truncate->getType()->getIntegerBitWidth();
    Value *word = truncate->getOperand(0);
    if (is_signed) {
      return builder.CreateAShr(builder.CreateShl(word, 32 - bits), 32 - bits);
    }
    return builder.CreateAnd(word, (1u << bits) - 1);
  };
  return builder.CreateICmp(compare.getPredicate(),
                            widen(compare.getOperand(0)),
                            widen(compare.getOperand(1)));
}

const ConstantFP *infinity_compare_literal(const FCmpInst &compare)
{
  if (compare.getPredicate() != FCmpInst::FCMP_ONE &&
      compare.getPredicate() != FCmpInst::FCMP_UEQ)
  {
    return nullptr;
  }
  const auto *infinity = dyn_cast<ConstantFP>(compare.getOperand(1));
  return infinity && infinity->getValueAPF().isInfinity() ? infinity : nullptr;
}

bool lower_spirv_values(Function &function)
{
  SmallVector<Instruction *, 32> replacements;
  for (BasicBlock &block : function) {
    for (Instruction &instruction : block) {
      if (auto *load = dyn_cast<LoadInst>(&instruction);
          load && needs_scalar_bitcast(load->getPointerOperand(), load->getType()))
      {
        replacements.push_back(load);
      }
      if (auto *store = dyn_cast<StoreInst>(&instruction);
          store &&
          needs_scalar_bitcast(store->getPointerOperand(), store->getValueOperand()->getType()))
      {
        replacements.push_back(store);
      }
      if (auto *call = dyn_cast<IntrinsicInst>(&instruction);
          call && ((call->getIntrinsicID() == Intrinsic::copysign && call->getType()->isFloatTy()) ||
                   (call->getIntrinsicID() == Intrinsic::ctlz && call->getType()->isIntegerTy(32))))
      {
        replacements.push_back(call);
      }
      if (auto *comparison = dyn_cast<ICmpInst>(&instruction);
          comparison && can_widen_int_compare(*comparison))
      {
        replacements.push_back(comparison);
      }
      if (auto *comparison = dyn_cast<FCmpInst>(&instruction);
          comparison && comparison->getOperand(0)->getType()->isFloatTy() &&
          (comparison->getPredicate() == FCmpInst::FCMP_ORD ||
           comparison->getPredicate() == FCmpInst::FCMP_UNO ||
           infinity_compare_literal(*comparison)))
      {
        replacements.push_back(comparison);
      }
    }
  }
  for (Instruction *instruction : replacements) {
    IRBuilder<> builder(instruction);
    if (auto *load = dyn_cast<LoadInst>(instruction)) {
      auto *word = builder.CreateAlignedLoad(
          scalar_word_address_type(load->getPointerOperand()),
          load->getPointerOperand(),
          load->getAlign(),
          load->isVolatile());
      load->replaceAllUsesWith(builder.CreateBitCast(word, load->getType()));
    }
    else if (auto *store = dyn_cast<StoreInst>(instruction)) {
      auto *word = builder.CreateBitCast(store->getValueOperand(),
                                         scalar_word_address_type(store->getPointerOperand()));
      builder.CreateAlignedStore(
          word, store->getPointerOperand(), store->getAlign(), store->isVolatile());
    }
    else if (auto *comparison = dyn_cast<ICmpInst>(instruction)) {
      /* WGSL exposes 32-bit integers. Restore the mask InstCombine folded
       * into a comparison of packed 8- or 16-bit fields. */
      comparison->replaceAllUsesWith(widen_int_compare(builder, *comparison));
    }
    else if (auto *comparison = dyn_cast<FCmpInst>(instruction)) {
      auto magnitude_bits = [&](Value *value) {
        Value *bits = builder.CreateBitCast(value, builder.getInt32Ty());
        return builder.CreateAnd(bits, builder.getInt32(0x7fffffff));
      };
      auto *infinity = builder.getInt32(0x7f800000);
      if (const auto *literal = infinity_compare_literal(*comparison)) {
        /* InstCombine recognizes the integer finite test and inserts a float
         * infinity literal, which WGSL does not permit. Restore a word test. */
        Value *bits = builder.CreateBitCast(comparison->getOperand(0), builder.getInt32Ty());
        const uint32_t literal_bits = literal->getValueAPF().isNegative() ? 0xff800000 : 0x7f800000;
        Value *is_infinity = builder.CreateICmpEQ(bits, builder.getInt32(literal_bits));
        Value *is_nan = builder.CreateICmpUGT(
            builder.CreateAnd(bits, builder.getInt32(0x7fffffff)), infinity);
        Value *non_finite = builder.CreateOr(is_infinity, is_nan);
        comparison->replaceAllUsesWith(comparison->getPredicate() == FCmpInst::FCMP_ONE ?
                                           builder.CreateNot(non_finite) :
                                           non_finite);
      }
      else {
        auto is_nan = [&](Value *value) {
          return builder.CreateICmpUGT(magnitude_bits(value), infinity);
        };
        auto *unordered = builder.CreateOr(is_nan(comparison->getOperand(0)),
                                           is_nan(comparison->getOperand(1)));
        comparison->replaceAllUsesWith(comparison->getPredicate() == FCmpInst::FCMP_ORD ?
                                           builder.CreateNot(unordered) :
                                           unordered);
      }
    }
    else if (auto *ctlz = dyn_cast<IntrinsicInst>(instruction);
             ctlz && ctlz->getIntrinsicID() == Intrinsic::ctlz)
    {
      /* The Vulkan backend cannot lower ctlz. Spread the high bit then count the zeros. */
      Value *bits = ctlz->getArgOperand(0);
      for (unsigned int shift = 1; shift < 32; shift *= 2) {
        bits = builder.CreateOr(bits, builder.CreateLShr(bits, shift));
      }
      Value *count = builder.CreateUnaryIntrinsic(Intrinsic::ctpop, bits);
      ctlz->replaceAllUsesWith(builder.CreateSub(builder.getInt32(32), count));
    }
    else {
      auto *call = cast<IntrinsicInst>(instruction);
      auto *magnitude = builder.CreateBitCast(call->getArgOperand(0), builder.getInt32Ty());
      auto *sign = builder.CreateBitCast(call->getArgOperand(1), builder.getInt32Ty());
      auto *bits = builder.CreateOr(builder.CreateAnd(magnitude, builder.getInt32(0x7fffffff)),
                                    builder.CreateAnd(sign, builder.getInt32(0x80000000)));
      call->replaceAllUsesWith(builder.CreateBitCast(bits, call->getType()));
    }
    instruction->eraseFromParent();
  }
  return !replacements.empty();
}

struct LegalizeSpirv : PassInfoMixin<LegalizeSpirv> {
  PreservedAnalyses run(Function &function, FunctionAnalysisManager &)
  {
    const bool access = type_array_accesses(function);
    const bool offsets = rebase_array_byte_offsets(function);
    const bool values = lower_spirv_values(function);
    return access || offsets || values ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

/* Give the original LLVM basic blocks a single structured loop. The Vulkan
 * backend cannot emit valid nested loops for the pinned Cycles traversal's
 * early exits. This pass changes the CFG while leaving each block's work in
 * place. Run reg2mem first to spill values carried across blocks. */
struct DispatchCFG : PassInfoMixin<DispatchCFG> {
  bool force_main;

  explicit DispatchCFG(const bool force_main = false) : force_main(force_main) {}

  PreservedAnalyses run(Function &function, FunctionAnalysisManager &analyses)
  {
    if (analyses.getResult<LoopAnalysis>(function).empty() &&
        !(force_main && function.getName() == "main"))
    {
      return PreservedAnalyses::all();
    }
    if (!function.getReturnType()->isVoidTy()) {
      report_fatal_error("Shader CFG dispatcher needs a void entry point");
    }

    SmallVector<BasicBlock *> cases;
    DenseMap<BasicBlock *, unsigned> indices;
    BasicBlock *entry = &function.getEntryBlock();
    for (BasicBlock &block : function) {
      if (!block.phis().empty()) {
        report_fatal_error("Run reg2mem before flattening the shader CFG");
      }
      if (&block != entry) {
        indices[&block] = cases.size() + 1;
        cases.push_back(&block);
      }
    }

    LLVMContext &context = function.getContext();
    auto *header = BasicBlock::Create(context, "dispatch.header", &function);
    auto *select = BasicBlock::Create(context, "dispatch.select", &function);
    auto *latch = BasicBlock::Create(context, "dispatch.latch", &function);
    auto *unrecognized = BasicBlock::Create(context, "dispatch.default", &function);
    auto *exit = BasicBlock::Create(context, "dispatch.exit", &function);

    IRBuilder<> builder(&*entry->getFirstInsertionPt());
    auto *state = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "dispatch.state");

    builder.SetInsertPoint(header);
    auto *id = builder.CreateLoad(builder.getInt32Ty(), state);
    auto *has_work = builder.CreateICmpNE(id, builder.getInt32(0));
    builder.CreateCondBr(has_work, select, exit);

    builder.SetInsertPoint(select);
    auto *dispatch = builder.CreateSwitch(id, unrecognized, cases.size());
    for (BasicBlock *block : cases) {
      dispatch->addCase(builder.getInt32(indices.at(block)), block);
    }

    builder.SetInsertPoint(unrecognized);
    builder.CreateStore(builder.getInt32(0), state);
    builder.CreateBr(latch);

    builder.SetInsertPoint(latch);
    builder.CreateBr(header);
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();

    auto redirect = [&](BasicBlock *block, BasicBlock *next) {
      Instruction *current = block->getTerminator();
      if (isa<UnreachableInst>(current)) {
        return;
      }
      builder.SetInsertPoint(current);
      Value *id = builder.getInt32(0);
      if (auto *branch = dyn_cast<BranchInst>(current)) {
        id = builder.getInt32(indices.at(branch->getSuccessor(0)));
        if (branch->isConditional()) {
          Value *on_false = builder.getInt32(indices.at(branch->getSuccessor(1)));
          id = builder.CreateSelect(branch->getCondition(), id, on_false);
        }
      }
      else if (!isa<ReturnInst>(current)) {
        report_fatal_error("Run lower-switch before flattening the shader CFG");
      }
      builder.CreateStore(id, state);
      builder.CreateBr(next);
      current->eraseFromParent();
    };

    redirect(entry, header);
    for (BasicBlock *block : cases) {
      redirect(block, latch);
    }
    return PreservedAnalyses::none();
  }
};

}  // namespace

extern "C" PassPluginLibraryInfo llvmGetPassPluginInfo()
{
  return {LLVM_PLUGIN_API_VERSION,
          "cycles-webgpu-bvh2",
          LLVM_VERSION_STRING,
          [](PassBuilder &builder) {
            builder.registerPipelineParsingCallback([](StringRef name,
                                                       ModulePassManager &passes,
                                                       ArrayRef<PassBuilder::PipelineElement>) {
              if (name != "cycles-webgpu-lower-film") {
                return false;
              }
              passes.addPass(LowerFilmStorage());
              return true;
            });
            builder.registerPipelineParsingCallback([](StringRef name,
                                                       FunctionPassManager &passes,
                                                       ArrayRef<PassBuilder::PipelineElement>) {
              if (name == "cycles-webgpu-hoist-resources") {
                passes.addPass(HoistResourceHandles());
              }
              else if (name == "cycles-webgpu-dispatch-cfg") {
                passes.addPass(DispatchCFG());
              }
              else if (name == "cycles-webgpu-dispatch-cfg-force-main") {
                passes.addPass(DispatchCFG(true));
              }
              else if (name == "cycles-webgpu-legalize-values") {
                passes.addPass(LegalizeSpirv());
              }
              else {
                return false;
              }
              return true;
            });
          }};
}
