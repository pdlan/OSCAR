#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <set>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <inttypes.h>
#include "json.hpp"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"

using namespace std;
using namespace llvm;
using json = nlohmann::json;

namespace {
static cl::opt<string> InstFilename("inst-out", cl::desc("inst out file"), cl::value_desc("filename"));
static cl::opt<string> StateFilename("state-out", cl::desc("state out file"), cl::value_desc("filename"));
static cl::opt<string> PosFilename("pos-out", cl::desc("pos out file"), cl::value_desc("filename"));
static cl::opt<int> MaxLength("max-len", cl::desc("max length"), cl::value_desc("length"));
static cl::opt<int> MinLength("min-len", cl::desc("min length"), cl::value_desc("length"));
static cl::opt<bool> ConcatFunc("concat-func", cl::desc("concatenate function"));
static cl::opt<bool> Truncate("truncate", cl::desc("truncate"));
static cl::opt<bool> FuncName("func-name", cl::desc("add function name tag"));
static cl::opt<bool> NewlineWhenSkipping("newline-when-skipping", cl::desc("emit newline when skipping a function"));
struct StaticAnalysis : public FunctionPass {
    static char ID;
	int totalPos;
    ofstream insts_ofs;
    ofstream states_ofs;
    ofstream pos_ofs;
    StaticAnalysis() : FunctionPass(ID), totalPos(0), insts_ofs(InstFilename), states_ofs(StateFilename),
        pos_ofs(PosFilename) {}
    
    bool runOnFunction(Function &F) override;
    void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.setPreservesCFG();
        AU.addRequired<DominatorTreeWrapperPass>();
        AU.addRequired<LoopInfoWrapperPass>();
        AU.addRequired<AAResultsWrapperPass>();
        AU.addRequired<MemorySSAWrapperPass>();
    }
};
}

struct Context {
    vector<Instruction *> values;
    map<Instruction *, size_t> values_id;
    vector<AllocaInst *> local_variables;
    map<AllocaInst *, size_t> local_variables_id;
    vector<Argument *> arguments;
    map<Argument *, size_t> arguments_id;
    vector<Instruction *> instructions;
    map<Instruction *, size_t> instruction_pos;
};

struct State {
    vector<string> tokens;
    virtual const vector<string> &getTokens() {
        return tokens;
    }
    string operandToString(Value *value, Context &context) {
        if (auto it = context.values_id.find((Instruction *)value); it != context.values_id.end()) {
            return "v" + to_string(it->second);
        } else if (auto it = context.arguments_id.find((Argument *)value); it != context.arguments_id.end()) {
            return "a" + to_string(it->second);
        } else if (Constant *constant = dyn_cast<Constant>(value)) {
            if (ConstantInt *cint = dyn_cast<ConstantInt>(constant); cint && cint->getBitWidth() <= 64) {
                return to_string(cint->getSExtValue());
            } else if (ConstantFP *cfp = dyn_cast<ConstantFP>(constant)) {
                ostringstream ss;
                ss << setprecision(8) << noshowpoint << cfp->getValueAPF().convertToDouble();
                return ss.str();
            } else if (ConstantPointerNull *cnull = dyn_cast<ConstantPointerNull>(constant)) {
                return "<null>";
            } else {
                return "<const>";
            }
        }
        return "";
    }
};

struct BinaryOpState : public State {
    BinaryOpState(Function &F, Instruction &inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[&inst]));
        tokens.push_back("=");
        tokens.push_back(operandToString(inst.getOperand(0), context));
        string opcode;
        switch (inst.getOpcode()) {
            case Instruction::Add:  opcode = "+"; break;
            case Instruction::FAdd: opcode = "+"; break;
            case Instruction::Sub:  opcode = "-"; break;
            case Instruction::FSub: opcode = "-"; break;
            case Instruction::Mul:  opcode = "*"; break;
            case Instruction::FMul: opcode = "*"; break;
            case Instruction::UDiv: opcode = "/"; break;
            case Instruction::SDiv: opcode = "/"; break;
            case Instruction::FDiv: opcode = "/"; break;
            case Instruction::URem: opcode = "%"; break;
            case Instruction::SRem: opcode = "%"; break;
            case Instruction::FRem: opcode = "%"; break;
            case Instruction::Shl:  opcode = "<<"; break;
            case Instruction::LShr: opcode = ">>"; break;
            case Instruction::AShr: opcode = ">>"; break;
            case Instruction::And:  opcode = "&"; break;
            case Instruction::Or:   opcode = "|"; break;
            case Instruction::Xor:  opcode = "^"; break;
        }
        tokens.push_back(opcode);
        tokens.push_back(operandToString(inst.getOperand(1), context));
    }
};

struct UnaryOpState : public State {
    UnaryOpState(Function &F, Instruction &inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[&inst]));
        tokens.push_back("=");
        tokens.push_back("-");
        tokens.push_back(operandToString(inst.getOperand(0), context));
    }
};

struct LoadState : public State {
    LoadState(Function &F, Instruction &inst, Context &context, MemorySSA &MSSA) {
        Value *ptr = inst.getOperand(0);
        tokens.push_back("v" + to_string(context.values_id[&inst]));
        tokens.push_back("<-");
        if (auto it = context.local_variables_id.find((AllocaInst *)ptr); it != context.local_variables_id.end()) {
            tokens.push_back("m" + to_string(it->second));
        } else {
            tokens.push_back("dereference");
            tokens.push_back(operandToString(ptr, context));
        }
        MemoryUseOrDef *use = MSSA.getMemoryAccess(&inst);
        MemoryAccess *access = use->getDefiningAccess();
        vector<string> possible_values;
        set<MemoryAccess *> found_access;
        find_possible_values(possible_values, found_access, context, access);
        if (possible_values.size()) {
            tokens.push_back("=");
            for (size_t i = 0; i < possible_values.size(); ++i) {
                tokens.push_back(possible_values[i]);
                if (i != possible_values.size() - 1) {
                    tokens.push_back(",");
                }
            }
        }
    }
    void find_possible_values(vector<string> &res, set<MemoryAccess *> &found_access, Context &context, MemoryAccess *access) {
        MemoryDef *def = dyn_cast<MemoryDef>(access);
        MemoryPhi *phi = dyn_cast<MemoryPhi>(access);
        found_access.insert(access);
        if (def) {
            Instruction *def_inst = def->getMemoryInst();
            if (def_inst && def_inst->getOpcode() == Instruction::Store) {
                Value *val = def_inst->getOperand(0);
                res.push_back(operandToString(val, context));
            }
        } else if (phi) {
            unsigned n = phi->getNumIncomingValues();
            for (unsigned i = 0; i < n; ++i) {
                MemoryAccess *a = phi->getIncomingValue(i);
                if (found_access.count(a)) {
                    continue;
                }
                find_possible_values(res, found_access, context, a);
            }
        }
    }
};

struct StoreState : public State {
    StoreState(Function &F, Instruction &inst, Context &context) {
        Value *val = inst.getOperand(0);
        Value *ptr = inst.getOperand(1);
        tokens.push_back(operandToString(val, context));
        tokens.push_back("->");
        if (auto it = context.local_variables_id.find((AllocaInst *)ptr); it != context.local_variables_id.end()) {
            tokens.push_back("m" + to_string(it->second));
        } else {
            tokens.push_back("dereference");
            tokens.push_back(operandToString(ptr, context));
        }
    }
};

struct GetElementPtrState : public State {
    GetElementPtrState(Function &F, GetElementPtrInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        tokens.push_back("gep");
        Value *ptr = inst->getPointerOperand();
        if (auto it = context.local_variables_id.find((AllocaInst *)ptr); it != context.local_variables_id.end()) {
            tokens.push_back("m" + to_string(it->second));
        } else {
            tokens.push_back(operandToString(ptr, context));
        }
        for (Use &use : inst->indices()) {
            Value *val = use.get();
            tokens.push_back(operandToString(val, context));
        }
    }
};

struct CmpState : public State {
    CmpState(Function &F, CmpInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        string predicate;
        switch (inst->getPredicate()) {
            case CmpInst::FCMP_FALSE:
                tokens.push_back("false");
                return;
            case CmpInst::FCMP_OEQ: predicate = "=="; break;
            case CmpInst::FCMP_OGT: predicate = ">"; break;
            case CmpInst::FCMP_OGE: predicate = ">="; break;
            case CmpInst::FCMP_OLT: predicate = "<"; break;
            case CmpInst::FCMP_OLE: predicate = "<="; break;
            case CmpInst::FCMP_ONE: predicate = "!="; break;
            case CmpInst::FCMP_ORD: predicate = "order"; break;
            case CmpInst::FCMP_UNO: predicate = "unorder"; break;
            case CmpInst::FCMP_UEQ: predicate = "=="; break;
            case CmpInst::FCMP_UGT: predicate = ">"; break;
            case CmpInst::FCMP_UGE: predicate = ">="; break;
            case CmpInst::FCMP_ULT: predicate = "<"; break;
            case CmpInst::FCMP_ULE: predicate = "<="; break;
            case CmpInst::FCMP_UNE: predicate = "!="; break;
            case CmpInst::FCMP_TRUE:
                tokens.push_back("true");
                return;
            case CmpInst::ICMP_EQ: predicate = "=="; break;
            case CmpInst::ICMP_NE: predicate = "!="; break;
            case CmpInst::ICMP_UGT: predicate = ">"; break;
            case CmpInst::ICMP_UGE: predicate = ">="; break;
            case CmpInst::ICMP_ULT: predicate = "<"; break;
            case CmpInst::ICMP_ULE: predicate = "<="; break;
            case CmpInst::ICMP_SGT: predicate = ">"; break;
            case CmpInst::ICMP_SGE: predicate = ">="; break;
            case CmpInst::ICMP_SLT: predicate = "<"; break;
            case CmpInst::ICMP_SLE: predicate = "<="; break;
        }
        tokens.push_back(operandToString(inst->getOperand(0), context));
        tokens.push_back(predicate);
        tokens.push_back(operandToString(inst->getOperand(1), context));
    }
};

struct PHIState : public State {
    PHIState(Function &F, PHINode *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        int n = inst->getNumIncomingValues();
        for (int i = 0; i < n; ++i) {
            tokens.push_back(operandToString(inst->getIncomingValue(i), context));
            if (i != n - 1) {
                tokens.push_back(",");
            }
        }
    }
};

struct SelectState : public State {
    SelectState(Function &F, Instruction &inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[&inst]));
        tokens.push_back("=");
        tokens.push_back("select");
        tokens.push_back(operandToString(inst.getOperand(0), context));
        tokens.push_back(operandToString(inst.getOperand(1), context));
        tokens.push_back(operandToString(inst.getOperand(2), context));
    }
};

struct RetState : public State {
    RetState(Function &F, Instruction &inst, Context &context) {
        if (inst.getNumOperands()) {
            tokens.push_back("ret");
            tokens.push_back("=");
            tokens.push_back(operandToString(inst.getOperand(0), context));
        }
    }
};

struct AllocaState : public State {
    AllocaState(Function &F, AllocaInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        tokens.push_back("reference");
        tokens.push_back("m" + to_string(context.local_variables_id[inst]));
    }
};

struct CastState : public State {
    CastState(Function &F, CastInst *inst, Context &context) {
        string opcode;
        switch (inst->getOpcode()) {
            case Instruction::ZExt:
            case Instruction::SExt:
            case Instruction::FPTrunc:
            case Instruction::FPExt:
            case Instruction::PtrToInt:
            case Instruction::IntToPtr:
            case Instruction::BitCast:
            case Instruction::AddrSpaceCast:
                break;
            case Instruction::Trunc:
                opcode = "trunc";
                break;
            case Instruction::FPToUI:
            case Instruction::FPToSI:
                opcode = "fptoint";
                break;
            case Instruction::UIToFP:
            case Instruction::SIToFP:
                opcode = "inttofp";
                break;
        }
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        if (!opcode.empty()) {
            tokens.push_back(opcode);
        }
        tokens.push_back(operandToString(inst->getOperand(0), context));
    }
};

struct ExtractElementState : public State {
    ExtractElementState(Function &F, ExtractElementInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        tokens.push_back(operandToString(inst->getVectorOperand(), context));
		tokens.push_back(".");
		tokens.push_back(operandToString(inst->getIndexOperand(), context));
		
    }
};

struct InsertElementState : public State {
    InsertElementState(Function &F, InsertElementInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
		tokens.push_back("insert");
        tokens.push_back(operandToString(inst->getOperand(0), context));
		tokens.push_back(".");
		tokens.push_back(operandToString(inst->getOperand(2), context));
		tokens.push_back(operandToString(inst->getOperand(1), context));
		
    }
};

struct ShuffleVectorState : public State {
    ShuffleVectorState(Function &F, ShuffleVectorInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        tokens.push_back(operandToString(inst->getOperand(0), context));
		tokens.push_back("[");
		for (int mask : inst->getShuffleMask()) {
			tokens.push_back(to_string(mask));
		}
		tokens.push_back("]");
    }
};

struct ExtractValueState : public State {
    ExtractValueState(Function &F, ExtractValueInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
        tokens.push_back(operandToString(inst->getAggregateOperand(), context));
		for (unsigned int index : inst->indices()) {
			tokens.push_back(".");
			tokens.push_back(to_string(index));
		}
    }
};

struct InsertValueState : public State {
    InsertValueState(Function &F, InsertValueInst *inst, Context &context) {
        tokens.push_back("v" + to_string(context.values_id[inst]));
        tokens.push_back("=");
		tokens.push_back("insert");
        tokens.push_back(operandToString(inst->getAggregateOperand(), context));
		for (unsigned int index : inst->indices()) {
			tokens.push_back(".");
			tokens.push_back(to_string(index));
		}
		tokens.push_back(operandToString(inst->getInsertedValueOperand(), context));
    }
};

struct FreezeState : public State {
    FreezeState(Function &F, Instruction &inst, Context &context) {
        auto it = context.values_id.find(&inst);
        if (it != context.values_id.end()) {
            tokens.push_back("v" + to_string(it->second));
            tokens.push_back("=");
            tokens.push_back(operandToString(inst.getOperand(0), context));
        }
    }
};

struct OtherState : public State {
    OtherState(Function &F, Instruction &inst, Context &context) {
        auto it = context.values_id.find(&inst);
        if (it != context.values_id.end()) {
            tokens.push_back("v" + to_string(it->second));
            tokens.push_back("=");
            tokens.push_back("?");
        }
    }
};

// Number the values
void findValues(Function &F, Context &context) {
    int instructions = 0;
    // Function arguments
    for (Argument &arg: F.args()) {
        size_t id = context.arguments.size();
        context.arguments_id[&arg] = id;
        context.arguments.push_back(&arg);
        arg.setName("a" + to_string(id));
    }
    // SSA values
    for (BasicBlock &bb: F) {
        for (Instruction &inst: bb) {
            context.instruction_pos[&inst] = instructions;
            context.instructions.push_back(&inst);
            ++instructions;
            unsigned opcode = inst.getOpcode();
            switch (opcode) {
                case Instruction::Alloca: {
                    size_t id = context.local_variables.size();
                    context.local_variables_id[(AllocaInst *)&inst] = id;
                    context.local_variables.push_back((AllocaInst *)&inst);
                }
                case Instruction::Load:
                case Instruction::GetElementPtr: 
                case Instruction::AtomicCmpXchg:
                case Instruction::AtomicRMW:
                case Instruction::FNeg:
                case Instruction::Add:
                case Instruction::FAdd:
                case Instruction::Sub:
                case Instruction::FSub:
                case Instruction::Mul:
                case Instruction::FMul:
                case Instruction::UDiv:
                case Instruction::SDiv:
                case Instruction::FDiv:
                case Instruction::URem:
                case Instruction::SRem:
                case Instruction::FRem:
                case Instruction::Shl:
                case Instruction::LShr:
                case Instruction::AShr:
                case Instruction::And:
                case Instruction::Or:
                case Instruction::Xor:
                case Instruction::ICmp:
                case Instruction::FCmp:
                case Instruction::PHI:
                case Instruction::Call:
                case Instruction::Select:
                case Instruction::VAArg:
                case Instruction::Trunc:
                case Instruction::ZExt:
                case Instruction::SExt:
                case Instruction::FPToUI:
                case Instruction::FPToSI:
                case Instruction::UIToFP:
                case Instruction::SIToFP:
                case Instruction::FPTrunc:
                case Instruction::FPExt:
                case Instruction::PtrToInt:
                case Instruction::IntToPtr:
                case Instruction::BitCast:
                case Instruction::AddrSpaceCast:
				case Instruction::CleanupPad:
				case Instruction::CatchPad:
				case Instruction::ExtractElement:
				case Instruction::InsertElement:
				case Instruction::ShuffleVector:
				case Instruction::ExtractValue:
				case Instruction::InsertValue:
				case Instruction::LandingPad:
				case Instruction::Freeze:
				{
                    if (CallBase *cb = dyn_cast<CallBase>(&inst);
                        cb && cb->getFunctionType()->getReturnType()->isVoidTy()) {
                        break;
                    }
                    size_t id = context.values.size();
                    context.values_id[&inst] = id;
                    context.values.push_back(&inst);
                    inst.setName("v" + to_string(id));
                }
            }
        }
    }
}

bool StaticAnalysis::runOnFunction(Function &F) {
    int instCount = 0;
    string funcName = F.getName().str();
    for (BasicBlock &bb: F) {
        instCount += distance(bb.begin(), bb.end());
    }
    if (!Truncate && (instCount > MaxLength || instCount < MinLength)) {
		if (NewlineWhenSkipping) {
            if (FuncName) {
                insts_ofs << funcName << endl;
                states_ofs << funcName << endl;
                pos_ofs << funcName << endl;
            }
			insts_ofs << endl;
			states_ofs << endl;
			pos_ofs << endl;
		}
        return false;
    }
	if (!Truncate && ConcatFunc && totalPos + instCount > MaxLength) {
		return false;
	}
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    MemorySSA &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();
    Context context;
    findValues(F, context);
    vector<State *> states;
    map<BasicBlock *, int> bb_pos;
    int pos = ConcatFunc ? totalPos : 0;
    map<BasicBlock *, vector<vector<string>>> bb_states;
    map<BasicBlock *, vector<vector<string>>> bb_unknown_states;
	bool hasExceedMaxLength = false;
    for (BasicBlock &bb: F) {
		if (pos >= MaxLength) {
			hasExceedMaxLength = true;
		}
		if (hasExceedMaxLength) {
			bb_pos[&bb] = -1;
			continue;
		}
        bb_pos[&bb] = pos;
        for (Instruction &inst: bb) {
            unsigned opcode = inst.getOpcode();
            State *new_state;
            if (inst.isBinaryOp()) {
                new_state = new BinaryOpState(F, inst, context);
            } else if (inst.isUnaryOp()) {
                new_state = new UnaryOpState(F, inst, context);
            } else if (opcode == Instruction::Load) {
                new_state = new LoadState(F, inst, context, MSSA);
            } else if (opcode == Instruction::Store) {
                new_state = new StoreState(F, inst, context);
            } else if (opcode == Instruction::GetElementPtr) {
                new_state = new GetElementPtrState(F, (GetElementPtrInst *)&inst, context);
            } else if (opcode == Instruction::ICmp || opcode == Instruction::FCmp) {
                new_state = new CmpState(F, (CmpInst *)&inst, context);
            } else if (opcode == Instruction::PHI) {
                new_state = new PHIState(F, (PHINode *)&inst, context);
            } else if (opcode == Instruction::Select) {
                new_state = new SelectState(F, inst, context);
            } else if (opcode == Instruction::Ret) {
                new_state = new RetState(F, inst, context);
            } else if (opcode == Instruction::Alloca) {
                new_state = new AllocaState(F, (AllocaInst *)&inst, context);
            } else if (inst.isCast()) {
                new_state = new CastState(F, (CastInst *)&inst, context);
            } else if (opcode == Instruction::ExtractElement) {
				new_state = new ExtractElementState(F, (ExtractElementInst *)&inst, context);
			} else if (opcode == Instruction::InsertElement) {
				new_state = new InsertElementState(F, (InsertElementInst *)&inst, context);
			} else if (opcode == Instruction::ShuffleVector) {
				new_state = new ShuffleVectorState(F, (ShuffleVectorInst *)&inst, context);
			} else if (opcode == Instruction::ExtractValue) {
				new_state = new ExtractValueState(F, (ExtractValueInst *)&inst, context);
			} else if (opcode == Instruction::InsertValue) {
				new_state = new InsertValueState(F, (InsertValueInst *)&inst, context);
			} else if (opcode == Instruction::Freeze) {
				new_state = new FreezeState(F, inst, context);
			} else {
                new_state = new OtherState(F, inst, context);
            }
            states.push_back(new_state);
            ++pos;
        }
    }
    pos = 0;
    json res_pos, res_states, res_insts;
	int posOffset = ConcatFunc ? totalPos : 0;
    for (BasicBlock &bb: F) {
        unsigned depth = LI.getLoopDepth(&bb);
        for (Instruction &inst: bb) {
			if (Truncate && pos + posOffset >= MaxLength - 1) {
				break;
			}
            string inst_str;
            raw_string_ostream ss(inst_str);
            ss << inst;
            res_insts.push_back(inst_str);
            int true_pos = pos + 1 + posOffset;
            int false_pos = pos + 1 + posOffset;
            if (inst.isTerminator()) {
                if (BranchInst *br = dyn_cast<BranchInst>(&inst)) {
                    if (br->isConditional()) {
                        BasicBlock *b1 = dyn_cast<BasicBlock>(br->getOperand(1));
                        BasicBlock *b2 = dyn_cast<BasicBlock>(br->getOperand(2));
                        true_pos = bb_pos[b1];
                        false_pos = bb_pos[b2];
                    } else {
                        true_pos = bb_pos[dyn_cast<BasicBlock>(br->getOperand(0))];
						false_pos = bb_pos[dyn_cast<BasicBlock>(br->getOperand(0))];
                    }
                } else {
                    true_pos = -1;
                    false_pos = -1;
                }
            }
            const vector<string> &state_tokens = states[pos]->getTokens();
            vector<string> state{"loop", "=", to_string(depth)};
            if (state_tokens.size()) {
                state.push_back(";");
                state.insert(state.end(), state_tokens.begin(), state_tokens.end());
            }
            res_pos.push_back({pos + posOffset, true_pos, false_pos});
            res_states.push_back(state);
            ++pos;
        }
    }
	totalPos += pos;
    if (FuncName) {
        insts_ofs << funcName << endl;
        states_ofs << funcName << endl;
        pos_ofs << funcName << endl;
    }
    insts_ofs << res_insts.dump() << endl;
    states_ofs << res_states.dump() << endl;
    pos_ofs << res_pos.dump() << endl;
    return false;
}

char StaticAnalysis::ID = 0;
static RegisterPass<StaticAnalysis> X("staticanalysis", "CROS Static Analysis Pass", false, true);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [](const PassManagerBuilder &Builder,
       legacy::PassManagerBase &PM) { PM.add(new StaticAnalysis()); });
