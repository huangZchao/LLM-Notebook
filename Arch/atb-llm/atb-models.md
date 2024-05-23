# 初始化入口
python：
```python
# flash causal_qwen2.py
torch.classes.ModelTorch.ModelTorch("qwen_PAModel")
```
C++对应：
```c++
// MindIE-LLM/examples/atb_models/pytorch/adapter/model/model_torch.cpp
ModelTorch::ModelTorch(std::string modelName) : modelName_(modelName)
{
    modelId_ = GetNewModelId();
    context_ = atb_speed::ContextFactory::GetAtbContext(Utils::GetCurrentStream());
    ATB_LOG(INFO) << "ModelTorch new modelName:" << modelName_ << ", modelId:" << modelId_;
}
```

# 创建模型实例
```c++
// MindIE-LLM/examples/atb_models/pytorch/adapter/model/model_torch.cpp
int64_t ModelTorch::SetParam(std::string param)
{
    ...
    model_ = atb_speed::ModelFactory::CreateInstance(modelName_, param);
    ...
}

...
// MindIE-LLM/examples/atb_models/core/utils/model_factory.cpp
std::shared_ptr<atb_speed::Model> ModelFactory::CreateInstance(const std::string &modelName, const std::string &param)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(INFO) << "find model: " << modelName;
        return it->second(param);
    }
    ATB_LOG(WARN) << "ModelName: " << modelName << " not find in model factory map";
    return nullptr;
}
...
// MindIE-LLM/examples/atb_models/models/qwen/model/decoder_model.cpp
DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}
```

# 模型初始化
```c++
// MindIE-LLM/examples/atb_models/pytorch/adapter/model/model_torch.cpp
int64_t ModelTorch::SetParam(std::string param)
{
    ...
        atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, nullptr);
    ...

// 
}
...

// MindIE-LLM/examples/atb_models/core/base/model.cpp
int64_t Model::Init(GetWorkspaceFunc getWorkSpaceFunc, CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
    RunTaskFunc runTaskFunc)
{
    const char *envStr = std::getenv("ATB_OPERATION_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1");
    if (isUsePlanExecuteAsync_ && !runTaskFunc) {
        std::thread thread = std::thread(std::bind(&Model::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ATB_LOG(FATAL) << modelName_ << " new, isTaskQueueEnable:" << (runTaskFunc != nullptr) 
                   << ", isUsePlanExecuteAsync:" << isUsePlanExecuteAsync_ << ", currentDevId:" << currentDevId_;
    
    getWorkSpaceFunc_ = getWorkSpaceFunc;
    createTensorFromTensorDescFunc_ = createTensorFromTensorDescFunc;
    runTaskFunc_ = runTaskFunc;

    int64_t atbStatus = BuildGraph();
    graph_.Init();
    ATB_LOG(DEBUG) << modelName_ << " init graph:\n" << graph_.ToString();
    return atbStatus;
}
...
```

# <font color='red'>核心，建图与推理</font>
```c++
// 建图
// MindIE-LLM/examples/atb_models/models/qwen/model/decoder_model.cpp
int64_t DecoderModel::BuildGraph()
{
    // set size
    const int weightTensorSize = (param_.withEmbedding ? WEIGHT_COUNT_WORD_EMBEDDINGNODE : 0) +
                                 WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers + WEIGHT_COUNT_POST_NORM +
                                 WEIGHT_COUNT_LM_HEAD;

    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.numHiddenLayers);
    graph_.vCacheTensors.resize(param_.numHiddenLayers);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    graph_.internalTensors.resize(INTERNAL_TENSOR_MAX);

    const int nodeSize = param_.numHiddenLayers +
                         (param_.withEmbedding ? OPERATION_COUNT_BEFORE_LAYER : OPERATION_COUNT_BEFORE_LAYER - 1) +
                         OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    ATB_LOG(INFO) << "weightTensors.size=" << graph_.weightTensors.size()
                  << ", inTensors.size=" << graph_.inTensors.size()
                  << ", outTensors.size=" << graph_.outTensors.size()
                  << ", internalTensor.size=" << graph_.internalTensors.size()
                  << ", nodes.size=" << graph_.nodes.size();

    ATB_LOG(INFO) << "DecoderModel build graph begin";
    int nodeId = 0;

    atb::Operation *op = nullptr;

    // wte
    if (param_.withEmbedding) {
        auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
        atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
        wordEmbeddingParam.unpadInputs = !param_.isFA;
        if (param_.isEmbeddingParallel) {
            wordEmbeddingParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend};
        };
        atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0),  // shape: [vocabSize + 1, hiddenSize]
            &graph_.inTensors.at(IN_TENSOR_INPUT)};
        wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(INTERNAL_HIDDENSTATES)};
        ATB_LOG(INFO) << "[+] wordEmbeddingNode";
    }

    // gather
    auto &peGatherNode = graph_.nodes.at(nodeId++);
    atb_speed::common::PositionalEmbeddingGather(&op);
    peGatherNode.operation.reset(op);
    peGatherNode.inTensors = {
        &graph_.inTensors.at(IN_TENSOR_POSITIONIDS),
        &graph_.inTensors.at(IN_TENSOR_COSTABLE),
        &graph_.inTensors.at(IN_TENSOR_SINTABLE),
    };
    peGatherNode.outTensors = {
        &graph_.internalTensors.at(INTERNAL_COSEMBED),
        &graph_.internalTensors.at(INTERNAL_SINEMBED)
    };
    ATB_LOG(INFO) << "[+] peGatherNode";

    atb::Tensor *firstInTensor = param_.withEmbedding ? &graph_.internalTensors.at(INTERNAL_HIDDENSTATES)
                                                      : &graph_.inTensors.at(IN_TENSOR_INPUT);

    // layers
    for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::qwen::DecoderLayerParam layerParam;
        layerParam.isFA = param_.isFA;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.isBF16 = param_.isBF16;
        layerParam.supportSwiGLU = param_.supportSwiGLU;
        layerParam.packQuantType = param_.packQuantType[layerId];
        layerParam.linearQuantType = param_.linearQuantType[layerId];
        layerParam.linearTransposeType = param_.linearTransposeType[layerId];
        layerParam.supportLcoc = param_.supportLcoc;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
        layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
        layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
        layerParam.rank = param_.rank;
        layerParam.worldSize = param_.worldSize;
        layerParam.backend = param_.backend;
        layerParam.enableLogN = param_.enableLogN;
        atb_speed::qwen::DecoderLayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId +
                                         (param_.withEmbedding ? WEIGHT_COUNT_WORD_EMBEDDINGNODE : 0));
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_COSEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(INTERNAL_SINEMBED);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQ_LENGTHS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_PLACEHOLDER);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKEN_OFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_KV_CACHE_IDX);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);

        layerNode.outTensors = {&graph_.internalTensors.at(INTERNAL_HIDDENSTATES)};
        ATB_LOG(INFO) << "[+] layerNode_" << layerId;
        firstInTensor = layerNode.outTensors.at(0);
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(INTERNAL_HIDDENSTATES)
    };
    ATB_LOG(INFO) << "[+] finalNormNode";

    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param_.lmHeadTransposeType;
    if (param_.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = param_.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = param_.worldSize;
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = param_.backend;
    }
    LmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode.inTensors = {
        &graph_.internalTensors.at(INTERNAL_HIDDENSTATES),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_PLACEHOLDER),
        &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)
    };
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    ATB_LOG(INFO) << "[+] lmHeadNode";

    ATB_LOG(INFO) << "DecoderModel build graph success";
    return atb::NO_ERROR;
}
```

# <font color='red'>图执行</font>
```c++
```

# 模型注册
```c++
// 以qwen为例
// MindIE-LLM/examples/atb_models/models/qwen/model/decoder_model.h
REGISTER_MODEL(qwen, DecoderModel);
...
// MindIE-LLM/examples/atb_models/core/include/atb_speed/utils/model_factory.h
#define REGISTER_MODEL(nameSpace, modelName)                                                      \
        struct Register##_##nameSpace##_##modelName {                                             \
            inline Register##_##nameSpace##_##modelName()                                         \
            {                                                                                     \
                ATB_LOG(INFO) << "register model " << #nameSpace << "_" << #modelName;                  \
                ModelFactory::Register(MODEL_NAMESPACE_STRINGIFY(nameSpace##_##modelName),        \
                    [](const std::string &param) { return std::make_shared<modelName>(param); }); \
            }                                                                                     \
        } static instance_##nameSpace##modelName;
...
// 真正的Register
// MindIE-LLM/examples/atb_models/core/include/atb_speed/utils/model_factory.cpp
bool ModelFactory::Register(const std::string &modelName, CreateModelFuncPtr createModel)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(WARN) << modelName << " model already exists, but the duplication doesn't matter.";
        return false;
    }
    ModelFactory::GetRegistryMap()[modelName] = createModel;
    return true;
}
```


# python和C++的接口对应关系
```c++
TORCH_LIBRARY(ModelTorch, m)
{
    m.class_<ModelTorch>("ModelTorch")
        .def(torch::init<std::string>())
        .def("set_param", &ModelTorch::SetParam)
        .def("set_weight", &ModelTorch::SetWeight)
        .def("set_kv_cache", &ModelTorch::SetKVCache)
        .def("execute", &ModelTorch::Execute)
        .def("execute_out", &ModelTorch::ExecuteOut);
}
```

