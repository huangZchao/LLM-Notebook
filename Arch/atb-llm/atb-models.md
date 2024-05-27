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
```
## Embedding
```c++
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
```

## Position Embedding
```c++
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
```

## Layer
<font color='red'>graph->node->operation</font>
```c++
// 重点
// MindIE-LLM\examples\atb_models\models\qwen\model\decoder_model.cpp
atb_speed::qwen::DecoderLayer(layerParam, &op);
...
    // MindIE-LLM\examples\atb_models\models\qwen\layer\decoder_layer.cpp
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpParallelNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    ...
    // attention 部分
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    ...
    Attention(fusionAttentionParam, &attentionNode.operation);
    ...
    attentionNode.inTensorIds = {...}
    attentionNode.outTensorIds = {...};
    ...
    // residual部分
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    ...
    // mlp
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    ...
    MlpSwiGLU(mlpParam, &mlpParallelNode.operation);
    ...
    // mlp residual
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
...
```

## FinalNorm
```c++
auto &finalNormNode = graph_.nodes.at(nodeId++);
atb::infer::RmsNormParam finalNormParam;
finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
finalNormParam.normParam.epsilon = param_.rmsNormEps;
CREATE_OPERATION(finalNormParam, &op);
```

## LM Head
```c++
...
atb_speed::common::LmHeadParam lmHeadParam;
...
LmHead(lmHeadParam, &op);
```



# <font color='red'>图执行</font>
- python
```python
self.acl_encoder_operation.execute(acl_inputs, acl_param)
或
self.acl_decoder_operation.execute(acl_inputs, acl_param)
```

- c++
```c++
// MindIE-LLM\examples\atb_models\pytorch\adapter\model\model_torch.cpp
std::vector<torch::Tensor> ModelTorch::Execute(std::vector<torch::Tensor> atInTensors, std::string param) 
{
    ...
    // input desc
    std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
    ...
    // output desc
    std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
    ...
    atb::Status st = model_->InferShape(inTensorDescs, outTensorDescs);
    ...
    // 执行推理
    int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
    ...
}

int64_t ModelTorch::ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                                const std::string &param)
{
    int64_t atbStatus = model_->Execute(context_.get(), inTensors, outTensors, param);
    executeCount_++;
    return atbStatus;
}   

// MindIE-LLM\examples\atb_models\core\base\model.cpp
atb::Status Model::Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
                           std::vector<atb::Tensor> &outTensors, const std::string &param)
{
    ...
    atb::Status st = ExecuteNode(nodeId);
    ...
}

atb::Status Model::ExecuteNode(int nodeId)
{
    ...
    st = ExecutePlanSync(nodeId);   
    ...
}

atb::Status Model::ExecutePlanSync(int nodeId)
{
    ...
    atb::Status st = node.operation->Execute(variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    ...
}
```

# <font color='red'>Extra：atb::operation的创建方式</font>
- 关键：opParam和opOperation；
- 主要是opParam，即对应参数的初始化；
- 其次是 op 的 inTensorIds 和 outTensorIds；
```c++
    int nodeId = 0;
    auto &inputIdEmbeddingNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam inputembedinggatherparam;
    inputembedinggatherparam.axis = param.axis;
    CREATE_OPERATION(inputembedinggatherparam, &inputIdEmbeddingNode.operation);
    inputIdEmbeddingNode.inTensorIds = {
        WordEmbeddingTensorIdx::IN_EMBEDDING_WEIGHTS, WordEmbeddingTensorIdx::IN_INPUT_IDS
    };
    inputIdEmbeddingNode.outTensorIds = {
        param.tensorParallelInfo.worldSize > 1 ? \
        WordEmbeddingTensorIdx::INTERMEDIATE_GATHER : WordEmbeddingTensorIdx::OUT_HIDDEN_STATES
    };
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

