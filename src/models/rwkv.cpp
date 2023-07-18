//
// Created by bbuf on 17/7/23.
//

#include "utils.h"

#include "rwkv.h"

#include <cmath>

#include <chrono>

#include <algorithm>

namespace fastllm {

    extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

    RWKVModel::RWKVModel() {
        this->model_type = "rwkv-raven-7b";
        this->pre_prompt = "### Instruction: {question}\n### Response:";
        block_cnt = 24;

        this->weight.embeddingNames.insert("transformer.rwkv.embeddings.weight");
    }

    Data RWKVModel::ZeroLike(Data data) {
        std::vector<int> shape;
        int shape_size = data.DimSize();
        for (int i = 0; i < data.dims.size(); i++){
            shape.push_back(data.dims[i]);
        }
        std::vector<float> zero(shape_size , 0);
        Data result(data.dataType, shape, zero);
        return result;
    }

    std::vector <Data> RWKVModel::RWKVLinearAttentionCpu(Data time_decay, Data time_first, Data key, Data value, Data state) {
        const int seq_length = key.dims[1];
        Data output = ZeroLike(key);
        Data num_state = state;
        Data den_state = state;
        Data max_state = state;
        std::vector <int> key_shape, value_shape;
        for (int i = 0; i < key.dims.size(); i++){
            key_shape.push_back(key.dims[i]);
        }
        for(int i = 0; i < value.dims.size(); i++){
            value_shape.push_back(value.dims[i]);
        }
        int time_decay_shape_size = time_decay.DimSize();
        float *data_ptr = (float*)time_decay.cpuData;
        for (int i = 0; i < time_decay_shape_size; i++) {
            data_ptr[i] = -std::exp(data_ptr[i]);
        }
        for (int current_index = 0; current_index < seq_length; current_index ++){
            // current_key = key[:, current_index].float()
            std::vector <int> new_key_shape, new_value_shape;
            for(int i = 0; i < key_shape.size(); i++){
                if(i != current_index) new_key_shape.push_back(key_shape[i]);
            }
            for(int i = 0; i < value_shape.size(); i++){
                if(i != current_index) new_value_shape.push_back(value_shape[i]);
            }
        }
    }

    int RWKVModel::Forward(const Data &inputIds, const Data &attentionMask,
                            const Data &positionIds, std::vector <std::pair <Data, Data> > &pastKeyValues,
                           const GenerationConfig &generationConfig, const LastTokensManager &lastTokens) {
        auto st = std::chrono::system_clock::now();

        Data inputEmbeddings;
        Embedding(inputIds, this->weight["transformer.rwkv.embeddings.weight"], inputEmbeddings);
        Data hidden = inputEmbeddings;

        for (int i = 0; i < block_cnt; i++) {
            // if self.layer_id == 0: hidden = self.pre_ln(hidden)
            if(i == 0){
                std::string prelnWeightName = "transformer.rwkv.blocks." + std::to_string(i) + ".attention.pre_ln.weight";
                std::string prelnBiastName = "transformer.rwkv.blocks." + std::to_string(i) + ".attention.pre_ln.bias";
                LayerNorm(hidden, weight[prelnWeightName], weight[prelnBiastName], -1, hidden);
            }

        }
    }

    std::string RWKVModel::Response(const std::string &input,
                                    RuntimeResult retCb,
                                    const GenerationConfig &generationConfig) {
        // match _rescale_layers func
        const int rescale_every = 6;
        for (int i = 0; i < block_cnt; i++) {
            std::string attention_output_weight_name = "transformer.rwkv.blocks." + std::to_string(i) + ".attention.output.weight";
            std::string feed_forward_value_weight_name = "transformer.rwkv.blocks." + std::to_string(i) + ".feed_forward.value.weight";
            const int tmp = std::pow(2, int(i / rescale_every));
            
            uint8_t* attention_output_weight_cpu_data = (uint8_t *)weight[attention_output_weight_name].cpuData;
            int64_t attention_output_weight_size = 1;
            for(int j = 0; j < weight[attention_output_weight_name].dims.size(); j++){
                attention_output_weight_size *= weight[attention_output_weight_name].dims[j];
            }
            for (int j = 0; j < attention_output_weight_size; j++){
                attention_output_weight_cpu_data[j] *= tmp;
            }
            weight[attention_output_weight_name].cpuData = (uint8_t*)attention_output_weight_cpu_data;

            uint8_t* feed_forward_value_weight_cpu_data = (uint8_t *)weight[feed_forward_value_weight_name].cpuData;
            int64_t feed_foward_value_weight_size = 1;
            for (int j = 0; j < weight[feed_forward_value_weight_name].dims.size(); j++){
                feed_foward_value_weight_size *= weight[feed_forward_value_weight_name].dims[j];
            }
            for (int j = 0; j < feed_foward_value_weight_size; j++){
                feed_forward_value_weight_cpu_data[j] *= tmp;
            }
            weight[feed_forward_value_weight_name].cpuData = (uint8_t*)feed_forward_value_weight_cpu_data;
        }
        
        Data inputIds = this->weight.tokenizer.Encode(input);
        Data attentionMask = inputIds;
        Data positionIds = inputIds;
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(), Data()));
        }

        int len = inputIds.dims[1];
        for (int i = 0; i < len; i++) {
            ((float *) attentionMask.cpuData)[i] = 1;
            ((float *) positionIds.cpuData)[i] = i;
        }
        
        // match if use_cache and state is None ...
        state.resize(5);
        const int32_t hidden_size = 2048;
        const int32_t num_hidden_layers = 24;
        std::vector<int> shape = {inputIds.dims[0], hidden_size, num_hidden_layers};
        std::vector<float> zero(inputIds.dims[0] * hidden_size * num_hidden_layers , 0);
        std::vector<float> nef(inputIds.dims[0] * hidden_size * num_hidden_layers, -1e30);
        for (int i = 0; i < 5; i++){
            Data state_tmp(DataType::FLOAT32, shape, i == 4 ? nef : zero);
            state[i] = state_tmp;
        }

        std::vector<float> results;
        std::string retString = "";
		int index = 0;
        LastTokensManager tokens (1, generationConfig.last_n);
        while (true) {
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == 0) {
                break;
            }

            results.push_back(ret);
            std::string current = weight.tokenizer.Decode(
                    Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
            retString += current;
			if (retCb)
#ifdef PY_API
				retCb(index, pybind11::bytes(retString));
#else
				retCb(index, current.c_str());
#endif
            index++;
            fflush(stdout);
            results.clear();

            len++;
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) ret}));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, len}, std::vector<float>(len, 1.0f)));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) (len - 1)}));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }

		if (retCb)
#ifdef PY_API
			retCb(-1, pybind11::bytes(retString));
#else
			retCb(-1, retString.c_str());
#endif
        return retString;
    }

    std::string RWKVModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + input;
    }

    std::string RWKVModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + output + history_sep;
    }

    int RWKVModel::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                        const GenerationConfig &generationConfig) {
        ErrorInFastLLM("Unsupport.\n");
        return 0;
    }

    int RWKVModel::FetchResponseTokens(int handleId) {
        ErrorInFastLLM("Unsupport.\n");
        return -1;
    }

}