//
// Created by huangyuyang on 5/12/23.
//

#ifndef TEST_RWKV_H
#define TEST_RWKV_H

#include "basellm.h"
#include "cmath"

namespace fastllm {
    class RWKVModel: public basellm {
	public:
        RWKVModel(); // 构造函数
        std::vector<Data> state;

        // 推理
		virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager());

		virtual std::string Response(const std::string &input, RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        virtual int LaunchResponseTokens(const std::vector <int> &inputTokens,
                                         const GenerationConfig &generationConfig = GenerationConfig()); // 启动一个response任务，返回分配的handleId

        virtual int FetchResponseTokens(int handelId); // 获取指定handle的输出, -1代表输出结束了
    private:
        Data ZeroLike(Data data);
        std::vector <Data> RWKVLinearAttentionCpu(Data time_decay, Data time_first, Data key, Data value, Data state);

    };
}

#endif //TEST_RWKV_H
