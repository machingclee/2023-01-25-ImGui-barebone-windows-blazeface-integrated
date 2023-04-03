
#pragma once
#include "blazebase.h"
namespace blazeface
{
    using namespace torch::indexing;
    using namespace torch;

    class BlazeFaceImpl : public blazebase::BlazeDetector
    {
        bool back_model = false;
        nn::Sequential backbone = nullptr;
        nn::Sequential backbone1 = nullptr;
        nn::Sequential backbone2 = nullptr;

        blazebase::FinalBlazeBlock final = nullptr;
        nn::Conv2d classifier_8 = nullptr;
        nn::Conv2d classifier_16 = nullptr;
        nn::Conv2d regressor_8 = nullptr;
        nn::Conv2d regressor_16 = nullptr;

    public:
        BlazeFaceImpl(bool back_model_ = false);
        void _define_layers();
        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    };

    TORCH_MODULE(BlazeFace);
} // namespace blazeface