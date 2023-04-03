#pragma once
#include "blazebase.h"
#include "torch/torch.h"

namespace blazeface_landmark
{
    using namespace torch;
    class BlazeFaceLandmarkImpl : public blazebase::BlazeLandmark
    {
        int resolution = 192;
        nn::Sequential backbone1 = nullptr;
        nn::Sequential backbone2a = nullptr;
        nn::Sequential backbone2b = nullptr;

        void _define_layers();

    public:
        BlazeFaceLandmarkImpl();
        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    };

    TORCH_MODULE(BlazeFaceLandmark);
} // namespace blazeface_landmark