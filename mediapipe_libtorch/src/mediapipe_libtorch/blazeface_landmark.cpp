#include "blazeface_landmark.h"

namespace blazeface_landmark
{
    BlazeFaceLandmarkImpl::BlazeFaceLandmarkImpl()
    {
        this->_define_layers();
    }

    void BlazeFaceLandmarkImpl::_define_layers()
    {
        backbone1 = register_module(
            "backbone1",
            nn::Sequential(
                nn::Conv2d(nn::Conv2dOptions(3, 16, 3).stride(2).padding(0).bias(true)),
                nn::PReLU(nn::PReLUOptions().num_parameters(16)),

                blazebase::BlazeBlock(16, 16, 3, 1, "prelu"),
                blazebase::BlazeBlock(16, 16, 3, 1, "prelu"),
                blazebase::BlazeBlock(16, 32, 3, 2, "prelu"),

                blazebase::BlazeBlock(32, 32, 3, 1, "prelu"),
                blazebase::BlazeBlock(32, 32, 3, 1, "prelu"),
                blazebase::BlazeBlock(32, 64, 3, 2, "prelu"),

                blazebase::BlazeBlock(64, 64, 3, 1, "prelu"),
                blazebase::BlazeBlock(64, 64, 3, 1, "prelu"),
                blazebase::BlazeBlock(64, 128, 3, 2, "prelu"),

                blazebase::BlazeBlock(128, 128, 3, 1, "prelu"),
                blazebase::BlazeBlock(128, 128, 3, 1, "prelu"),
                blazebase::BlazeBlock(128, 128, 3, 2, "prelu"),

                blazebase::BlazeBlock(128, 128, 3, 1, "prelu"),
                blazebase::BlazeBlock(128, 128, 3, 1, "prelu")));

        backbone2a = register_module(
            "backbone2a",
            nn::Sequential(
                blazebase::BlazeBlock(128, 128, 3, 2, "prelu"),
                blazebase::BlazeBlock(128, 128, 3, 1, "prelu"),
                blazebase::BlazeBlock(128, 128, 3, 1, "prelu"),
                nn::Conv2d(nn::Conv2d(nn::Conv2dOptions(128, 32, 1).padding(0).bias(true))),
                nn::PReLU(nn::PReLUOptions().num_parameters(32)),
                blazebase::BlazeBlock(32, 32, 3, 1, "prelu"),
                nn::Conv2d(nn::Conv2dOptions(32, 1404, 3).padding(0).bias(true))));

        backbone2b = register_module(
            "backbone2b",
            nn::Sequential(
                blazebase::BlazeBlock(128, 128, 3, 2, "prelu"),
                nn::Conv2d(nn::Conv2dOptions(128, 32, 1).padding(0).bias(true)),
                nn::PReLU(nn::PReLUOptions().num_parameters(32)),
                blazebase::BlazeBlock(32, 32, 3, 1, "prelu"),
                nn::Conv2d(nn::Conv2dOptions(32, 1, 3).padding(0).bias(true))));
    }

    std::tuple<torch::Tensor, torch::Tensor> BlazeFaceLandmarkImpl::forward(torch::Tensor x)
    {
        if (x.sizes()[0] == 0)
        {
            return { torch::zeros({ 0 }), torch::zeros({ 0, 468, 3 }) };
        }

        x = nn::functional::pad(x, nn::functional::PadFuncOptions({ 0, 1, 0, 1 }).value(0));
        x = backbone1->forward(x);
        torch::Tensor landmarks = backbone2a->forward(x).view({ -1, 468, 3 }) / 192;
        torch::Tensor flag = backbone2b->forward(x).sigmoid().view({ -1 });
        return { flag, landmarks };
    }
} // namespace blazeface_landmark