#include "blazeface.h"
#include "config/global_config.h"

namespace blazeface
{
    BlazeFaceImpl::BlazeFaceImpl(bool back_model_)
    {
        num_classes           = 1;
        num_anchors           = 896;
        num_coords            = 16;
        score_clipping_thresh = 100.0;
        back_model            = back_model_;

        if (back_model)
        {
            x_scale          = 256.0;
            y_scale          = 256.0;
            h_scale          = 256.0;
            w_scale          = 256.0;
            min_score_thresh = 0.65;
        }
        else
        {
            x_scale          = 128.0;
            y_scale          = 128.0;
            h_scale          = 128.0;
            w_scale          = 128.0;
            min_score_thresh = global_config::blazeface_min_score_thres;
        }
        min_suppression_threshold = 0.3;
        num_keypoints             = 6;

        detection2roi_method = "box";
        kp1                  = 1;
        kp2                  = 0;
        theta0               = 0;
        dscale               = 1.5;
        dy                   = 0;

        this->_define_layers();
    }

    void BlazeFaceImpl::_define_layers()
    {
        if (back_model)
        {
            nn::Sequential backbone_ = nn::Sequential();
            backbone_->push_back(nn::Conv2d(nn::Conv2dOptions(3, 24, 5).stride(2).padding(0).bias(true)));
            backbone_->push_back(nn::ReLU(nn::ReLUOptions(true)));

            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3, 2));

            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone_->push_back(blazebase::BlazeBlock(24, 48, 3, 2));

            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 48, 3));
            backbone_->push_back(blazebase::BlazeBlock(48, 96, 3, 2));

            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone_->push_back(blazebase::BlazeBlock(96, 96, 3));

            backbone = register_module(
                "backbone",
                backbone_);

            final = register_module("final", blazebase::FinalBlazeBlock(96));

            classifier_8  = register_module("classifier_8", nn::Conv2d(nn::Conv2dOptions(96, 2, 1).bias(true)));
            classifier_16 = register_module("classifier_16", nn::Conv2d(nn::Conv2dOptions(96, 6, 1).bias(true)));

            regressor_8  = register_module("regressor_8", nn::Conv2d(nn::Conv2dOptions(96, 32, 1).bias(true)));
            regressor_16 = register_module("regressor_16", nn::Conv2d(nn::Conv2dOptions(96, 96, 1).bias(true)));
        }
        else
        {
            nn::Sequential backbone1_ = nn::Sequential();
            nn::Sequential backbone2_ = nn::Sequential();

            backbone1_->push_back(nn::Conv2d(nn::Conv2dOptions(3, 24, 5).stride(2).padding(0).bias(true)));
            backbone1_->push_back(nn::ReLU(nn::ReLUOptions(true)));

            backbone1_->push_back(blazebase::BlazeBlock(24, 24, 3));
            backbone1_->push_back(blazebase::BlazeBlock(24, 28, 3));
            backbone1_->push_back(blazebase::BlazeBlock(28, 32, 3, 2));

            backbone1_->push_back(blazebase::BlazeBlock(32, 36, 3));
            backbone1_->push_back(blazebase::BlazeBlock(36, 42, 3));
            backbone1_->push_back(blazebase::BlazeBlock(42, 48, 3, 2));

            backbone1_->push_back(blazebase::BlazeBlock(48, 56, 3));
            backbone1_->push_back(blazebase::BlazeBlock(56, 64, 3));
            backbone1_->push_back(blazebase::BlazeBlock(64, 72, 3));
            backbone1_->push_back(blazebase::BlazeBlock(72, 80, 3));
            backbone1_->push_back(blazebase::BlazeBlock(80, 88, 3));

            backbone2_->push_back(blazebase::BlazeBlock(88, 96, 3, 2));
            backbone2_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone2_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone2_->push_back(blazebase::BlazeBlock(96, 96, 3));
            backbone2_->push_back(blazebase::BlazeBlock(96, 96, 3));

            backbone1 = register_module(
                "backbone1",
                backbone1_);

            backbone2 = register_module(
                "backbone2",
                backbone2_);

            classifier_8  = register_module("classifier_8", nn::Conv2d(nn::Conv2dOptions(88, 2, 1).bias(true)));
            classifier_16 = register_module("classifier_16", nn::Conv2d(nn::Conv2dOptions(96, 6, 1).bias(true)));

            regressor_8  = register_module("regressor_8", nn::Conv2d(nn::Conv2dOptions(88, 32, 1).bias(true)));
            regressor_16 = register_module("regressor_16", nn::Conv2d(nn::Conv2dOptions(96, 96, 1).bias(true)));
        }
    }
    std::tuple<torch::Tensor, torch::Tensor> BlazeFaceImpl::forward(torch::Tensor x)
    {
        x     = nn::functional::pad(x, nn::functional::PadFuncOptions({ 1, 2, 1, 2 }).value(0));
        int b = x.sizes()[0];
        torch::Tensor h;

        if (back_model)
        {
            x = backbone->forward(x);
            h = final->forward(x);
        }
        else
        {
            x = backbone1->forward(x);
            h = backbone2->forward(x);
        }

        torch::Tensor c1 = classifier_8->forward(x);
        c1               = c1.permute({ 0, 2, 3, 1 });
        c1               = c1.reshape({ b, -1, 1 });

        torch::Tensor c2 = classifier_16->forward(h);
        c2               = c2.permute({ 0, 2, 3, 1 });
        c2               = c2.reshape({ b, -1, 1 });

        torch::Tensor c = torch::cat({ c1, c2 }, 1);

        torch::Tensor r1 = regressor_8->forward(x);
        r1               = r1.permute({ 0, 2, 3, 1 });
        r1               = r1.reshape({ b, -1, 16 });

        torch::Tensor r2 = regressor_16->forward(h);
        r2               = r2.permute({ 0, 2, 3, 1 });
        r2               = r2.reshape({ b, -1, 16 });

        torch::Tensor r = torch::cat({ r1, r2 }, 1);

        return { r, c };
    }

} // namespace blazeface