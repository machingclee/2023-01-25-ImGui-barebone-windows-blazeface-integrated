#pragma once
#include "pch.h"

namespace blazebase
{
    using namespace torch::indexing;
    using namespace torch;

    class nnModule : public nn::Module
    {
    public:
        std::vector<char> get_the_bytes(std::string filename);
        void load_parameters(std::string pt_pth);
        void print_parameters(std::string file_path, bool with_weight = false);
    };

    class BlazeBlockImpl : public nnModule
    {
    protected:
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        std::string act;
        nn::Conv2d skip_proj = nullptr;
        int channel_pad;
        int padding;
        nn::Sequential convs   = nullptr;
        nn::MaxPool2d max_pool = nullptr;
        std::variant<nn::ReLU, nn::PReLU> act_layer;
        bool use_skip_proj = false;

    public:
        BlazeBlockImpl(
            int in_channels,
            int out_channels,
            int kernel_size    = 3,
            int stride         = 1,
            std::string act    = "relu",
            bool use_skip_proj = false);

        BlazeBlockImpl& stride_(int stride)
        {
            this->stride = stride;
            return *this;
        }
        torch::Tensor forward(torch::Tensor x);
    };

    TORCH_MODULE(BlazeBlock);

    class BlazeBase : public nnModule
    {
    public:
        nn::Conv2d classifier_8 = nullptr;
        torch::Device _device();
    };

    class BlazeLandmark : public BlazeBase
    {
    public:
        std::optional<int> resolution;
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> extract_roi(
            cv::Mat frame,
            torch::Tensor xc,
            torch::Tensor yc,
            torch::Tensor theta,
            torch::Tensor scale);
        torch::Tensor denormalize_landmarks(torch::Tensor landmarks, torch::Tensor affines);
    };

    class BlazeDetector : public BlazeBase
    {
    protected:
        std::optional<torch::Tensor> anchors;
        std::optional<float> x_scale;
        std::optional<float> y_scale;
        std::optional<float> w_scale;
        std::optional<float> h_scale;
        std::optional<int> num_coords;
        std::optional<int> num_anchors;
        std::optional<int> num_classes;
        std::string detection2roi_method = "box";
        std::optional<int> kp1;
        std::optional<int> kp2;
        std::optional<float> dy;
        std::optional<float> dscale;
        std::optional<float> theta0;
        std::optional<float> score_clipping_thresh;
        std::optional<float> min_suppression_threshold;
        std::optional<float> min_score_thresh;
        std::optional<float> num_keypoints;

    public:
        virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor tensor) = 0;
        void load_anchors(std::string pt_path);
        void _preprocess(torch::Tensor& x);
        std::vector<torch::Tensor> predict_on_batch(torch::Tensor& x);
        torch::Tensor predict_on_image(torch::Tensor& img);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> detection2roi(torch::Tensor detection);
        std::vector<torch::Tensor> _tensors_to_detections(
            torch::Tensor raw_box_tensor,
            torch::Tensor raw_score_tensor,
            torch::Tensor anchors);
        std::vector<torch::Tensor> _weighted_non_max_suppression(torch::Tensor detections);
        torch::Tensor _decode_boxes(torch::Tensor raw_boxes, torch::Tensor anchors);
    };

    class FinalBlazeBlockImpl : public nnModule
    {
        nn::Sequential convs = nullptr;
        nn::ReLU act         = nullptr;

    public:
        FinalBlazeBlockImpl(int channels, int kernel_size = 3);
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(FinalBlazeBlock);

    class IHasForward
    {
    public:
        virtual torch::Tensor forward(torch::Tensor tensor) = 0;
    };

    torch::Tensor overlap_similarity(torch::Tensor box, torch::Tensor other_boxes);
    torch::Tensor denormalize_detections(torch::Tensor detections, float scale, std::tuple<int, int> pad);
    torch::Tensor frame_to_input_tensor(cv::Mat& mat);
    std::tuple<cv::Mat, cv::Mat, float, std::tuple<int, int>> resize_pad(cv::InputArray& img);
} // namespace blazebase