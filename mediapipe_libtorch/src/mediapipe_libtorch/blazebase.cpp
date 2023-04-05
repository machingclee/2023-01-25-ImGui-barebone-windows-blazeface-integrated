#include "torch/torch.h"
#include "blazebase.h"
#include <tuple>
#include <string>
#include <stdexcept>
#include <variant>
#include <optional>
#include "utils/npy_utils.h"
#include "opencv2/opencv.hpp"
#include <iostream>
namespace blazebase
{
    torch::Tensor frame_to_input_tensor(cv::Mat& mat)
    {
        int height           = mat.rows;
        int width            = mat.cols;
        torch::Tensor result = torch::from_blob(mat.data, { 1, height, width, 3 }).permute({ 0, 3, 1, 2 }).toType(torch::kByte).toType(torch::kFloat32);
        return result;
    }

    std::tuple<cv::Mat, cv::Mat, float, std::tuple<int, int>> resize_pad(const cv::InputArray& img)
    {
        cv::Size size0 = img.size();
        int h1 = 1, w1 = 1, padh = 1, padw = 1;
        float scale = 1;
        int size0_0 = size0.height;
        int size0_1 = size0.width;
        if (size0_0 >= size0_1)
        {
            h1    = 256;
            w1    = static_cast<int>(256 * static_cast<float>(size0_1) / size0_0);
            padh  = 0;
            padw  = 256 - w1;
            scale = static_cast<float>(size0_1) / w1;
        }
        else
        {
            h1    = static_cast<int>(256 * static_cast<float>(size0_0) / size0_1);
            w1    = 256;
            padh  = 256 - h1;
            padw  = 0;
            scale = ((float)size0_0) / h1;
        }
        int padh1 = (int)(static_cast<float>(padh) / 2);
        int padh2 = (int)(static_cast<float>(padh) / 2) + (padh % 2);
        int padw1 = (int)(static_cast<float>(padw) / 2);
        int padw2 = (int)(static_cast<float>(padw) / 2) + (padw % 2);

        cv::Mat img1;
        cv::resize(img, img1, cv::Size(w1, h1));
        cv::copyMakeBorder(img1, img1, padh1, padh2, padw1, padw2, cv::BORDER_CONSTANT, cv::Scalar(0));
        std::tuple<int, int> pad = { static_cast<int>(padh1 * scale), static_cast<int>(padw1 * scale) };
        cv::Mat img2;
        cv::resize(img1, img2, cv::Size(128, 128));

        return { img1, img2, scale, pad };
    }

    torch::Tensor denormalize_detections(torch::Tensor detections, float scale, std::tuple<int, int> pad)
    {
        auto [pad0, pad1] = pad;
        detections.index_put_({ Slice(), 0 }, detections.index({ Slice(), 0 }) * scale * 256 - pad0);
        detections.index_put_({ Slice(), 1 }, detections.index({ Slice(), 1 }) * scale * 256 - pad1);
        detections.index_put_({ Slice(), 2 }, detections.index({ Slice(), 2 }) * scale * 256 - pad0);
        detections.index_put_({ Slice(), 3 }, detections.index({ Slice(), 3 }) * scale * 256 - pad1);
        detections.index_put_({ Slice(), Slice(4, None, 2) }, detections.index({ Slice(), Slice(4, None, 2) }) * scale * 256 - pad1);
        detections.index_put_({ Slice(), Slice(5, None, 2) }, detections.index({ Slice(), Slice(5, None, 2) }) * scale * 256 - pad0);
        return detections;
    }

    std::vector<char> nnModule::get_the_bytes(std::string filename)
    {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

        input.close();
        return bytes;
    }

    void nnModule::load_parameters(std::string pt_pth)
    {
        std::vector<char> f               = this->get_the_bytes(pt_pth);
        c10::Dict<IValue, IValue> weights = torch::pickle_load(f).toGenericDict();

        const torch::OrderedDict<std::string, at::Tensor>& model_params = this->named_parameters();
        std::vector<std::string> param_names;
        for (auto const& w : model_params)
        {
            param_names.push_back(w.key());
        }

        torch::NoGradGuard no_grad;
        for (auto const& w : weights)
        {
            std::string name = w.key().toStringRef();
            at::Tensor param = w.value().toTensor();

            if (std::find(param_names.begin(), param_names.end(), name) != param_names.end())
            {
                auto target_model_param = model_params.find(name);

                for (int i = 0; i < target_model_param->sizes().size(); i++)
                {
                    assert(target_model_param->sizes()[i] == param.sizes()[i]);
                }

                target_model_param->copy_(param);
            }
            else
            {
                std::cout << name << " does not exist among model parameters." << std::endl;
            };
        }
    }

    void nnModule::print_parameters(std::string file_path, bool with_weight)
    {

        std::ostringstream oss;

        for (const auto& pair : named_parameters())
        {
            oss << "[" << pair.key() << "] ";
            int shape_arr_size = pair.value().sizes().size();

            std::string size_tuple_str = "torch.Size([";
            for (int i = 0; i < shape_arr_size; i++)
            {
                std::string curr_dim_len = std::to_string(pair.value().sizes()[i]);
                size_tuple_str += curr_dim_len;
                if (i != (shape_arr_size - 1))
                {
                    size_tuple_str += ", ";
                }
            }
            size_tuple_str += "])";

            oss << size_tuple_str << "\n";

            if (with_weight)
            {
                oss << pair.value()
                    << "\n"
                    << "---------------"
                    << "\n";
            }
        }

        std::ofstream file;
        file.open(file_path);
        try
        {
            file << oss.str();
        }
        catch (std::exception err)
        {
            std::cout << err.what() << std::endl;
        }
        file.close();
    }

    BlazeBlockImpl::BlazeBlockImpl(
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        std::string act,
        bool use_skip_proj) : in_channels(in_channels),
                              out_channels(out_channels),
                              kernel_size(kernel_size),
                              stride(stride),
                              act(act),
                              use_skip_proj(use_skip_proj)
    {
        channel_pad = out_channels - in_channels;
        if (stride == 2)
        {
            max_pool = nn::MaxPool2d(nn::MaxPool2dOptions({ stride, stride }));
            padding  = 0;
        }
        else
        {
            padding = (int)((kernel_size - 1) / 2);
        }

        nn::Sequential convs_ = nn::Sequential();
        convs_->push_back(nn::Conv2d(nn::Conv2dOptions(in_channels, in_channels, kernel_size)
                                         .stride(stride)
                                         .padding(padding)
                                         .groups(in_channels)
                                         .bias(true)));
        convs_->push_back(nn::Conv2d(nn::Conv2dOptions(in_channels, out_channels, 1)
                                         .stride(1)
                                         .padding(0)
                                         .bias(true)));

        convs = register_module("convs", convs_);

        if (use_skip_proj)
        {
            skip_proj = register_module(
                "skip_proj",
                nn::Conv2d(nn::Conv2dOptions(in_channels, out_channels, 1)
                               .stride(1)
                               .padding(0)
                               .bias(true)));
        }
        else
        {
            skip_proj = nullptr;
        }

        if (act == "relu")
        {
            act_layer = nn::ReLU(nn::ReLUOptions(true));
        }
        else if ("prelu")
        {
            act_layer = register_module("act", nn::PReLU(nn::PReLUOptions().num_parameters(out_channels)));
        }
        else
        {
            throw std::exception("activation layer not implemented.");
        }
    }

    torch::Tensor BlazeBlockImpl::forward(torch::Tensor x)
    {
        torch::Tensor h;
        if (stride == 2)
        {
            if (kernel_size == 3)
            {
                h = nn::functional::pad(x, nn::functional::PadFuncOptions({ 0, 2, 0, 2 }).value(0));
            }
            else
            {
                h = nn::functional::pad(x, nn::functional::PadFuncOptions({ 1, 2, 1, 2 }).value(0));
            }
            x = this->max_pool(x);
        }
        else
        {
            h = x;
        }

        if (skip_proj)
        {
            x = skip_proj->forward(x);
        }
        else if (channel_pad > 0)
        {
            x = nn::functional::pad(x, nn::functional::PadFuncOptions({ 0, 0, 0, 0, 0, channel_pad }).value(0));
        }

        torch::Tensor y = convs->forward(h) + x;
        // y = reinterpret_cast<IHasForward*>(&act_layer)->forward(y);
        // I want to avoid the following:

        if (auto act_layer_ptr = std::get_if<nn::ReLU>(&act_layer))
        {
            y = (*act_layer_ptr)->forward(y);
        }
        else if (auto act_layer_ptr = std::get_if<nn::PReLU>(&act_layer))
        {
            y = (*act_layer_ptr)->forward(y);
        }

        return y;
    };

    FinalBlazeBlockImpl::FinalBlazeBlockImpl(int channels, int kernel_size)
    {
        nn::Sequential convs_;
        convs_->push_back(nn::Conv2d(
            nn::Conv2dOptions(channels, channels, kernel_size)
                .stride(2)
                .padding(0)
                .groups(channels)
                .bias(true)));
        convs_->push_back(nn::Conv2d(
            nn::Conv2dOptions(channels, channels, 1)
                .stride(1)
                .padding(0)
                .bias(true)));

        convs = register_module("convs", convs_);
        act   = nn::ReLU(nn::ReLUOptions(true));
    }

    torch::Tensor FinalBlazeBlockImpl::forward(torch::Tensor x)
    {
        torch::Tensor h = nn::functional::pad(x, nn::functional::PadFuncOptions({ 0, 2, 0, 2 }).value(0));
        return act(convs->forward(h));
    }

    torch::Device BlazeBase::_device()
    {
        // if (auto classifier_8_ptr = std::get_if<nn::Module>(&classifier_8))
        if (classifier_8)
        {
            return classifier_8->weight.device();
        }
        else
        {
            return torch::Device(torch::kCPU);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BlazeLandmark::extract_roi(
        cv::Mat frame,
        torch::Tensor xc,
        torch::Tensor yc,
        torch::Tensor theta,
        torch::Tensor scale)
    {
        float points_[2][4] = {
            { -1, -1, 1, 1 },
            { -1, 1, -1, 1 }
        };
        torch::Tensor points = torch::from_blob(points_, { 2, 4 }).toType(torch::kFloat32).view({ 1, 2, 4 });
        points               = points * scale.view({ -1, 1, 1 }) / 2;
        theta                = theta.view({ -1, 1, 1 });
        torch::Tensor R      = torch::cat(
            {
                torch::cat({ torch::cos(theta), -torch::sin(theta) }, 2),
                torch::cat({ torch::sin(theta), torch::cos(theta) }, 2),
            },
            1);

        torch::Tensor center = torch::cat({ xc.view({ -1, 1, 1 }), yc.view({ -1, 1, 1 }) }, 1);
        points               = R.matmul(points) + center;

        int res           = resolution.value_or(192);
        float _data[2][3] = {
            { 0, 0, res - 1 },
            { 0, res - 1, 0 }
        };
        cv::Mat points1 = cv::Mat(2, 3, CV_32FC1, &_data).t();

        std::vector<torch::Tensor> affines;
        std::vector<torch::Tensor> imgs;
        torch::Tensor imgs_;
        torch::Tensor affines_;

        for (int i = 0; i < points.sizes()[0]; i++)
        {

            torch::Tensor points_slice = points.index({ i, Slice(), Slice(None, 3) }).cpu();
            auto sizes                 = points_slice.sizes();
            auto pts                   = cv::Mat(sizes[0], sizes[1], CV_32FC1, points_slice.data_ptr()).t();

            cv::Mat M = cv::getAffineTransform(pts, points1);
            cv::Mat frame_transformed;
            cv::warpAffine(frame, frame_transformed, M, cv::Size(res, res));
            torch::Tensor img_ = torch::from_blob(frame_transformed.data, { res, res }).to(scale.device());
            imgs.push_back(img_);

            cv::Mat M_reverse;
            cv::invertAffineTransform(M, M_reverse);
            torch::Tensor affine = torch::from_blob(M_reverse.data, { M_reverse.rows, M_reverse.cols }).to(scale.device());
            affines.push_back(affine);

            if (imgs.size() > 0)
            {
                imgs_    = torch::stack(imgs).permute({ 0, 3, 1, 2 }).toType(torch::kFloat32) / 255.0f;
                affines_ = torch::stack(affines);
            }
            else
            {
                imgs_    = torch::zeros({ 0, 3, res, res }).to(scale.device());
                affines_ = torch::zeros({ 0, 2, 3 }).to(scale.device());
            }
        }
        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(imgs_, affines_, points);
    }

    torch::Tensor BlazeLandmark::denormalize_landmarks(torch::Tensor landmarks, torch::Tensor affines)
    {
        auto indices = std::initializer_list<at::indexing::TensorIndex>({ Slice(), Slice(), Slice(None, 2) });
        landmarks.index_put_(indices, resolution.value_or(192) * landmarks.index(indices));
        int num_landmarks = landmarks.sizes()[0];
        for (int i = 0; i < num_landmarks; i++)
        {
            auto landmark = landmarks[i];
            auto affine   = affines[i];
            // landmark = (affine[:, :2] @landmark[:, :2].T + affine[:, 2:]).T
            landmark = (affine.index({ Slice(), Slice(None, 2) }).matmul(landmark.index({ Slice(), Slice(None, 2) }).t()) + affine.index({ Slice(), Slice(2, None) })).t();
            landmarks.index_put_({ i, Slice(), Slice(None, 2) }, landmark);
        }
        return landmarks;
    }

    void BlazeDetector::load_anchors()
    {
        anchors = npy_utils::load_anchors_face();
    }

    void BlazeDetector::_preprocess(torch::Tensor& x)
    {
        x = x / 255.0;
    }

    std::vector<torch::Tensor> BlazeDetector::predict_on_batch(torch::Tensor& x)
    {
        assert(x.sizes()[1] == 3);
        assert(y_scale.has_value() && x.sizes()[2] == y_scale.value());
        assert(x_scale.has_value() && x.sizes()[3] == x_scale.value());
        assert(num_coords.has_value());
        x = x.to(this->_device());
        std::tuple<at::Tensor, at::Tensor> out;
        {
            torch::NoGradGuard no_grad;
            out = forward(x);
        }
        auto [raw_box, raw_score] = out;

        std::vector<torch::Tensor> detections = this->_tensors_to_detections(raw_box, raw_score, this->anchors.value());

        std::vector<torch::Tensor> filtered_detections;
        for (int i = 0; i < detections.size(); i++)
        {
            torch::Tensor faces_;
            std::vector<torch::Tensor> faces = this->_weighted_non_max_suppression(detections[i]);
            if (faces.size() > 0)
            {
                faces_ = torch::stack(faces);
            }
            else
            {
                faces_ = torch::zeros({ 0, num_coords.value() + 1 });
            }
            filtered_detections.push_back(faces_);
        }
        return filtered_detections;
    }

    torch::Tensor BlazeDetector::predict_on_image(torch::Tensor& img)
    {
        return this->predict_on_batch(img)[0];
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BlazeDetector::detection2roi(torch::Tensor detection)
    {
        assert(dy.has_value() && dscale.has_value() && theta0.has_value());

        torch::Tensor xc, yc, scale, x1, y1, x0, y0, theta;

        if (this->detection2roi_method == "box")
        {
            xc    = (detection.index({ Slice(), 1 }) + detection.index({ Slice(), 3 })) / 2;
            yc    = (detection.index({ Slice(), 0 }) + detection.index({ Slice(), 0 })) / 2;
            scale = detection.index({ Slice(), 3 }) - detection.index({ Slice(), 1 });
        }
        else if (this->detection2roi_method == "alignment" && kp1.has_value() && kp2.has_value())
        {
            xc = detection.index({ Slice(), 4 + 2 * kp1.value() });
            yc = detection.index({ Slice(), 4 + 2 * kp1.value() + 1 });
            x1 = detection.index({ Slice(), 4 + 2 * kp2.value() });
            y1 = detection.index({ Slice(), 4 + 2 * kp2.value() + 1 });
        }
        else
        {
            throw std::exception("detection method not supported, or k1, k2 are missing");
        }

        yc += dy.value() * scale;
        scale *= dscale.value();

        x0 = detection.index({ Slice(), 4 + 2 * kp1.value() });
        y0 = detection.index({ Slice(), 4 + 2 * kp1.value() + 1 });
        x1 = detection.index({ Slice(), 4 + 2 * kp2.value() });
        y1 = detection.index({ Slice(), 4 + 2 * kp2.value() + 1 });

        theta = torch::atan2(y0 - y1, x0 - x1) - theta0.value();
        return { xc, yc, scale, theta };
    }

    std::vector<torch::Tensor> BlazeDetector::_tensors_to_detections(
        torch::Tensor raw_box_tensor,
        torch::Tensor raw_score_tensor,
        torch::Tensor anchors)
    {
        assert(raw_box_tensor.ndimension() == 3);
        assert(num_anchors.has_value());
        assert(num_coords.has_value());
        assert(num_classes.has_value());
        assert(score_clipping_thresh.has_value());
        assert(min_score_thresh.has_value());
        assert(raw_box_tensor.sizes()[1] == num_anchors.value());
        assert(raw_box_tensor.sizes()[2] == num_coords.value());

        assert(raw_score_tensor.ndimension() == 3);
        assert(raw_score_tensor.sizes()[1] == num_anchors.value());
        assert(raw_score_tensor.sizes()[2] == num_classes.value());

        //         assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        assert(raw_box_tensor.sizes()[0] == raw_score_tensor.sizes()[0]);
        torch::Tensor detection_boxes  = this->_decode_boxes(raw_box_tensor, anchors);
        float thresh                   = this->score_clipping_thresh.value();
        raw_score_tensor               = raw_score_tensor.clamp(-thresh, thresh);
        torch::Tensor detection_scores = raw_score_tensor.sigmoid().squeeze(-1);
        torch::Tensor mask             = detection_scores >= min_score_thresh.value();

        // std::cout << "max score: " << detection_scores.max() << std::endl;

        // std::cout << "mask\n"
        //           << mask
        //           << std::endl;

        std::vector<torch::Tensor> output_detections;

        for (int i = 0; i < raw_box_tensor.sizes()[0]; i++)
        {
            // boxes = detection_boxes[i, mask[i]];
            torch::Tensor boxes = detection_boxes.index({ i, mask.index({ i }) });
            // scores = detection_scores[i, mask[i]].unsqueeze(dim = -1);
            std::cout << "boxes" << boxes << std::endl;
            torch::Tensor scores = detection_scores.index({ i, mask.index({ i }) }).unsqueeze(-1);
            std::cout << "scores" << scores << std::endl;
            // output_detections.append(torch.cat((boxes, scores), dim = -1));
            output_detections.push_back(torch::cat({ boxes, scores }, -1));
        }
        return output_detections;
    }

    std::vector<torch::Tensor> BlazeDetector::_weighted_non_max_suppression(torch::Tensor detections)
    {

        if (detections.sizes()[0] == 0)
        {
            return std::vector<torch::Tensor>();
        }
        std::vector<torch::Tensor> output_detections;
        torch::Tensor remaining = torch::argsort(detections.index({ Slice(), num_coords.value() }), -1, true);
        while (remaining.sizes()[0] > 0)
        {
            torch::Tensor detection   = detections.index({ remaining.index({ 0 }) });
            torch::Tensor first_box   = detection.index({ Slice(None, 4) });
            torch::Tensor other_boxes = detections.index({ remaining, Slice(None, 4) });
            torch::Tensor ious        = overlap_similarity(first_box, other_boxes);

            // mask = ious > self.min_suppression_threshold
            // overlapping = remaining[mask]
            // remaining = remaining[~mask]
            torch::Tensor mask        = ious >= min_suppression_threshold.value();
            torch::Tensor overlapping = remaining.index({ mask });
            torch::Tensor remaining   = remaining.index({ 1 - mask });

            torch::Tensor weighted_detection = detection.clone();

            if (overlapping.sizes()[0] > 1)
            {
                torch::Tensor coordinates = detections.index({ overlapping, Slice(None, num_coords.value()) });
                torch::Tensor scores      = detections.index({ overlapping, Slice(num_coords.value(), num_coords.value() + 1) });
                torch::Tensor total_score = scores.sum();
                torch::Tensor weighted    = (coordinates * scores).sum({ 0 }) / total_score;
                weighted_detection.index_put_({ Slice(None, num_coords.value()) }, weighted);
                weighted_detection.index_put_({ num_coords.value() }, total_score / overlapping.sizes()[0]);
            }
            output_detections.push_back(weighted_detection);
        }
        return output_detections;
    }

    torch::Tensor BlazeDetector::_decode_boxes(torch::Tensor raw_boxes, torch::Tensor anchors)
    {
        assert(w_scale.has_value() && h_scale.has_value() && num_keypoints.has_value());

        torch::Tensor boxes    = torch::zeros_like(raw_boxes);
        torch::Tensor x_center = raw_boxes.index({ "...", 0 }) / x_scale.value() * anchors.index({ Slice(), 2 }) +
                                 anchors.index({ Slice(), 0 });
        torch::Tensor y_center = raw_boxes.index({ "...", 1 }) / y_scale.value() * anchors.index({ Slice(), 3 }) +
                                 anchors.index({ Slice(), 1 });

        torch::Tensor w = raw_boxes.index({ "...", 2 }) / w_scale.value() * anchors.index({ Slice(), 2 });
        torch::Tensor h = raw_boxes.index({ "...", 3 }) / h_scale.value() * anchors.index({ Slice(), 3 });

        boxes.index_put_({ "...", 0 }, y_center - h / 2);
        boxes.index_put_({ "...", 1 }, x_center - w / 2);
        boxes.index_put_({ "...", 2 }, y_center + h / 2);
        boxes.index_put_({ "...", 3 }, x_center + w / 2);

        for (int k = 0; k < num_keypoints.value(); k++)
        {
            int offset               = 4 + k * 2;
            torch::Tensor keypoint_x = raw_boxes.index({ "...", offset }) / x_scale.value() * anchors.index({ Slice(), 2 }) +
                                       anchors.index({ Slice(), 0 });
            torch::Tensor keypoint_y = raw_boxes.index({ "...", offset + 1 }) / y_scale.value() * anchors.index({ Slice(), 3 }) +
                                       anchors.index({ Slice(), 1 });
            boxes.index_put_({ "...", offset }, keypoint_x);
            boxes.index_put_({ "...", offset + 1 }, keypoint_y);
        }
        return boxes;
    }

    torch::Tensor intersect(torch::Tensor box_a, torch::Tensor box_b)
    {
        int A                = box_a.sizes()[0];
        int B                = box_b.sizes()[0];
        torch::Tensor max_xy = torch::min(
            box_a.index({ Slice(), Slice(2, None) }).unsqueeze(1).expand({ A, B, 2 }),
            box_b.index({ Slice(), Slice(2, None) }).unsqueeze(0).expand({ A, B, 2 }));
        torch::Tensor min_xy = torch::max(
            box_a.index({ Slice(), Slice(None, 2) }).unsqueeze(1).expand({ A, B, 2 }),
            box_b.index({ Slice(), Slice(None, 2) }).unsqueeze(0).expand({ A, B, 2 }));
        torch::Tensor inter = torch::clamp(max_xy - min_xy, 0);
        return inter.index({ Slice(), Slice(), 0 }) * inter.index({ Slice(), Slice(), 1 });
    }

    torch::Tensor jaccard(torch::Tensor box_a, torch::Tensor box_b)
    {
        torch::Tensor inter  = intersect(box_a, box_b);
        torch::Tensor area_a = ((box_a.index({ Slice(), 2 }) - box_a.index({ Slice(), 0 })) *
                                (box_a.index({ Slice(), 3 }) - box_a.index({ Slice(), 1 })))
                                   .unsqueeze(1)
                                   .expand_as(inter);
        torch::Tensor area_b = ((box_b.index({ Slice(), 2 }) - box_b.index({ Slice(), 0 })) *
                                (box_b.index({ Slice(), 3 }) - box_b.index({ Slice(), 1 })))
                                   .unsqueeze(0)
                                   .expand_as(inter);
        torch::Tensor union_ = area_a + area_b - inter;
        return inter / union_;
    }

    torch::Tensor overlap_similarity(torch::Tensor box, torch::Tensor other_boxes)
    {
        return jaccard(box.unsqueeze(0), other_boxes).squeeze(0);
    }

} // namespace blazebase
