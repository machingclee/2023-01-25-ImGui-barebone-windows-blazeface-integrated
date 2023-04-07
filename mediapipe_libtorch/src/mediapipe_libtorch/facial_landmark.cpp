#include "facial_landmark.h"
#include "visualize.h"
#include "config/global_config.h"

namespace facial_landmark
{
    // torch::Device device = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    void start_detection()
    {
        blazeface::BlazeFace face_detector = blazeface::BlazeFace(global_config::BACK_DETECTOR);
        face_detector->to(torch::kFloat32);
        face_detector->load_parameters(global_config::FACE_DETECTOR_WEIGHT_PATH);
        face_detector->load_anchors(global_config::FACE_DETECTOR_ANCHOR_TENSOR_PATH);

        auto face_regressor = blazeface_landmark::BlazeFaceLandmark();
        face_regressor->to(torch::kFloat32);
        face_regressor->load_parameters(global_config::FACE_REGRESSOR_WEIGHT_PATH);

        cv::VideoCapture capture(global_config::VIDEO_CAPTURE_DEVICE_ENUM);

        bool hasFrame = false;
        cv::Mat frame;

        if (!capture.isOpened())
        {
            throw std::exception("Unable to open camera.");
        }

        while (true)
        {
            capture.read(frame);
            if (frame.empty())
            {
                throw std::exception("Blank frame grabbed.");
            }
            assert(frame.channels() == 3);

            cv::flip(frame, frame, 1);
            cv::flip(frame, frame, 0);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            auto [_, img2, scale, pad]      = blazebase::resize_pad(frame);
            torch::Tensor img2_             = blazebase::frame_to_input_tensor(img2);
            auto normalized_face_detections = face_detector->predict_on_image(img2_);

            auto face_detections         = blazebase::denormalize_detections(normalized_face_detections, scale, pad);
            auto [xc, yc, scale_, theta] = face_detector->detection2roi(face_detections.to(torch::kCPU));

            auto [img, affine, box]            = face_regressor->extract_roi(frame, xc, yc, theta, scale_);
            auto [flags, normalized_landmarks] = face_regressor->forward(img);
            auto landmarks                     = face_regressor->denormalize_landmarks(normalized_landmarks.to(torch::kCPU), affine);
            
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

            for (int i = 0; i < flags.sizes()[0]; i++)
            {
                torch::Tensor landmark = landmarks[i];
                float flag             = flags[i].item<float>();
                if (flag > 0.5)
                {
                    visualize::draw_landmarks(
                        frame,
                        landmark,
                        visualize::FACE_CONNECTIONS,
                        1);
                }
            }
            cv::flip(frame, frame, 0);
            cv::imshow(global_config::CV2_NAMED_WINDOW, frame);

            if (cv::waitKey(5) >= 0)
            {
                break;
            }
        }
    }
} // namespace facial_landmark