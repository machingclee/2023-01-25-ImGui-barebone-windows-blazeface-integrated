#include "facial_landmark.h"
#include "visualize.h"

namespace facial_landmark
{
    // torch::Device device = torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    static bool back_detector                     = false;
    static std::string CV2_NAMED_WINDOW           = "Facial Landmarks";
    static int VIDEO_CAPTURE_DEVICE_ENUM          = 1;
    static std::string FACE_DETECTOR_WEIGHT_PATH  = "C:\\Users\\user\\Repos\\C++\\2023-01-25-ImGui-barebone-windows-blazeface-integrated\\mediapipe_libtorch\\src\\mediapipe_libtorch\\face_detector.pt";
    static std::string FACE_REGRESSOR_WEIGHT_PATH = "C:\\Users\\user\\Repos\\C++\\2023-01-25-ImGui-barebone-windows-blazeface-integrated\\mediapipe_libtorch\\src\\mediapipe_libtorch\\face_regressor.pt";

    void start_detection()
    {
        blazeface::BlazeFace face_detector = blazeface::BlazeFace(back_detector);
        face_detector->to(torch::kFloat32);
        face_detector->load_parameters(FACE_DETECTOR_WEIGHT_PATH);
        face_detector->eval();
        face_detector->load_anchors();

        auto face_regressor = blazeface_landmark::BlazeFaceLandmark();
        face_regressor->to(torch::kFloat32);
        face_regressor->load_parameters(FACE_REGRESSOR_WEIGHT_PATH);
        face_regressor->eval();

        cv::VideoCapture capture(1);

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
            cv::Mat frame_for_prediction;
            cv::cvtColor(frame, frame_for_prediction, cv::COLOR_BGR2RGB);
            frame_for_prediction.convertTo(frame_for_prediction, CV_32FC3, 1 / 255.0);

            auto [_, img2, scale, pad]         = blazebase::resize_pad(frame_for_prediction);
            torch::Tensor img2_                = blazebase::frame_to_input_tensor(img2);
            auto normalized_face_detections    = face_detector->predict_on_image(img2_);
            auto face_detections               = blazebase::denormalize_detections(normalized_face_detections, scale, pad);
            auto [xc, yc, scale_, theta]       = face_detector->detection2roi(face_detections.to(torch::kCPU));
            auto [img, affine, box]            = face_regressor->extract_roi(frame, xc, yc, theta, scale_);
            auto [flags, normalized_landmarks] = face_regressor->forward(img);
            auto landmarks                     = face_regressor->denormalize_landmarks(normalized_landmarks.to(torch::kCPU), affine);

            for (int i = 0; i < flags.sizes()[0]; i++)
            {
                torch::Tensor landmark = landmarks[i];
                torch::Tensor flag     = flags[i];
                visualize::draw_landmarks(
                    frame,
                    landmark.index({ Slice(), Slice(None, 2) }),
                    visualize::FACE_CONNECTIONS,
                    1);
            }

            cv::imshow(CV2_NAMED_WINDOW, frame);

            if (cv::waitKey(5) >= 0)
            {
                break;
            }
        }
    }
} // namespace facial_landmark