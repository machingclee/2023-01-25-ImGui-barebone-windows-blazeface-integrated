#include <string>

namespace global_config
{
    static int local_server_port                        = 8080;
    static std::string custom_app_path                  = "C:\\Users\\user\\Repos\\C++\\2023-01-25-ImGui-barebone-windows-blazeface-integrated\\build\\Release\\EyeCatching.exe";
    static const char* application_title                = "Eye Tracking Backend";
    static const char* company_name                     = "Eye Catching Ltd";
    static int VIDEO_CAPTURE_DEVICE_ENUM                = 1;
    static std::string FACE_DETECTOR_ANCHOR_TENSOR_PATH = "face_detector_anchors.pt";
    static std::string FACE_DETECTOR_WEIGHT_PATH        = "face_detector.pt";
    static std::string FACE_REGRESSOR_WEIGHT_PATH       = "face_regressor.pt";
    static std::string CV2_NAMED_WINDOW                 = "Facial Landmarks";
    static bool BACK_DETECTOR                           = false;
    static const float blazeface_min_score_thres        = 0.75;

} // namespace global_config