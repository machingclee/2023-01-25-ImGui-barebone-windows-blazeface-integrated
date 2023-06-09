#include "visualize.h"

namespace visualize
{
    void draw_landmarks(
        cv::Mat& frame,
        torch::Tensor& landmark,
        const std::vector<std::tuple<int, int>>& connections,
        int size,
        cv::Scalar color)
    {
        landmark = landmark.index({ Slice(), Slice(None, 2) });

        int num_of_points = landmark.sizes()[0];
        for (int i = 0; i < num_of_points; i++)
        {
            torch::Tensor point = landmark[i];
            int x               = point[0].item<int>();
            int y               = point[1].item<int>();
            cv::circle(frame, cv::Point2i({ x, y }), size, color, size);
        }
        for (auto& connection : connections)
        {
            auto [i, j]          = connection;
            torch::Tensor pointA = landmark[i];
            torch::Tensor pointB = landmark[j];
            int x0               = pointA[0].item<int>();
            int y0               = pointA[1].item<int>();
            int x1               = pointB[0].item<int>();
            int y1               = pointB[1].item<int>();

            cv::line(frame, cv::Point2i({ x0, y0 }), cv::Point2i({ x1, y1 }), cv::Scalar({ 0, 0, 0 }), size);
        }
    }
} // namespace visualize