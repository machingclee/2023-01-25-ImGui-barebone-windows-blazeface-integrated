#include "visualize.h"

namespace visualize
{
    void draw_landmarks(
        cv::Mat& frame,
        torch::Tensor points,
        const std::vector<std::tuple<int, int>>& connections,
        cv::Scalar color,
        int size)
    {
        // points = points[:,:2]
        points = points.index({ Slice(), Slice(None, 2) });
        // for point in points:
        for (int i = 0; i < points.sizes()[0]; i++)
        { //     x, y = point
            torch::Tensor point = points[i];
            int* pt = (int*)point.data_ptr();
            int x = pt[0];
            int y = pt[1];
            cv::circle(frame, cv::Point2i({ x, y }), size, color, size);
            for (auto& connection : connections)
            {
                auto [i, j] = connection;
                int* pointA = (int*)points[i].data_ptr();
                int* pointB = (int*)points[j].data_ptr();
                int x0 = pointA[0];
                int y0 = pointA[1];
                int x1 = pointB[0];
                int y1 = pointB[1];

                cv::line(frame, cv::Point2i({ x0, y0 }), cv::Point2i({ x1, y1 }), cv::Scalar({ 0, 0, 0 }), size);
            }
        }
    }
} // namespace visualize