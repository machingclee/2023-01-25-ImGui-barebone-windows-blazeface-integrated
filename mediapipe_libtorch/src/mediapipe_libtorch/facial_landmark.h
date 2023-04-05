#pragma once

#include "blazebase.h"
#include "blazeface.h"
#include "blazeface_landmark.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "visualize.h"

namespace facial_landmark
{
    using namespace torch::indexing;
    void start_detection();
} // namespace facial_landmark
