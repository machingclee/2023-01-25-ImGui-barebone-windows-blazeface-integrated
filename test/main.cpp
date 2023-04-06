#pragma once

#include "torch/torch.h"
#include <vector>
#include "npy.hpp"
#include <iostream>
#include "utils/npy_utils.h"
#include "mediapipe_libtorch/facial_landmark.h"
#include <string>

using namespace torch::indexing;
int main()
{
    facial_landmark::start_detection();
}
