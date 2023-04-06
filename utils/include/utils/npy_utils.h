#pragma once
#include "pch.h"
#include <vector>
#include <string>

namespace npy_utils
{

std::vector<double> load_1d_array_from_npy_file(std::string& npy_filepath);

torch::Tensor load_anchors_face();
} // namespace npy_utils
