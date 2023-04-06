#include "utils/npy_utils.h"
#include "config/global_config.h"
#include "npy.hpp"

namespace npy_utils
{
    std::vector<double> load_1d_array_from_npy_file(std::string& npy_filepath)
    {
        std::vector<unsigned long> shape{ 896, 4 };
        bool fortran_order{ false };
        std::vector<double> data;
        npy::LoadArrayFromNumpy(npy_filepath, shape, fortran_order, data);
        return data;
    }

    torch::Tensor load_anchors_face()
    {
        std::vector<double> npy_array = load_1d_array_from_npy_file(global_config::FACE_DETECTOR_ANCHOR_TENSOR_PATH);

        torch::Tensor anchors = torch::randn({ 896, 4 }).toType(torch::kFloat32);

        for (int i = 0; i < 896; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int pos            = 4 * i + j;
                float target_value = npy_array[pos];
                anchors[i][j]      = target_value;
            }
        }
        std::cout << "anchors"
                  << "\n"
                  << anchors[0] << "\n"
                  << anchors[1] << "\n"
                  << anchors[2] << "\n"
                  << anchors[3] << "\n"
                  << anchors[4] << "\n"
                  << anchors[5] << "\n"
                  << anchors << "\n";
        return anchors;
    };
} // namespace npy_utils