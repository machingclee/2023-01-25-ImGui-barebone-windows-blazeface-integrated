#include "utils/npy_utils.h"
#include "config/global.h"
#include "npy.hpp"

namespace npy_utils
{
std::vector<double> load_1d_array_from_npy_file(std::string& npy_filepath)
{
    std::vector<unsigned long> shape{};
    bool fortran_order{ false };
    std::vector<double> data;
    npy::LoadArrayFromNumpy(npy_filepath, shape, fortran_order, data);
    return data;
}

torch::Tensor load_anchors_face()
{
    std::vector<double> npy_array = load_1d_array_from_npy_file(Global::anchors_face_filepath);

    torch::Tensor anchors = torch::randn({ 896, 4 });

    for (int i = 0; i < 896; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            int pos = 4 * i + j;
            float target_value = npy_array[pos];
            anchors[i][j] = target_value;
        }
    }

    return anchors;
};
} // namespace npy_utils