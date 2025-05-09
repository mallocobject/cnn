//#include "net.h"
//#include <cmath>
//
//Net::Net(const std::vector<int>& input_dim, const Dict& conv_param_1, const Dict& conv_param_2, const Dict& conv_param_3, const Dict& conv_param_4, const Dict& conv_param_5, const Dict& conv_param_6, int hidden_size, int output_size)
//{
//
//	std::vector<int> wight_init_scales = {
//		1 * 3 * 3,
//		16 * 3 * 3,
//		16 * 3 * 3,
//		32 * 3 * 3,
//		32 * 3 * 3,
//		64 * 3 * 3,
//		64 * 4 * 4,
//		hidden_size
//	};
//	for (int& i : wight_init_scales)
//	{
//		i = sqrt(2.0 / i); // 使用ReLU的情况下推荐的初始值
//	}
//	int pre_channel_num = input_dim[0];
//
//	params_["w1"] = wight_init_scales[0] * Eigen::
//}
