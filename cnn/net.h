#pragma once
#include "utils.h"
#include "pch.h"
#include "relu.h"
#include "softmax_with_loss.h"
#include "dropout.h"
#include "affine.h"
#include "convolution.h"
#include "pooling.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "pch.h"

using Dict = std::unordered_map<std::string, int>;

class Net
{
public:
	Net(const std::vector<int>& input_dim = { 1, 28, 28 },
		const Dict& conv_param_1 = {
			{"filter_num", 16}, {"filter_size" , 3}, {"padding" , 1}, {"stride" , 1}
		},
		const Dict& conv_param_2 = {
			{"filter_num", 16}, {"filter_size" , 3}, {"padding" , 1}, {"stride" , 1}
		},
		const Dict& conv_param_3 = {
			{"filter_num", 32}, {"filter_size" , 3}, {"padding" , 1}, {"stride" , 1}
		},
		const Dict& conv_param_4 = {
			{"filter_num", 32}, {"filter_size" , 3}, {"padding" , 2}, {"stride" , 1}
		},
		const Dict& conv_param_5 = {
			{"filter_num", 64}, {"filter_size" , 3}, {"padding" , 1}, {"stride" , 1}
		},
		const Dict& conv_param_6 = {
			{"filter_num", 64}, {"filter_size" , 3}, {"padding" , 1}, {"stride" , 1}
		},
		int hidden_size = 50, int output_size = 10);



private:
	std::unordered_map<std::string, Tensor4D> params_w_4D_;
	std::unordered_map<std::string, Tensor2D> params_w_2D_;
	std::unordered_map<std::string, BiasVector> params_b_;
};