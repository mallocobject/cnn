#pragma once

#include "pch.h"

class Pooling
{
public:
	Pooling(size_t pool_h, size_t pool_w, size_t stride = 1, size_t padding = 0)
		:pool_h_(pool_h), pool_w_(pool_w), stride_(stride), padding_(padding)
	{
	}

	Tensor4D forward(const Tensor4D& x);

	Tensor4D backward(const Tensor4D& dout);

private:
	size_t pool_h_;
	size_t pool_w_;
	size_t stride_;
	size_t padding_;

	Tensor4D::Dimensions x_shape_;
	ArgVector arg_max_;
};
