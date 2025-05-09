#pragma once

#include "pch.h"

class Convolution
{
public:
	Convolution(Tensor4D w, BiasVector b, size_t stride = 1, size_t padding = 0)
		: w_(w), b_(b), stride_(stride), padding_(padding)
	{
	}

	Tensor4D forward(const Tensor4D& x);
	Tensor4D backward(const Tensor4D& dout);


private:
	Tensor4D w_;
	BiasVector b_;
	size_t stride_;
	size_t padding_;
	Tensor4D::Dimensions x_shape_;
	Tensor2D col_x_;
	Tensor2D col_w_;
	Tensor4D dw_;
	BiasVector db_;
};
