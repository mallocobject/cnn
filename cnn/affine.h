#pragma once

#include "pch.h"

class Affine
{
public:
	Affine(const Tensor2D& w, const BiasVector& b)
		:w_(w), b_(b)
	{
	}

	Tensor2D forward(const Tensor2D& x);
	Tensor2D forward(const Tensor4D& x);
	TensorVariant backward(const Tensor2D& dout);


private:
	Tensor2D w_;
	BiasVector b_;
	Tensor2D x_;
	Tensor4D::Dimensions original_x_shape_;
	bool input_is_4D = false;
	Tensor2D dw_;
	BiasVector db_;
};