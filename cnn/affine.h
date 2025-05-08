#pragma once

#include "pch.h"

class Affine
{
public:
	Affine(const RowMatrix& w, const BiasVector& b)
		:w_(w), b_(b)
	{
	}

	RowMatrix forward(const RowMatrix& x);
	RowMatrix forward(const Tensor2D& x);
	RowMatrix forward(const Tensor4D& x);
	Tensor2D backward_2D(const RowMatrix& dout);
	Tensor4D backward_4D(const RowMatrix& dout);


private:
	RowMatrix w_;
	BiasVector b_;
	RowMatrix x_;
	Tensor4D::Dimensions original_x_shape_;
	RowMatrix dw_;
	BiasVector db_;
};