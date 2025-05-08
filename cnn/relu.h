#pragma once

#include "pch.h"

class ReLU
{
public:
	ReLU() = default;

	RowMatrix forward(const RowMatrix& x);
	Tensor2D forward(const Tensor2D& x);
	Tensor4D forward(const Tensor4D& x);

	RowMatrix backward(const RowMatrix& dout);
	Tensor2D backward(const Tensor2D& dout);
	Tensor4D backward(const Tensor4D& dout);


private:
	MaskMatrix mask_;
};