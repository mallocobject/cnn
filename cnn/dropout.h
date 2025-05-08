#pragma once

#include "pch.h"

class Dropout
{
public:
	Dropout(float dropout_ratio = 0.5);

	RowMatrix forward(const RowMatrix& x, bool is_train = true);
	Tensor2D forward(const Tensor2D& x, bool is_train = true);
	Tensor4D forward(const Tensor4D& x, bool is_train = true);

	RowMatrix backward(const RowMatrix& dout);
	Tensor2D backward(const Tensor2D& dout);
	Tensor4D backward(const Tensor4D& dout);

private:
	float dropout_ratio_;
	MaskMatrix mask_;
};