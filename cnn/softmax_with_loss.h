#pragma once

#include "pch.h"


class SoftmaxWithLoss
{
public:
	SoftmaxWithLoss() = default;
	~SoftmaxWithLoss()
	{
		delete t_;
		delete y_;
		t_ = nullptr;
		y_ = nullptr;
	}
	float forward(const RowMatrix& x, const LabelVector& t);
	float forward(const Tensor2D& x, const LabelVector& t);

	RowMatrix backward(const float dout = 1);

private:
	float loss_ = 0;
	LabelVector* t_ = nullptr;
	RowMatrix* y_ = nullptr;
};
