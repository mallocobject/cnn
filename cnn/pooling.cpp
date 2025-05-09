#include "pooling.h"
#include "utils.h"
#include <iostream>

Tensor4D Pooling::forward(const Tensor4D& x)
{
	for (auto& idx : x.dimensions())
	{
		if (idx == 0)
		{
			throw std::invalid_argument("Affine  ‰»Îæÿ’ÛŒ™ø’");
		}
	}
	eidx N = x.dimension(0);
	eidx C = x.dimension(1);
	eidx H = x.dimension(2);
	eidx W = x.dimension(3);
	eidx output_h = (H + 2 * padding_ - pool_h_) / stride_ + 1;
	eidx output_w = (W + 2 * padding_ - pool_h_) / stride_ + 1;
	Tensor2D col = Utils::im2col(x, pool_h_, pool_w_, stride_, padding_);

	eidx pro = col.dimension(0) * col.dimension(1);
	eidx cols = pool_h_ * pool_w_;
	eidx rows = pro / cols;
	Tensor2D col_reshaped = col.reshape(vec(2, rows, cols));
	Eigen::Tensor<float, 1, Eigen::RowMajor> out(rows);
	arg_max_.resize(rows);
	for (int i = 0; i < rows; i++)
	{
		float max_val = col_reshaped(i, 0);
		eidx idx = 0;
		for (int j = 1; j < cols; j++)
		{
			if (col_reshaped(i, j) > max_val)
			{
				max_val = col_reshaped(i, j);
				idx = j;
			}
		}
		out(i) = max_val;
		arg_max_(i) = idx;
	}

	x_shape_ = x.dimensions();

	Tensor4D result = out.reshape(vec(4, N, output_h, output_w, C)).shuffle(vec(4, 0, 3, 1, 2));

	return result;
}

Tensor4D Pooling::backward(const Tensor4D& dout)
{
	for (auto& idx : dout.dimensions())
	{
		if (idx == 0)
		{
			throw std::invalid_argument("Affine  ‰»Îæÿ’ÛŒ™ø’");
		}
	}
	eidx N = dout.dimension(0);
	eidx C = dout.dimension(1);
	eidx output_h = dout.dimension(2);
	eidx output_w = dout.dimension(3);

	eidx pool_size = pool_h_ * pool_w_;
	Eigen::Tensor<float, 1, Eigen::RowMajor> dout_reshaped = dout.shuffle(vec(4, 0, 2, 3, 1))
		.reshape(vec(1, N * C * output_h * output_w));
	Tensor2D dcol_tensor(dout_reshaped.dimension(0), pool_size);
	dcol_tensor.setZero();

	for (int i = 0; i < dcol_tensor.dimension(0); i++)
	{
		eidx j = arg_max_(i);
		dcol_tensor(i, j) = dout_reshaped(i);
	}

	auto dcol = dcol_tensor.reshape(vec(2, N * output_h * output_w, C * pool_size));

	return Utils::col2im(dcol, x_shape_, pool_h_, pool_w_, stride_, padding_);
}
