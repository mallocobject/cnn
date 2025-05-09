#include "convolution.h"
#include "utils.h"

#include <iostream>

Tensor4D Convolution::forward(const Tensor4D& x)
{
	for (auto& idx : x.dimensions())
	{
		if (idx == 0)
		{
			throw std::invalid_argument("Affine 输入矩阵为空");
		}
	}
	eidx N = x.dimension(0);
	eidx C = x.dimension(1);
	eidx H = x.dimension(2);
	eidx W = x.dimension(3);
	eidx FN = w_.dimension(0);
	eidx FC = w_.dimension(1);
	eidx FH = w_.dimension(2);
	eidx FW = w_.dimension(3);

	if (C != FC)
	{
		throw std::invalid_argument("输入通道不相等");
	}

	eidx output_h = (H + 2 * padding_ - FH) / stride_ + 1;
	eidx output_w = (W + 2 * padding_ - FW) / stride_ + 1;

	col_x_ = Utils::im2col(x, FH, FW, stride_, padding_);
	col_w_ = w_.reshape(vec(2, FN, C * FH * FW));  // 未转置，与py不同
	x_ = x;

	auto col_w_matrix = t2m(col_w_);
	auto col_x_matrix = t2m(col_x_);

	std::cout << col_x_matrix.rows() << col_x_matrix.cols() << std::endl;
	std::cout << col_w_matrix.transpose().rows() << col_w_matrix.transpose().cols() << std::endl;

	eidx r_n = col_x_matrix.rows();
	BiasVector b_extent(FN * r_n);
	// 使用块操作和指针操作来高效复制元素
	for (int i = 0; i < FN; ++i)
	{
		b_extent.segment(i * r_n, r_n) = BiasVector::Constant(r_n, b_(i));
	}

	RowMatrix out = (col_x_matrix * col_w_matrix.transpose()).colwise() + b_extent;
	Tensor4D result = m2t(out).reshape(vec(4, N, output_h, output_w, FC)).shuffle(vec(4, 0, 3, 1, 2));

	return result;
}

Tensor4D Convolution::backward(const Tensor4D& dout)
{
	eidx FN = w_.dimension(0);
	eidx FC = w_.dimension(1);
	eidx FH = w_.dimension(2);
	eidx FW = w_.dimension(3);
	eidx N = dout.dimension(0);
	eidx C = dout.dimension(1);
	eidx output_h = dout.dimension(2);
	eidx output_w = dout.dimension(3);

	eidx pro = C * N * output_h * output_w;

	Tensor2D dout_tensor = dout.shuffle(vec(4, 1, 0, 2, 3)).reshape(vec(2, C, pro / C));
	auto dout_matrix = t2m(dout_tensor);
	db_ = dout_matrix.rowwise().sum();

	std::cout << db_.rows() << std::endl;
	std::cout << db_ << std::endl;

	Tensor2D dout_flat = dout.shuffle(vec(4, 0, 2, 3, 1)).reshape(vec(2, pro / FN, FN));
	auto dout_flat_matrix = t2m(dout_flat);
	auto col_w_matrix = t2m(col_w_);
	RowMatrix dcol = dout_flat_matrix * col_w_matrix;
	auto dcol_tensor = m2t(dcol);
	return Utils::col2im(dcol_tensor, x_.dimensions(), FH, FW, stride_, padding_);
}
