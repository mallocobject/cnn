#include "affine.h"

#include <iostream>

RowMatrix Affine::forward(const RowMatrix& x)
{
	eidx rows = x.rows();
	eidx cols = x.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
	}
	x_ = x;
	RowMatrix result = (x_ * w_).colwise() + b_;
	return result;
}

RowMatrix Affine::forward(const Tensor2D& x)
{
	const auto& dims = x.dimensions();
	eidx rows = dims[0];
	eidx cols = dims[1];

	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
	}


	RowMatrix x_matrix(rows, cols);
	Eigen::TensorMap<const Tensor2D>(x_matrix.data(), rows, cols) = x;

	return forward(x_matrix);
}

RowMatrix Affine::forward(const Tensor4D& x)
{
	original_x_shape_ = x.dimensions();

	for (auto& idx : original_x_shape_)
	{
		if (idx == 0)
		{
			throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
		}
	}

	eidx N = original_x_shape_[0];
	eidx C = original_x_shape_[1];
	eidx H = original_x_shape_[2];
	eidx W = original_x_shape_[3];
	Tensor2D result_tensor =
		x.shuffle(Eigen::array<eidx, 4>{0, 2, 3, 1}).reshape(Eigen::array<eidx, 2>{N, C* H* W}).eval();

	return forward(result_tensor);
}



Tensor2D Affine::backward_2D(const RowMatrix& dout)
{
	eidx rows = dout.rows();
	eidx cols = dout.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
	}

	dw_ = x_.transpose() * dout;
	db_ = dout.rowwise().sum();

	RowMatrix dx = dout * w_.transpose();
	Tensor2D result = Eigen::TensorMap<const Tensor2D>(dx.data(), dx.rows(), dx.cols());


	return result;
}

Tensor4D Affine::backward_4D(const RowMatrix& dout)
{
	Tensor2D result = backward_2D(dout);

	eidx N = original_x_shape_[0];
	eidx C = original_x_shape_[1];
	eidx H = original_x_shape_[2];
	eidx W = original_x_shape_[3];
	Eigen::array<eidx, 4> reshape = { N, H, W, C };
	Eigen::array<eidx, 4> shuffle = { 0, 3, 1, 2 };

	return Tensor4D(result.reshape(reshape).shuffle(shuffle));
}
