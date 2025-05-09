#include "affine.h"

#include <iostream>


Tensor2D Affine::forward(const Tensor2D& x)
{
	const auto& dims = x.dimensions();
	eidx rows = dims[0];
	eidx cols = dims[1];

	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
	}

	x_ = x;

	auto x_matrix = ct2m(x);

	RowMatrix result_matrix = (x_matrix * w_).colwise() + b_;
	//Tensor2D result(result_matrix.rows(), result_matrix.cols());
	//result = Eigen::TensorMap<Tensor2D>(result_matrix.data(), result_matrix.rows(), result_matrix.cols());
	Tensor2D result = m2t(result_matrix);
	return result;
}

Tensor2D Affine::forward(const Tensor4D& x)
{
	original_x_shape_ = x.dimensions();
	input_is_4D = true;

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
		x.shuffle(vec(4, 0, 2, 3, 1)).reshape(vec(2, N, C * H * W));

	return forward(result_tensor);
}



TensorVariant Affine::backward(const Tensor2D& dout)
{
	const auto& dims = dout.dimensions();
	eidx rows = dims[0];
	eidx cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Affine ÊäÈë¾ØÕóÎª¿Õ");
	}
	Eigen::Map<const RowMatrix> dout_matrix(dout.data(), rows, cols);

	auto x_matrix = t2m(x_);
	dw_ = x_matrix.transpose() * dout_matrix;
	db_ = dout_matrix.rowwise().sum();

	RowMatrix dx = dout_matrix * w_.transpose();
	/*Tensor2D result = Eigen::TensorMap<Tensor2D>(dx.data(), dx.rows(), dx.cols());*/
	Tensor2D result = m2t(dx);

	if (input_is_4D)
	{
		eidx N = original_x_shape_[0];
		eidx C = original_x_shape_[1];
		eidx H = original_x_shape_[2];
		eidx W = original_x_shape_[3];
		Eigen::array<eidx, 4> reshape = { N, H, W, C };
		Eigen::array<eidx, 4> shuffle = { 0, 3, 1, 2 };

		return Tensor4D(result.reshape(reshape).shuffle(shuffle));
	}

	return result;
}

