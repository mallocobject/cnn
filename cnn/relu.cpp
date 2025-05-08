#include "relu.h"

RowMatrix ReLU::forward(const RowMatrix& x)
{
	Eigen::Index rows = x.rows();
	Eigen::Index cols = x.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("ReLU 输入矩阵为空");
	}

	mask_.resize(rows, cols);
	mask_ = (x.array() <= 0).cast<bool>().matrix();
	RowMatrix result = x.array().max(0.0f);

	return result;
}

Tensor2D ReLU::forward(const Tensor2D& x)
{
	const auto& dims = x.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("ReLU 输入矩阵为空");
	}
	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);
	RowMatrix result_matrix = forward(x_matrix);  // 转给RowMatrix版本函数处理
	Tensor2D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

Tensor4D ReLU::forward(const Tensor4D& x)
{
	const auto& dims = x.dimensions();
	if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0)
	{
		throw std::invalid_argument("ReLU 输入 4D 张量为空");
	}

	// 保存掩码（展平为 2D）
	Eigen::Index rows = dims[0] * dims[1];
	Eigen::Index cols = dims[2] * dims[3];

	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);
	RowMatrix result_matrix = forward(x_matrix);  // 转给RowMatrix版本函数处理
	Tensor4D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

RowMatrix ReLU::backward(const RowMatrix& dout)
{
	Eigen::Index rows = dout.rows();
	Eigen::Index cols = dout.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("ReLU 输入矩阵为空");
	}

	RowMatrix result = dout - (dout.array() * mask_.array().cast<float>()).matrix();
	return result;
}

Tensor2D ReLU::backward(const Tensor2D& dout)
{
	const auto& dims = dout.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("ReLU 输入矩阵为空");
	}
	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(dout.data(), rows, cols);
	RowMatrix result_matrix = backward(x_matrix);
	Tensor2D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

Tensor4D ReLU::backward(const Tensor4D& dout)
{
	const auto& dims = dout.dimensions();
	if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0)
	{
		throw std::invalid_argument("ReLU 输入 4D 张量为空");
	}

	// 保存掩码（展平为 2D）
	Eigen::Index rows = dims[0] * dims[1];
	Eigen::Index cols = dims[2] * dims[3];

	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(dout.data(), rows, cols);
	RowMatrix result_matrix = backward(x_matrix);
	Tensor4D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}
