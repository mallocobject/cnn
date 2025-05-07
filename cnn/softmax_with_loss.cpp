#include "softmax_with_loss.h"
#include "utils.h"
#include <Eigen/dense>

float SoftmaxWithLoss::forward(const RowMatrix& x, const LabelVector& t)
{
	Eigen::Index rows = x.rows();
	Eigen::Index cols = x.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("SoftmaxWithLoss  ‰»Îæÿ’ÛŒ™ø’");
	}
	if (t_ != nullptr)
	{
		delete t_;
	}
	if (y_ != nullptr)
	{
		delete y_;
	}
	t_ = new LabelVector(rows);
	*t_ = t;
	y_ = new RowMatrix(rows, cols);
	*y_ = Utils::softmax(x);
	loss_ = Utils::crossEntropyError(*y_, t);

	return loss_;
}

float SoftmaxWithLoss::forward(const Tensor2D& x, const LabelVector& t)
{
	const auto& dims = x.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("SoftmaxWithLoss  ‰»Îæÿ’ÛŒ™ø’");
	}
	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);
	loss_ = forward(x_matrix, t);

	return loss_;
}


RowMatrix SoftmaxWithLoss::backward(const float dout)
{
	Eigen::Index rows = t_->rows();;
	size_t batch_size = rows;
	RowMatrix result = *y_;
	for (int i = 0; i < rows; ++i)
	{
		result(i, (*t_)(i)) -= 1;
	}
	return 1.0f * result / batch_size;
}
