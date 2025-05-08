#include "dropout.h"
#include <chrono> // ���ڻ�ȡʱ������

Dropout::Dropout(float dropout_ratio)
	: dropout_ratio_(dropout_ratio)
{
	if (dropout_ratio < 0.0f || dropout_ratio > 1.0f)
	{
		throw std::invalid_argument("Dropout ���ʱ����� [0, 1] ��Χ��");
	}
}


RowMatrix Dropout::forward(const RowMatrix& x, bool is_train)
{
	Eigen::Index rows = x.rows();
	Eigen::Index cols = x.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Dropout �������Ϊ��");
	}
	if (!x.allFinite())
	{
		throw std::invalid_argument("�������������ֵ��NaN �� Inf��");
	}

	if (is_train)
	{
		mask_.resize(rows, cols);
		auto seed = static_cast<unsigned int>(
			std::chrono::high_resolution_clock::now().time_since_epoch().count());
		std::mt19937 gen(static_cast<unsigned int>(seed));
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		RowMatrix rand = RowMatrix::Zero(rows, cols).unaryExpr([&](float value)
			{
				return dist(gen);
			});
		mask_ = (rand.array() > dropout_ratio_).cast<bool>().matrix();

		RowMatrix result = (x.array() * mask_.array().cast<float>()).matrix();

		return result;
	}
	return x * (1.0 - dropout_ratio_);
}

Tensor2D Dropout::forward(const Tensor2D& x, bool is_train)
{
	const auto& dims = x.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Dropout �������Ϊ��");
	}

	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);
	RowMatrix result_matrix = forward(x_matrix, is_train);  // ת��RowMatrix�汾��������
	Tensor2D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

Tensor4D Dropout::forward(const Tensor4D& x, bool is_train)
{
	const auto& dims = x.dimensions();
	if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0)
	{
		throw std::invalid_argument("ReLU ���� 4D ����Ϊ��");
	}

	// �������루չƽΪ 2D��
	Eigen::Index rows = dims[0] * dims[1];
	Eigen::Index cols = dims[2] * dims[3];

	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);
	RowMatrix result_matrix = forward(x_matrix, is_train);  // ת��RowMatrix�汾��������
	Tensor4D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

RowMatrix Dropout::backward(const RowMatrix& dout)
{
	Eigen::Index rows = dout.rows();
	Eigen::Index cols = dout.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Dropout �������Ϊ��");
	}

	RowMatrix result = (dout.array() * mask_.array().cast<float>()).matrix();
	return result;
}

Tensor2D Dropout::backward(const Tensor2D& dout)
{
	const auto& dims = dout.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("Dropout �������Ϊ��");
	}

	RowMatrix dout_matrix = Eigen::Map<const RowMatrix>(dout.data(), rows, cols);
	RowMatrix result_matrix = backward(dout_matrix);  // ת��RowMatrix�汾��������
	Tensor2D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

Tensor4D Dropout::backward(const Tensor4D& dout)
{
	const auto& dims = dout.dimensions();
	if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0 || dims[3] == 0)
	{
		throw std::invalid_argument("ReLU ���� 4D ����Ϊ��");
	}

	// �������루չƽΪ 2D��
	Eigen::Index rows = dims[0] * dims[1];
	Eigen::Index cols = dims[2] * dims[3];

	RowMatrix dout_matrix = Eigen::Map<const RowMatrix>(dout.data(), rows, cols);
	RowMatrix result_matrix = backward(dout_matrix);  // ת��RowMatrix�汾��������
	Tensor4D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}
