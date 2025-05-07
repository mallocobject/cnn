#include "utils.h"
#include <opencv2/opencv.hpp>

uint32_t Utils::read_uint32(std::ifstream& file)
{
	uint32_t num;
	file.read(reinterpret_cast<char*>(&num), sizeof(num));
	num = ((num << 8) & 0xFF00FF00) | ((num >> 8) & 0xFF00FF);
	return (num << 16) | (num >> 16);
}

void Utils::read_images(const std::string& path, ImageTensor& images, ImageInfo& image_info)
{
	// 打开文件
	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		throw std::runtime_error("无法打开文件");
	}

	// 读取文件头部
	uint32_t magic_number = read_uint32(file);
	uint32_t num_images = read_uint32(file);
	image_info.num_images = num_images;
	uint32_t rows = read_uint32(file);
	image_info.rows = rows;
	uint32_t cols = read_uint32(file);
	image_info.cols = cols;

	// 打印信息
	std::cout << "图像数量: " << num_images << std::endl;
	std::cout << "图像尺寸: " << rows << " x " << cols << std::endl;

	// 验证魔数
	if (magic_number != 2051)
	{
		throw std::runtime_error("无效的图像文件格式");
	}

	// 设置张量大小
	images.resize(Eigen::array<Eigen::Index, 4>{num_images, 1, rows, cols});

	// 读取所有图像数据
	const size_t image_size = rows * cols;
	char* buffer = new char[image_size];
	for (uint32_t i = 0; i < num_images; ++i)
	{
		// 读取一张图像
		file.read(reinterpret_cast<char*>(buffer), image_size);
		for (uint32_t r = 0; r < rows; ++r)
		{
			for (uint32_t c = 0; c < cols; ++c)
			{
				images(i, 0, r, c) = buffer[r * cols + c];
			}
		}
	}

	delete[] buffer;
	file.close();
}

void Utils::read_labels(const std::string& path, LabelVector& labels)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		throw std::runtime_error("无法打开文件");
	}

	// 读取文件头
	int32_t magic_number = read_uint32(file);
	int32_t num_labels = read_uint32(file);

	// 验证魔数
	if (magic_number != 2049)
	{
		throw std::runtime_error("无效的标签文件格式");
	}

	// 读取标签数据
	labels.resize(num_labels, 1);
	file.read(reinterpret_cast<char*>(labels.data()), num_labels);

	file.close();
}

Tensor3D Utils::sigmoid(const Tensor3D& x)
{
	const auto& dims = x.dimensions();
	Tensor3D result;
	result.resize(dims);

	// 向量化操作（高效）
	result = x.unaryExpr([](float val) -> float
		{
			return 1.0f / (1.0f + std::exp(-val));
		});

	return result;
}

RowMatrix Utils::softmax(const RowMatrix& x)
{
	Eigen::Index rows = x.rows();
	Eigen::Index cols = x.cols();
	// 验证输入
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("输入矩阵为空，无法应用 softmax");
	}
	if (!x.allFinite())
	{
		throw std::invalid_argument("输入包含非有限值（NaN 或 Inf）");
	}

	RowMatrix exp_x(rows, cols);
	exp_x = (x.colwise() - x.rowwise().maxCoeff()).array().exp().matrix();
	RowMatrix exp_sum(rows, 1);
	exp_sum = exp_x.rowwise().sum();

	RowMatrix result(rows, cols);
	result = exp_x.array() / exp_sum.replicate(1, cols).array(); // [rows, cols]

	return result;
}

Tensor2D Utils::softmax(const Tensor2D& x)
{
	const auto& dims = x.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	// 验证输入
	if (dims[0] == 0 || dims[1] == 0)
	{
		throw std::invalid_argument("输入张量为空，无法应用 softmax");
	}
	// 映射为 RowMatrix
	RowMatrix x_matrix = Eigen::Map<const RowMatrix>(x.data(), rows, cols);

	// 调用 RowMatrix 版本的 softmax
	RowMatrix result_matrix = softmax(x_matrix);

	// 转换回 Tensor2D
	Tensor2D result;
	result.resize(dims);
	Eigen::Map<RowMatrix>(result.data(), rows, cols) = result_matrix;

	return result;
}

float Utils::crossEntropyError(const RowMatrix& y, const LabelVector& t)
{
	size_t rows = y.rows();
	size_t cols = y.cols();
	// 验证输入
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("输入矩阵为空，无法应用 softmax");
	}
	if (!y.allFinite())
	{
		throw std::invalid_argument("输入包含非有限值（NaN 或 Inf）");
	}
	size_t batch_size = rows;
	float loss = 0;
	for (Eigen::Index i = 0; i < rows; ++i)
	{
		float prob = std::max(y(i, t(i)), epsilon);
		loss -= std::log(prob);
	}
	loss /= static_cast<float>(batch_size);
	return loss;
}

float Utils::crossEntropyError(const Tensor2D& y, const LabelVector& t)
{
	const auto& dims = y.dimensions();
	Eigen::Index rows = dims[0];
	Eigen::Index cols = dims[1];
	// 验证输入
	if (dims[0] == 0 || dims[1] == 0)
	{
		throw std::invalid_argument("输入张量为空，无法应用 softmax");
	}

	RowMatrix y_matrix = Eigen::Map<const RowMatrix>(y.data(), rows, cols);
	float loss = crossEntropyError(y_matrix, t);

	return loss;
}
