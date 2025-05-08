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
	//std::vector<unsigned char> buffer(image_size);
	Eigen::Tensor<uint8_t, 2, Eigen::RowMajor> buffer(1, image_size);
	for (uint32_t i = 0; i < num_images; ++i)
	{
		// 读取一张图像
		file.read(reinterpret_cast<char*>(buffer.data()), image_size);
		/*for (uint32_t r = 0; r < rows; ++r)
		{
			for (uint32_t c = 0; c < cols; ++c)
			{
				images(i, 0, r, c) = buffer[r * cols + c];
			}
		}*/
		/*Eigen::TensorMap<Eigen::Tensor<unsigned char, 2>> buffer_tensor(
			buffer.data(), rows, cols);*/
		Eigen::array<Eigen::Index, 4> offsets = { i, 0, 0, 0 };
		Eigen::array<Eigen::Index, 4> extents = { 1, 1, rows, cols };
		Eigen::array<Eigen::Index, 2> slice_shape = { 1, image_size };
		images.slice(offsets, extents).reshape(slice_shape) = buffer;
		//cv::Mat image_tmp(rows, cols, CV_8UC1, buffer.data());
		//// resize bigger for showing
		//cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
		//cv::imshow("picture", image_tmp);
		//cv::waitKey(0);
	}

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
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("输入张量为空，无法应用 softmax");
	}

	RowMatrix y_matrix = Eigen::Map<const RowMatrix>(y.data(), rows, cols);
	float loss = crossEntropyError(y_matrix, t);

	return loss;
}

Tensor2D Utils::im2col(const Tensor4D& input, size_t filter_h, size_t filter_w, size_t stride, size_t padding)
{
	const auto& dims = input.dimensions();
	eidx N = dims[0];
	eidx C = dims[1];
	eidx H = dims[2];
	eidx W = dims[3];
	if ((N * C * H * W) == 0)
	{
		throw std::invalid_argument("输入张量为空，无法应用 im2col");
	}
	if (filter_h > H || filter_w > W)
	{
		throw std::invalid_argument("卷积核大于图像，无法应用 im2col");
	}

	eidx filter_h_idx = static_cast<eidx>(filter_h);
	eidx filter_w_idx = static_cast<eidx>(filter_w);
	eidx stride_idx = static_cast<eidx>(stride);
	eidx padding_idx = static_cast<eidx>(padding);

	eidx output_h_idx = (H + 2 * padding_idx - filter_h_idx) / stride_idx + 1;
	eidx output_w_idx = (W + 2 * padding_idx - filter_w_idx) / stride_idx + 1;

	Tensor4D img(N, C, H + 2 * padding_idx, W + 2 * padding_idx);
	img.setZero();

	// 复制输入到填充张量的中心
	Eigen::array<Eigen::Index, 4> offsets = { 0, 0, padding_idx, padding_idx };
	Eigen::array<Eigen::Index, 4> extents = { N, C, H, W };
	img.slice(offsets, extents) = input;

	Tensor6D col(N, C, filter_h_idx, filter_w_idx, output_h_idx, output_w_idx);
	col.setZero();

	// 提取图像块
	for (eidx y = 0; y < filter_h_idx; ++y)
	{
		for (eidx x = 0; x < filter_w_idx; ++x)
		{
			Tensor4D slice(N, C, output_h_idx, output_w_idx);
			slice.setZero();

			for (eidx oh = 0; oh < output_h_idx; ++oh)
			{
				for (eidx ow = 0; ow < output_w_idx; ++ow)
				{
					eidx h_idx = y + oh * stride_idx;
					eidx w_idx = x + ow * stride_idx;

					// 验证索引范围
					if (h_idx >= img.dimension(2) || w_idx >= img.dimension(3))
					{
						throw std::out_of_range("slice 索引超出 img 维度: h_idx=" + std::to_string(h_idx) +
							", w_idx=" + std::to_string(w_idx));
					}

					Eigen::array<eidx, 4> img_offsets = { 0, 0, h_idx, w_idx };
					Eigen::array<eidx, 4> img_extents = { N, C, 1, 1 };
					Tensor4D pixel = img.slice(img_offsets, img_extents);

					Eigen::array<eidx, 4> slice_offsets = { 0, 0, oh, ow };
					Eigen::array<eidx, 4> slice_extents = { N, C, 1, 1 };
					slice.slice(slice_offsets, slice_extents) = pixel;
				}
			}

			Eigen::array<eidx, 6> col_offsets = { 0, 0, y, x, 0, 0 };
			Eigen::array<eidx, 6> col_extents = { N, C, 1, 1, output_h_idx, output_w_idx };
			col.slice(col_offsets, col_extents) = slice.reshape(
				Eigen::array<eidx, 6>{N, C, 1, 1, output_h_idx, output_w_idx});
		}
	}

	// 维度调整
	Eigen::array<eidx, 6> col_shuffle = { 0, 4, 5, 1, 2, 3 };

	// 重塑为 2D
	Eigen::array<eidx, 2> col_reshape = { N * output_h_idx * output_w_idx, filter_h_idx * filter_w_idx * C };
	Tensor2D result = col.shuffle(col_shuffle).reshape(col_reshape);

	return result;
}

Tensor4D Utils::col2im(const Tensor2D& col, Tensor4D::Dimensions input_shape, size_t filter_h, size_t filter_w, size_t stride, size_t padding)
{
	eidx N = input_shape[0];
	eidx C = input_shape[1];
	eidx H = input_shape[2];
	eidx W = input_shape[3];

	// 验证输入
	if ((N * C * H * W) == 0)
	{
		throw std::invalid_argument("输入形状无效，无法应用 col2im");
	}

	// 转换为 Eigen::Index
	eidx filter_h_idx = static_cast<eidx>(filter_h);
	eidx filter_w_idx = static_cast<eidx>(filter_w);
	eidx stride_idx = static_cast<eidx>(stride);
	eidx padding_idx = static_cast<eidx>(padding);

	// 计算输出维度
	eidx out_h = (H + 2 * padding_idx - filter_h_idx) / stride_idx + 1;
	eidx out_w = (W + 2 * padding_idx - filter_w_idx) / stride_idx + 1;
	if (out_h <= 0 || out_w <= 0)
	{
		throw std::invalid_argument("无效的滤波器尺寸或步幅");
	}

	// 验证 col 形状
	if (col.dimension(0) != N * out_h * out_w ||
		col.dimension(1) != filter_h_idx * filter_w_idx * C)
	{
		throw std::invalid_argument("col 张量形状与预期不符");
	}

	// 重塑和重排 col
	Tensor6D col_reshaped = col.reshape(
		Eigen::array<eidx, 6>{N, out_h, out_w, C, filter_h_idx, filter_w_idx});
	Eigen::array<eidx, 6> permute = { 0, 3, 4, 5, 1, 2 };
	Tensor6D col_permuted = col_reshaped.shuffle(permute); // 形状 (N, C, filter_h, filter_w, out_h, out_w)

	// 创建填充图像张量
	eidx H_padded = H + 2 * padding_idx + stride_idx - 1;
	eidx W_padded = W + 2 * padding_idx + stride_idx - 1;
	Tensor4D img(N, C, H_padded, W_padded);
	img.setZero();

	// 累加图像块
	for (eidx y = 0; y < filter_h_idx; ++y)
	{
		for (eidx x = 0; x < filter_w_idx; ++x)
		{
			for (eidx oh = 0; oh < out_h; ++oh)
			{
				for (eidx ow = 0; ow < out_w; ++ow)
				{
					eidx h_idx = y + oh * stride_idx;
					eidx w_idx = x + ow * stride_idx;

					// 验证索引范围
					if (h_idx >= H_padded || w_idx >= W_padded)
					{
						throw std::out_of_range("索引超出 img 维度: h_idx=" + std::to_string(h_idx) +
							", w_idx=" + std::to_string(w_idx));
					}

					// 提取 col[:, :, y, x, oh, ow]
					Eigen::array<eidx, 6> col_offsets = { 0, 0, y, x, oh, ow };
					Eigen::array<eidx, 6> col_extents = { N, C, 1, 1, 1, 1 };
					Tensor4D col_pixel = col_permuted.slice(col_offsets, col_extents)
						.reshape(Eigen::array<eidx, 4>{N, C, 1, 1});

					// 提取 img[:, :, h_idx, w_idx]
					Eigen::array<eidx, 4> img_offsets = { 0, 0, h_idx, w_idx };
					Eigen::array<eidx, 4> img_extents = { N, C, 1, 1 };
					Tensor4D img_pixel = img.slice(img_offsets, img_extents);

					// 累加
					img_pixel += col_pixel;

					// 写回 img
					img.slice(img_offsets, img_extents) = img_pixel;
				}
			}
		}
	}

	// 去除填充
	Eigen::array<eidx, 4> final_offsets = { 0, 0, padding_idx, padding_idx };
	Eigen::array<eidx, 4> final_extents = { N, C, H, W };
	Tensor4D result = img.slice(final_offsets, final_extents);

	return result;
}

Tensor4D Utils::col2im(const RowMatrix& col, Tensor4D::Dimensions input_shape, size_t filter_h, size_t filter_w, size_t stride, size_t padding)
{
	eidx rows = col.rows();
	eidx cols = col.cols();
	if (rows == 0 || cols == 0)
	{
		throw std::invalid_argument("col2im 输入矩阵为空");
	}
	Tensor2D col_tensor = Eigen::TensorMap<const Tensor2D>(col.data(), rows, cols);
	return col2im(col_tensor, input_shape, filter_h, filter_w, stride, padding);
}
