#include "utils.h"
#include "pch.h"
#include "relu.h"
#include "softmax_with_loss.h"
#include "dropout.h"
#include "affine.h"
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

//TEST(test, one)
//{
//	EigenTensor3D x;
//	x.resize(Eigen::array<Eigen::Index, 3>{2, 2, 2});
//	x.setValues({ {{0.0f, 1.0f}, {2.0f, 3.0f}}, {{4.0f, 5.0f}, {6.0f, 7.0f}} });
//
//	EigenTensor3D result = Utils::sigmoid(x);
//
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 2; j++)
//		{
//			for (int k = 0; k < 2; k++)
//			{
//				std::cout << std::setw(4) << result(i, j, k);
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}
//
// 
//TEST(softmax, RowMatrix)
//{
//	RowMatrix x;
//	x.resize(2, 3);
//	x << 1, 2, 3,
//		4, 5, 6;
//	RowMatrix y = Utils::softmax(x);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//}
//TEST(softmax, EigenTensor2D)
//{
//	EigenTensor2D x;
//	x.resize(2, 3);
//	x.setValues({
//		{1, 2, 3 },
//		{ 4, 5, 6 }
//		});
//	EigenTensor2D y = Utils::softmax(x);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(crossEntropyError, RowMatrix)
//{
//	RowMatrix y(4, 3);
//	y << 0.3, 0.4, 0.3,
//		0.5, 0.2, 0.3,
//		0.1, 0.8, 0.1,
//		0.5, 0.5, 0;
//	LabelVector t(4);
//	t << 0, 2, 1, 1;
//	float loss = Utils::crossEntropyError(y, t);
//	std::cout << loss << std::endl;
//}
//
//TEST(crossEntropyError, EigenTensor2D)
//{
//	EigenTensor2D y(4, 3);
//	y.setValues({
//		{0.3, 0.4, 0.3},
//		{0.5, 0.2, 0.3},
//		{0.1, 0.8, 0.1},
//		{0.5, 0.5, 0}
//		});
//	LabelVector t(4);
//	t << 0, 2, 1, 1;
//	float loss = Utils::crossEntropyError(y, t);
//	std::cout << loss << std::endl;
//}

//TEST(RELU, RowMatrix)
//{
//	ReLU relu;
//	RowMatrix x(4, 3);
//	x << 0.3, 0.4, 0.3,
//		0.5, 0.2, -0.3,
//		-0.1, 0.8, 0.1,
//		0.5, -0.5, 0;
//	RowMatrix y = relu.forward(x);
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//	RowMatrix dout = RowMatrix::Ones(4, 3); // 全 1 矩阵
//	RowMatrix dx = relu.backward(dout);
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//}
//
//TEST(RELU, EigenTensor2D)
//{
//	ReLU relu;
//	EigenTensor2D x(4, 3);
//	x.setValues({
//		{0.3, 0.4, 0.3},
//		{0.5, 0.2, -0.3},
//		{-0.1, 0.8, 0.1},
//		{0.5, -0.5, 0}
//		});
//	EigenTensor2D y = relu.forward(x);
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	EigenTensor2D dout(4, 3);
//	dout.setConstant(1);
//	EigenTensor2D dx = relu.backward(dout);
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << std::setw(4) << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//}
//
//TEST(RELU, EigenTensor4D)
//{
//	ReLU relu;
//	EigenTensor4D x(1, 2, 1, 3);
//	x.setValues({
//		{{{0.1, -0.2, 0.3}},
//		{{0.5, -0.9, 0.5}}}
//		});
//	EigenTensor4D y = relu.forward(x);
//	std::cout << std::endl;
//	for (int i = 0; i < 1; i++)
//	{
//		for (int j = 0; j < 2; j++)
//		{
//			for (int k = 0; k < 1; k++)
//			{
//				for (int l = 0; l < 3; l++)
//				{
//					std::cout << std::setw(4) << y(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	EigenTensor4D dout(1, 2, 1, 3);
//	dout.setConstant(1);
//	EigenTensor4D dx = relu.backward(dout);
//	for (int i = 0; i < 1; i++)
//	{
//		for (int j = 0; j < 2; j++)
//		{
//			for (int k = 0; k < 1; k++)
//			{
//				for (int l = 0; l < 3; l++)
//				{
//					std::cout << std::setw(4) << dx(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(SoftmaxWithLoss, RowMatrix)
//{
//	SoftmaxWithLoss sl;
//	RowMatrix x(2, 3);
//	x << 2, 4, 6,
//		0.6, 9.1, -4.2;
//	LabelVector t(2);
//	t << 0, 1;
//	float loss = sl.forward(x, t);
//	std::cout << loss << std::endl;
//	RowMatrix dx = sl.backward();
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//}
//
//TEST(SoftmaxWithLoss, Tensor2D)
//{
//	SoftmaxWithLoss sl;
//	Tensor2D x(2, 3);
//	x.setValues({
//		{2, 4, 6},
//		{0.6, 9.1, -4.2 }
//		});
//	LabelVector t(2);
//	t << 0, 1;
//	float loss = sl.forward(x, t);
//	std::cout << loss << std::endl;
//	Tensor2D dx = Eigen::TensorMap<Tensor2D>(sl.backward().data(), 2, 3);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(Dropout, RowMatrix)
//{
//	Dropout dp;
//	RowMatrix x(2, 3);
//	x << 1, 2, 3,
//		4, 5, 6;
//	RowMatrix y = dp.forward(x, true);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	RowMatrix dout = RowMatrix::Ones(2, 3);
//	RowMatrix dx = dp.backward(dout);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//}
//
//TEST(Dropout, Tensor2D)
//{
//	Dropout dp;
//	Tensor2D x(2, 3);
//	x.setValues({
//		{1, 2, 3},
//		{4, 5, 6}
//		});
//	Tensor2D y = dp.forward(x, true);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << y(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	Tensor2D dout(2, 3);
//	dout.setConstant(1);
//	Tensor2D dx = dp.backward(dout);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			std::cout << '\t' << dx(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//}
//
//TEST(Dropout, Tensor4D)
//{
//	Dropout dp;
//	Tensor4D x(1, 2, 1, 3);
//	x.setValues({
//		{{{1, 2, 3}},
//		{{4, 5, 6}}}
//		});
//	Tensor4D y = dp.forward(x, true);
//	for (int i = 0; i < 1; i++)
//	{
//		for (int j = 0; j < 2; j++)
//		{
//			for (int k = 0; k < 1; k++)
//			{
//				for (int l = 0; l < 3; l++)
//				{
//					std::cout << std::setw(4) << y(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	Tensor4D dout(1, 2, 1, 3);
//	dout.setConstant(1);
//	Tensor4D dx = dp.backward(dout);
//	for (int i = 0; i < 1; i++)
//	{
//		for (int j = 0; j < 2; j++)
//		{
//			for (int k = 0; k < 1; k++)
//			{
//				for (int l = 0; l < 3; l++)
//				{
//					std::cout << std::setw(4) << dx(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(Im2col, IMG)
//{
//	/*ImageTensor images;
//	ImageInfo image_info;
//	LabelVector labels;
//	Utils::read_images(train_image_path, images, image_info);
//	Eigen::array<Eigen::Index, 4> offsets = { 0, 0, 0, 0 };
//	Eigen::array<Eigen::Index, 4> extents = { 1, 1, 28, 28 };
//	Tensor4D img = images.slice(offsets, extents).cast<float>();
//	std::cout << img.dimensions()[0] << std::endl;
//	std::cout << img.dimensions()[1] << std::endl;
//	std::cout << img.dimensions()[2] << std::endl;
//	std::cout << img.dimensions()[3] << std::endl;*/
//	Tensor4D img(1, 1, 2, 3);
//	img.setValues({ {
//		{
//			{1, 2, 3},
//		{4, 5, 6}
//}
//} });
//	const auto& shape = img.dimensions();
//
//	Tensor2D col = Utils::im2col(img, 2, 2, 1, 1);
//	size_t rows = col.dimensions()[0];
//	size_t cols = col.dimensions()[1];
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			std::cout << "  " << col(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//
//	Tensor4D y = Utils::col2im(col, shape, 2, 2, 1, 1);
//	const auto& y_dims = y.dimensions();
//	for (int i = 0; i < y_dims[0]; i++)
//	{
//		for (int j = 0; j < y_dims[1]; j++)
//		{
//			for (int k = 0; k < y_dims[2]; k++)
//			{
//				for (int l = 0; l < y_dims[3]; l++)
//				{
//					std::cout << "  " << y(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}
//
//
//TEST(Im2col, matrix)
//{
//	Tensor4D img(1, 1, 2, 3);
//	img.setValues({ {
//		{
//			{1, 2, 3},
//		{4, 5, 6}
//}
//} });
//	const auto& shape = img.dimensions();
//
//	Tensor2D col = Utils::im2col(img, 2, 2, 1, 1);
//	size_t rows = col.dimensions()[0];
//	size_t cols = col.dimensions()[1];
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			std::cout << "  " << col(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//
//	RowMatrix col_matrix = Eigen::Map<const RowMatrix>(col.data(), rows, cols);
//	Tensor4D y = Utils::col2im(col_matrix, shape, 2, 2, 1, 1);
//	const auto& y_dims = y.dimensions();
//	for (int i = 0; i < y_dims[0]; i++)
//	{
//		for (int j = 0; j < y_dims[1]; j++)
//		{
//			for (int k = 0; k < y_dims[2]; k++)
//			{
//				for (int l = 0; l < y_dims[3]; l++)
//				{
//					std::cout << "  " << y(i, j, k, l);
//				}
//				std::cout << std::endl;
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(STRIDE, None)
//{
//	Tensor2D m(3, 4);
//	m.setRandom();
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 4; j++)
//		{
//			std::cout << "  " << m(i, j);
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	Eigen::array<Eigen::Index, 2> strides = { 2, 2 };
//	Tensor2D m_strided = m.stride(strides);
//	for (int i = 0; i < m_strided.dimensions()[0]; i++)
//	{
//		for (int j = 0; j < m_strided.dimensions()[1]; j++)
//		{
//			std::cout << "  " << m_strided(i, j);
//		}
//		std::cout << std::endl;
//	}
//}

//TEST(Shuffle, Tensor)
//{
	//Tensor3D input(2, 3, 4);
	//input.setRandom();
	//std::cout << input.dimensions().size();
	/*for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 4; k++)
			{

				std::cout << "  " << input(i, j, k);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	Tensor3D output = input.shuffle(Eigen::array<Eigen::Index, 3>{1, 2, 0});
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 2; k++)
			{

				std::cout << "  " << output(i, j, k);
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}*/
	//}

TEST(Affine, RowMatrix)
{
	RowMatrix w(3, 2);
	w << 1, 2,
		4, 5,
		7, 8;
	BiasVector b(2, 1);
	b << 4, 6;
	Affine aff(w, b);
	RowMatrix x(2, 3);
	x << 1, 2, 3,
		4, 5, 6;
	RowMatrix y = aff.forward(x);
	for (int i = 0; i < y.rows(); i++)
	{
		for (int j = 0; j < y.cols(); j++)
		{
			std::cout << "  " << y(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	Tensor2D z = aff.backward_2D(y);
	for (int i = 0; i < z.dimension(0); i++)
	{
		for (int j = 0; j < z.dimension(1); j++)
		{
			std::cout << "  " << z(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

TEST(Affine, Tensor2D)
{
	RowMatrix w(3, 2);
	w << 1, 2,
		4, 5,
		7, 8;
	BiasVector b(2, 1);
	b << 4, 6;
	Affine aff(w, b);
	Tensor2D x(2, 3);
	x.setValues({
		{1, 2, 3},
		{4, 5, 6}
		});
	RowMatrix y = aff.forward(x);
	for (int i = 0; i < y.rows(); i++)
	{
		for (int j = 0; j < y.cols(); j++)
		{
			std::cout << "  " << y(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	Tensor2D z = aff.backward_2D(y);
	for (int i = 0; i < z.dimension(0); i++)
	{
		for (int j = 0; j < z.dimension(1); j++)
		{
			std::cout << "  " << z(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

TEST(Affine, Tensor4D)
{
	RowMatrix w(6, 1);
	w << 1, 2,
		4, 5,
		7, 8;
	BiasVector b(1, 1);
	b << 4;
	Affine aff(w, b);
	Tensor4D x(1, 1, 2, 3);
	x.setValues({ {
		{
			{1, 2, 3},
		{4, 5, 6}
}
} });
	RowMatrix y = aff.forward(x);
	for (int i = 0; i < y.rows(); i++)
	{
		for (int j = 0; j < y.cols(); j++)
		{
			std::cout << "  " << y(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	Tensor4D z = aff.backward_4D(y);
	for (int i = 0; i < z.dimension(0); i++)
	{
		for (int j = 0; j < z.dimension(1); j++)
		{
			for (int k = 0; k < z.dimension(2); k++)
			{
				for (int l = 0; l < z.dimension(3); l++)
				{
					std::cout << "  " << z(i, j, k, l);
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << z.dimensions() << std::endl;
}


int main()
{
	ImageTensor images;
	ImageInfo image_info;
	LabelVector labels;

	//Utils::read_images(train_image_path, images, image_info);
	//Utils::read_labels(train_label_path, labels);



	return RUN_ALL_TESTS();
}