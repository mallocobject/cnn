#include "utils.h"
#include "pch.h"
#include "relu.h"
#include "softmax_with_loss.h"
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

TEST(SoftmaxWithLoss, RowMatrix)
{
	SoftmaxWithLoss sl;
	RowMatrix x(2, 3);
	x << 2, 4, 6,
		0.6, 9.1, -4.2;
	LabelVector t(2);
	t << 0, 1;
	float loss = sl.forward(x, t);
	std::cout << loss << std::endl;
	RowMatrix dx = sl.backward();
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << '\t' << dx(i, j);
		}
		std::cout << std::endl;
	}
}

TEST(SoftmaxWithLoss, Tensor2D)
{
	SoftmaxWithLoss sl;
	Tensor2D x(2, 3);
	x.setValues({
		{2, 4, 6},
		{0.6, 9.1, -4.2 }
		});
	LabelVector t(2);
	t << 0, 1;
	float loss = sl.forward(x, t);
	std::cout << loss << std::endl;
	Tensor2D dx = Eigen::TensorMap<Tensor2D>(sl.backward().data(), 2, 3);
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << '\t' << dx(i, j);
		}
		std::cout << std::endl;
	}
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