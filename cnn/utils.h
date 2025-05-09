#pragma once
#include "pch.h"
#include <cmath>
#include <fstream>
#include <Eigen/dense>
#include <string>

namespace Utils
{
	uint32_t read_uint32(::std::ifstream& file);
	void read_images(const ::std::string& path, ImageTensor& images, ImageInfo& image_info);
	void read_labels(const ::std::string& path, LabelVector& labels);

	Tensor3D sigmoid(const Tensor3D& x);

	RowMatrix softmax(const RowMatrix& x);
	Tensor2D softmax(const Tensor2D& x);

	float crossEntropyError(const RowMatrix& y, const LabelVector& t);
	float crossEntropyError(const Tensor2D& y, const LabelVector& t);

	Tensor2D im2col(const Tensor4D& input,
		size_t filter_h, size_t filter_w, size_t stride = 1, size_t padding = 0);
	//Tensor2D im2col_(const Tensor4D& input,
	//	size_t filter_h, size_t filter_w, size_t stride = 1, size_t padding = 0);

	Tensor4D col2im(const Tensor2D& col, Tensor4D::Dimensions input_shape,
		size_t filter_h, size_t filter_w, size_t stride = 1, size_t padding = 0);
	Tensor4D col2im(const RowMatrix& col, Tensor4D::Dimensions input_shape,
		size_t filter_h, size_t filter_w, size_t stride = 1, size_t padding = 0);
}