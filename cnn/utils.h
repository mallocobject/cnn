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

	/*RowMatrix im2col(const EigenTensor4D& input, const ConvParams& params);
	RowMatrix col2im(const EigenTensor4D& conv_output);*/
}