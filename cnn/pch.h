#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/dense>
#include <variant>

using Tensor6D = Eigen::Tensor<float, 6, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageTensor = Eigen::Tensor<uint8_t, 4, Eigen::RowMajor>;
using LabelVector = Eigen::Matrix<uint8_t, -1, Eigen::RowMajor>;
using ArgVector = Eigen::Matrix<Eigen::Index, -1, Eigen::RowMajor>;
using BiasVector = Eigen::Matrix<float, -1, Eigen::RowMajor>;
using MaskMatrix = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TensorVariant = std::variant<Tensor2D, Tensor4D>;

using eidx = Eigen::Index;

struct ImageInfo
{
	uint32_t num_images = 0;
	uint32_t rows = 0;
	uint32_t cols = 0;
};

const std::string train_image_path = R"(.\data\MNIST\raw\train-images-idx3-ubyte)";
const std::string train_label_path = R"(.\data\MNIST\raw\train-labels-idx1-ubyte)";
const float epsilon = 1e-10;

#define t2m(t) (Eigen::Map<RowMatrix>(t.data(), t.dimension(0), t.dimension(1)))
#define ct2m(t) (Eigen::Map<const RowMatrix>(t.data(), t.dimension(0), t.dimension(1)))

#define m2t(m) (Eigen::TensorMap<Tensor2D>(m.data(), m.rows(), m.cols()))
#define cm2t(m) (Eigen::TensorMap<const Tensor2D>(m.data(), m.rows(), m.cols()))

#define vec(num, ...) Eigen::array<eidx, num>{__VA_ARGS__}