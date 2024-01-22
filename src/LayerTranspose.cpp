#include "LayerTranspose.h"

beednn::LayerTranspose::LayerTranspose()
	:Layer("LayerTranspose")
{
}

beednn::LayerTranspose::~LayerTranspose()
{
}

beednn::Layer* beednn::LayerTranspose::clone() const
{
	return new LayerTranspose();
}

void beednn::LayerTranspose::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	mOut = mIn.transpose();
}

void beednn::LayerTranspose::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	mGradientIn = mGradientOut.transpose();
}
