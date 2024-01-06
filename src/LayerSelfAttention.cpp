#include "LayerSelfAttention.h"

namespace beednn {
	LayerSelfAttention::LayerSelfAttention()
		:Layer("LayerSelfAttention") {
	}
	Layer* LayerSelfAttention::clone() const
	{
		return new LayerSelfAttention();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfAttention::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		LayerSelfDot::forward(mIn, mOut);
		for (int s = 0; s < mOut.rows(); s++)
			for (int f = s + 1; f < mOut.cols(); f++) {
				mOut(s, f) = -INFINITY;
			}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfAttention::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		MatrixFloat mf = mGradientOut;
		for (int s = 0; s < mf.rows(); s++)
			for (int f = s + 1; f < mf.cols(); f++) {
				mf(s, f) = 0;
			}
		LayerSelfDot::backpropagation(mIn, mf, mGradientIn);
	}
}