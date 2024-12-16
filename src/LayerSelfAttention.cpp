#include "LayerSelfAttention.h"

namespace beednn {
	LayerSelfAttention::LayerSelfAttention()
		:LayerSelfDot("LayerSelfAttention") {
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
		LayerSelfDot::backpropagation(mIn, mf, mGradientIn);//this might be before zeroing
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfAttention::save(std::ostream& to) const {

	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSelfAttention::load(std::istream& from) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSelfAttention::construct(std::initializer_list<float> fArgs, std::string sArg) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	std::string LayerSelfAttention::constructUsage() {
		return "self attention mechanism\n \n ";
	}
	///////////////////////////////////////////////////////////////
	bool LayerSelfAttention::has_weights() const
	{
		return false;
	}
	///////////////////////////////////////////////////////////////////////////////
	bool LayerSelfAttention::init(size_t& in, size_t& out, bool debug)
	{
		//except l2d
		out = in;
		Layer::init(in, out, debug);
		return true;
	}
}