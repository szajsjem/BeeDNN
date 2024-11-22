#include "LayerSelfDot.h"

namespace beednn {
	LayerSelfDot::LayerSelfDot(const std::string& sType) :
		Layer("sType")
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerSelfDot::LayerSelfDot() :
		Layer("LayerSelfDot")
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerSelfDot::~LayerSelfDot()
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerSelfDot::clone() const
	{
		return new LayerSelfDot();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfDot::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		//assert(mIn.rows%2==0)
		int middle = mIn.rows() / 2;
		mOut = mIn.middleRows(0, middle) * mIn.middleRows(middle, middle).transpose();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfDot::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		mGradientIn.resizeLike(mIn);
		int middle = mIn.rows() / 2;
		mGradientIn.middleRows(middle, middle) = (mIn.middleRows(0, middle).transpose() * mGradientOut).transpose() ;//might be reverse or might be completly diffrent
		mGradientIn.middleRows(0, middle) = mGradientOut * mIn.middleRows(middle, middle);
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfDot::save(std::ostream& to) const {

	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSelfDot::load(std::istream& from) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSelfDot::construct(std::initializer_list<float> fArgs, std::string sArg) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	std::string LayerSelfDot::constructUsage() {
		return "error";
	}
	///////////////////////////////////////////////////////////////
	bool LayerSelfDot::has_weights() const
	{
		return false;
	}
	///////////////////////////////////////////////////////////////////////////////
	bool LayerSelfDot::init(size_t& in, size_t& out, bool debug)
	{
		//except l2d
		out = in;
		Layer::init(in, out, debug);
		return true;
	}
}