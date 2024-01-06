#include "LayerSelfDot.h"

namespace beednn {
	LayerSelfDot::LayerSelfDot() :
		Layer("LayerSelfDot")
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
		mOut = mIn.middleRows(0, middle) * mIn.middleRows(middle, middle);
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSelfDot::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		int middle = mIn.rows() / 2;
		mGradientIn.middleRows(0, middle) = mGradientOut * mIn.middleRows(middle, middle);//might be reverse
		mGradientIn.middleRows(middle, middle) = mGradientOut * mIn.middleRows(0, middle);
	}
}