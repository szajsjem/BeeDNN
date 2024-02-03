#include "LayerNormalize.h"

using namespace std;
namespace beednn {
	///////////////////////////////////////////////////////////////////////////////
	LayerNormalize::LayerNormalize() :
		Layer("Normalize")
	{
		LayerNormalize::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerNormalize::~LayerNormalize()
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerNormalize::clone() const
	{
		LayerNormalize* pLayer = new LayerNormalize();
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::init()
	{
		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		mOut.resizeLike(mIn);
		for (int c = 0; c < mIn.rows(); c++) {
			double avg = mIn.row(c).mean(), stdev = 0;
			for (int i = 0; i < mIn.cols(); i++) {
				mOut(c, i) = mIn(c, i) - avg;
				stdev += pow(mOut(c, i), 2);
			}
			stdev = sqrt((stdev+1e-6) / (mIn.cols() - 1));
			for (int i = 0; i < mIn.cols(); i++) {
				mOut(c, i) /= stdev;
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		mGradientIn.resizeLike(mIn);
		//mGradientIn = mGradientOut;
		//return;

		for (int c = 0; c < mIn.rows(); c++) {
			double avg = mIn.row(c).mean(), stdev = 0;

			for (int i = 0; i < mIn.cols(); i++) {
				stdev += pow(mIn(c, i) - avg, 2);
			}
			stdev = sqrt((stdev + 1e-6) / (mIn.cols() - 1));

			// Backpropagation
			for (int i = 0; i < mIn.cols(); i++) {
				mGradientIn(c, i) = (mGradientOut(c, i)) / stdev
					- (1.0 / mIn.cols()) * ((mGradientOut.row(c).sum()) / stdev
						- (mGradientOut(c, i)  * (mIn(c, i) - avg)) / pow(stdev, 2));
			}
		}
	}
	///////////////////////////////////////////////////////////////
}