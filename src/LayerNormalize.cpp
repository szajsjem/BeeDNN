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
			double avg = 0, stdev = 0;;
			for (int i = 0; i < mIn.cols(); i++) {
				avg += mIn(c,i);
			}
			avg /= mIn.cols();
			for (int i = 0; i < mIn.cols(); i++) {
				mOut(c, i) = mIn(c, i) - avg;
				stdev += pow(mOut(c, i), 2);
			}
			stdev = sqrt((stdev+1e-6) / (mIn.cols() - 1));
			for (int i = 0; i < mIn.cols(); i++) {
				mOut(c, i) /= stdev;
			}
		}
		/*
		this function needs to produce:
		average(mOut.row(i))==0
		standard_deviation(mOut.row(i))==1

		mOut = mIn.rowwise().normalized();
		this does not
		*/
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		mGradientIn = mGradientOut;
		//we should probably need to do somthing
	}
	///////////////////////////////////////////////////////////////
}