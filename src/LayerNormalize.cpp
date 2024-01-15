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
		for (int c = 0; c < mIn.cols(); c++) {
			double avg = 0, stdev = 0;;
			for (int i = 0; i < mIn.rows(); i++) {
				avg += mIn(i, c);
			}
			avg /= mIn.rows();
			for (int i = 0; i < mIn.rows(); i++) {
				mOut(i, c) = mIn(i, c) - avg;
				stdev += pow(mOut(i, c), 2);
			}
			stdev = sqrt(stdev / (mIn.rows() - 1));
			for (int i = 0; i < mIn.rows(); i++) {
				mOut(i, c) /= stdev;
			}
		}
		/*
		this function needs to produce:
		average(mOut.col(i))==0
		standard_deviation(mOut.col(i))==1

		mOut = mIn.colwise().normalized();
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