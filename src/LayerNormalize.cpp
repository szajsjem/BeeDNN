#include "LayerNormalize.h"

using namespace std;
namespace beednn {
	///////////////////////////////////////////////////////////////////////////////
	LayerNormalize::LayerNormalize() :
		Layer("Normalize")
	{
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
	bool LayerNormalize::init(size_t& in, size_t& out, bool debug)
	{
		out = in;
		Layer::init(in, out, debug);
		return true;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		assert(mIn.cols() >= 2);
		mOut.resizeLike(mIn);
		for (int c = 0; c < mIn.rows(); c++) {
			double avg = mIn.row(c).mean(), stdev = 0;
			for (int i = 0; i < mIn.cols(); i++) {
				mOut(c, i) = mIn(c, i) - avg;
				stdev += mOut(c, i)* mOut(c, i);
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
	///////////////////////////////////////////////////////////////////////////////
	void LayerNormalize::save(std::ostream& to) const {

	}
	///////////////////////////////////////////////////////////////
	Layer* LayerNormalize::load(std::istream& from) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	Layer* LayerNormalize::construct(std::initializer_list<float> fArgs, std::string sArg) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	std::string LayerNormalize::constructUsage() {
		return "error";
	}
	///////////////////////////////////////////////////////////////
	bool LayerNormalize::has_weights() const
	{
		return false;
	}
	///////////////////////////////////////////////////////////////
}