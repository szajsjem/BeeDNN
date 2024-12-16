#include "LayerParallel.h"

namespace beednn {
	LayerParallel::LayerParallel(std::vector<Layer*> mParallelLayers, ParallelReduction mReduction)
		:Layer("LayerParallel")
	{
		for (auto layer : mParallelLayers) {
			_Layers.push_back(layer);
		}
		_ParallelReduction = mReduction;
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerParallel::LayerParallel() :
		Layer("LayerParallel")
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerParallel::~LayerParallel()
	{
		for (auto x : _Layers) {
			delete x;
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerParallel::clone() const
	{
		LayerParallel* pLayer = new LayerParallel();
		for (auto x : _Layers) {
			pLayer->_Layers.push_back(x->clone());
		}
		pLayer->_ParallelReduction = _ParallelReduction;
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	bool LayerParallel::init(size_t& in, size_t& out, bool debug)
	{
		if (_ParallelReduction != COLSTACK) {
			for (auto l : _Layers) {
				if (!l->init(in, out, debug))
					return false;
			}
		}
		else {
			size_t clout = 0;
			for (auto l : _Layers) {
				size_t tout = -1;
				if (!l->init(in, tout, debug))
					return false;
				if (tout == -1 || clout == -1)
					clout = -1;
				else
					clout += tout;
			}
			if (out == -1)
				out = clout;
			if (out != clout)
				return false;
		}
		Layer::init(in, out, debug);
		return true;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		mOut.resize(0, 0);//todo save output sizes of diffrent layers
		for (auto x : _Layers) {
			MatrixFloat mf;
			x->forward(mIn, mf);
			if (mOut.size() == 0) {
				mOut = mf;
			}
			else if (_ParallelReduction == SUM) {
				mOut += mf;
			}
			else if (_ParallelReduction == DOT) {
				mOut *= mf;
			}
			else if (_ParallelReduction == ROWSTACK) {
				mOut = concatenateRows(mOut, mf);
			}
			else if (_ParallelReduction == COLSTACK) {
				mOut = concatenateCols(mOut, mf);
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		mGradientIn.resize(0, 0);//todo save output sizes of diffrent layers
		if (_ParallelReduction == ROWSTACK) {
			Index start = 0, n = mGradientOut.rows();
			for (auto x : _Layers) {
				MatrixFloat mf;
				MatrixFloat temp;
				x->forward(mIn, temp);
				n = temp.rows();

				//assert(start+n<=mGradientOut.rows())
				x->backpropagation(mIn, mGradientOut.middleRows(start, n), mf);
				if (mGradientIn.size() == 0) {
					mGradientIn = mf;
				}
				else {
					mGradientIn += mf;
				}

				start += n;
			}
		}
		if (_ParallelReduction == COLSTACK) {
			Index start = 0, n = mGradientOut.cols();
			for (auto x : _Layers) {
				MatrixFloat mf;
				MatrixFloat temp;
				x->forward(mIn, temp);
				n = temp.cols();

				//assert(start+n<=mGradientOut.rows())
				x->backpropagation(mIn, mGradientOut.middleCols(start, n), mf);
				if (mGradientIn.size() == 0) {
					mGradientIn = mf;
				}
				else {
					mGradientIn += mf;
				}

				start += n;
			}
		}
		if (_ParallelReduction == SUM) {
			for (auto x : _Layers) {
				MatrixFloat mf;
				x->backpropagation(mIn, mGradientOut, mf);
				if (mGradientIn.size() == 0) {
					mGradientIn = mf;
				}
				else {
					mGradientIn += mf;
				}
			}
		}
		if (_ParallelReduction == DOT) {
			std::vector<MatrixFloat> mOuts(_Layers.size());
			for (int i = 0; i < _Layers.size(); i++)
				_Layers[i]->forward(mIn, mOuts[i]);

			MatrixFloat mG = mGradientOut;
			for (int i = _Layers.size() - 1; i >= 0; i--) {
				MatrixFloat mf, mD;
				if (i > 0) {
					mD = mOuts[0];
					for (int j = 1; j < i; j++) {
						mD *= mOuts[j];
					}
					_Layers[i]->backpropagation(mIn, mD.transpose() * mG, mf);//todo check mD*mG
					mG *= mOuts[i].transpose();
				}
				else {
					_Layers[i]->backpropagation(mIn, mG, mf);//todo check mD*mG
				}
				//mG *= mOuts[i].inverse();//inverse symbol not found
				if (mGradientIn.size() == 0) {
					mGradientIn = mf;
				}
				else {
					mGradientIn += mf;
				}
			}
		}
		mGradientIn /= _Layers.size();
	}
	///////////////////////////////////////////////////////////////
	bool LayerParallel::has_weights() const
	{
		for (auto layer : _Layers)
			if (layer->has_weights())
				return true;
		return false;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerParallel::weights() const
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_weights()) {
				auto vi = layer->weights();
				if (vi.size() > 0) {//this probably not needed
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		return v;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerParallel::gradient_weights() const
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_weights()) {
				auto vi = layer->gradient_weights();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		return v;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::save(std::ostream& to) const {

	}
	///////////////////////////////////////////////////////////////
	Layer* LayerParallel::load(std::istream& from) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	Layer* LayerParallel::construct(std::initializer_list<float> fArgs, std::string sArg) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	std::string LayerParallel::constructUsage() {
		return "parallel layer combination\nsReduction;layers\niNumLayers";
	}
	///////////////////////////////////////////////////////////////
}