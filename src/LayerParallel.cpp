#include "LayerParallel.h"

namespace beednn {
	LayerParallel::LayerParallel(std::vector<Layer*> mParallelLayers, ParallelReduction mReduction)
		:Layer("TransformerHeads")
	{
		for (auto layer : mParallelLayers) {
			_Layers.push_back(layer);
		}
		_ParallelReduction = mReduction;
		LayerParallel::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerParallel::LayerParallel() :
		Layer("TransformerHeads")
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
		pLayer->init();
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::init()
	{
		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		for (auto x : _Layers) {
			MatrixFloat mf;
			x->forward(mIn, mf);
			if (mOut.size() == 0) {
				mOut = mf;
			}
			else if (_ParallelReduction == SUM) {
				mOut += mf;
			}
			else if (_ParallelReduction == None) {
				for (int i = 0; i < mf.rows(); i++)
					mOut.row(mOut.rows()) = mf.row(i);
			}
			else if (_ParallelReduction == Dot) {
				mOut *= mf;
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerParallel::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		if (_ParallelReduction == None) {
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

				mGradientIn += mf;

				start += n;
			}
		}
		else if (_ParallelReduction == SUM) {
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
		else if (_ParallelReduction == Dot) {
			std::vector<MatrixFloat> mOuts(_Layers.size());
			for (int i = 0; i < _Layers.size(); i++)
				_Layers[i]->forward(mIn, mOuts[i]);

			MatrixFloat mG = mGradientOut;
			for (int i = _Layers.size() - 1; i >= 0; i--) {
				MatrixFloat mf, mD(1);
				for (int j = 0; j < i; j++) {
					mD *= mOuts[j];
				}
				_Layers[i]->backpropagation(mIn, mG * mD, mf);//todo check mD*mG

				mG *= mOuts[i].inverse();
				if (mGradientIn.size() == 0) {
					mGradientIn = mf;
				}
				else {
					mGradientIn += mf;
				}
			}
		}
	}
}