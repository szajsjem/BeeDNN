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
		mOut.resize(0, 0);//todo save output sizes of diffrent layers
		for (auto x : _Layers) {
			MatrixFloat mf;
			x->forward(mIn, mf);
			if (mOut.size() == 0 || *_Layers.begin()==x) {
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
	std::vector<MatrixFloat*> LayerParallel::weights()
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
	std::vector<MatrixFloat*> LayerParallel::gradient_weights()
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
	///////////////////////////////////////////////////////////////
	bool LayerParallel::has_biases() const
	{
		for (auto layer : _Layers)
			if (layer->has_biases())
				return true;
		return false;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerParallel::biases()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_biases()) {
				auto vi = layer->biases();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		return v;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerParallel::gradient_biases()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_biases()) {
				auto vi = layer->gradient_biases();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		return v;
	}
	///////////////////////////////////////////////////////////////
}