#include "LayerSequential.h"

namespace beednn {
	LayerSequential::LayerSequential(std::vector<Layer*> mSequentialLayers)
		:Layer("LayerSequential")
	{
		for (auto layer : mSequentialLayers) {
			_Layers.push_back(layer);
		}
		LayerSequential::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerSequential::LayerSequential() :
		Layer("LayerSequential")
	{
	}
	///////////////////////////////////////////////////////////////////////////////
	LayerSequential::~LayerSequential()
	{
		for (auto x : _Layers) {
			delete x;
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	Layer* LayerSequential::clone() const
	{
		LayerSequential* pLayer = new LayerSequential();
		for (auto x : _Layers) {
			pLayer->_Layers.push_back(x->clone());
		}
		pLayer->init();
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSequential::init()
	{
		Layer::init();
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSequential::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		MatrixFloat mf = mIn;
		for (auto x : _Layers) {
			x->forward(mf, mOut);
			mf = mOut;
		}
	}
	///////////////////////////////////////////////////////////////////////////////
	void LayerSequential::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		std::vector<MatrixFloat> in; in.push_back(mIn);
		for (auto x : _Layers) {
			MatrixFloat mf;
			x->forward(*(in.end()), mf);
			in.push_back(mf);
		}
		in.pop_back();
		MatrixFloat grad = mGradientOut;
		for (auto it = _Layers.rbegin(); it != _Layers.rend(); it++) {
			(*it)->backpropagation(*(in.end()), grad, mGradientIn);
			grad = mGradientIn;
			in.pop_back();
		}
	}
}