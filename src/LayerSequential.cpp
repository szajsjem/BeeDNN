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
		for (auto x : _Layers) {//todo optimize this in netfit
			MatrixFloat mf;
			x->forward(*(in.end()-1), mf);
			in.push_back(mf);
		}
		in.pop_back();
		MatrixFloat grad = mGradientOut;
		for (auto it = _Layers.rbegin(); it != _Layers.rend(); it++) {
			(*it)->backpropagation(*(in.end()-1), grad, mGradientIn);
			grad = mGradientIn;
			in.pop_back();
		}
	}
	///////////////////////////////////////////////////////////////
	bool LayerSequential::has_weights() const
	{
		for (auto layer : _Layers)
			if (layer->has_weights())
				return true;
		return false;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerSequential::weights()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_weights()) {
				auto vi = layer->weights();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		return v;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerSequential::gradient_weights()
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
	///////////////////////////////////////////////////////////////
	bool LayerSequential::has_biases() const
	{
		for (auto layer : _Layers)
			if (layer->has_biases())
				return true;
		return false;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerSequential::biases()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers) 
			if (layer->has_biases()){
				auto vi = layer->biases();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
		}
		return v;
	}
	///////////////////////////////////////////////////////////////
	std::vector<MatrixFloat*> LayerSequential::gradient_biases()
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