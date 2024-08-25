#include "LayerRouter.h"

namespace beednn {
	LayerRouter::LayerRouter(Layer* RouterLayer, float selectedexperts, std::vector<Layer*> mExperts, ParallelReduction mReduction)
	:Layer("LayerRouter")
	{
		for (auto layer : mExperts) {
			_Layers.push_back(layer);
		}
		assert(mReduction != DOT);
		_ParallelReduction = mReduction;
		_router = RouterLayer;
		computeLayers = selectedexperts;
		LayerRouter::init();
	}
	LayerRouter::LayerRouter()
		:Layer("LayerRouter")
	{
	}
	LayerRouter::~LayerRouter()
	{
		for (auto x : _Layers) {
			delete x;
		}
		delete _router;
	}
	Layer* LayerRouter::clone() const
	{
		LayerRouter* pLayer = new LayerRouter();
		for (auto x : _Layers) {
			pLayer->_Layers.push_back(x->clone());
		}
		pLayer->_ParallelReduction = _ParallelReduction;
		pLayer->_router = _router->clone();
		pLayer->init();
		return pLayer;
	}
	void LayerRouter::init()
	{
		Layer::init();
	}
	void LayerRouter::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		MatrixFloat routing;
		_router->forward(mIn, routing);
#if 1
#define LayerRouterVersion1
		for (int i = 0; i < _Layers.size();i++) {
			MatrixFloat mf;
			_Layers[i]->forward(mIn, mf);
			mf.array() *= routing.col(i).array();
			if (i == 0) {
				mOut = mf;
			}
			else if (_ParallelReduction == SUM) {
				mOut += mf;
			}
			else if (_ParallelReduction == ROWSTACK) {
				mOut = concatenateRows(mOut, mf);
			}
			else if (_ParallelReduction == COLSTACK) {
				mOut = concatenateCols(mOut, mf);
			}
		}
#else
		std::vector<MatrixFloat> outs(computeLayers);
		for (int i = 0; i < routing.rows(); i++) {
			std::vector<std::pair<float, int>> sorted;
			for(int j=0;j<routing.cols();j++)
				sorted.push_back(std::pair<float, int>(routing(i, j), j));
			std::sort(sorted.begin(), sorted.end());
			if (i == 0) {
				MatrixFloat a=mIn.row(0), b;
				_Layers[0]->forward(a, b);
				for (int j = 0; j < computeLayers; j++)
					outs[j].resize(mIn.rows(), b.cols());
			}
			MatrixFloat a = mIn.row(i), b;
			for (int j = 0; j < computeLayers; j++) {
				_Layers[sorted[j].second]->forward(a, b);
				outs[j].row(i) = b.row(0);
			}
		}
		for (int i = 0; i < computeLayers;i++) {
			if (i == 0) {
				mOut = outs[0];
			}
			else if (_ParallelReduction == SUM) {
				mOut += outs[i];
			}
			else if (_ParallelReduction == DOT) {
				mOut *= outs[i];
			}
			else if (_ParallelReduction == ROWSTACK) {
				mOut = concatenateRows(mOut, outs[i]);
			}
			else if (_ParallelReduction == COLSTACK) {
				mOut = concatenateCols(mOut, outs[i]);
			}
		}
#endif
	}
	void LayerRouter::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
#ifdef LayerRouterVersion1

#else
		//todo
#endif 

	}
	bool LayerRouter::has_weights() const
	{
		for (auto layer : _Layers)
			if (layer->has_weights())
				return true;
		return _router->has_weights();
	}
	std::vector<MatrixFloat*> LayerRouter::weights()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_weights()) {
				auto vi = layer->weights();
				if (vi.size() > 0) {//this probably not needed
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		auto vi = _router->weights();
		if (vi.size() > 0) {//this probably not needed
			v.insert(v.end(), vi.begin(), vi.end());
		}
		return v;
	}
	std::vector<MatrixFloat*> LayerRouter::gradient_weights()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_weights()) {
				auto vi = layer->gradient_weights();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		auto vi = _router->gradient_weights();
		if (vi.size() > 0) {
			v.insert(v.end(), vi.begin(), vi.end());
		}
		return v;
	}
	bool LayerRouter::has_biases() const
	{
		for (auto layer : _Layers)
			if (layer->has_biases())
				return true;
		return _router->has_biases();
	}
	std::vector<MatrixFloat*> LayerRouter::biases()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_biases()) {
				auto vi = layer->biases();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		auto vi = _router->biases();
		if (vi.size() > 0) {
			v.insert(v.end(), vi.begin(), vi.end());
		}
		return v;
	}
	std::vector<MatrixFloat*> LayerRouter::gradient_biases()
	{
		std::vector<MatrixFloat*> v;
		for (auto layer : _Layers)
			if (layer->has_biases()) {
				auto vi = layer->gradient_biases();
				if (vi.size() > 0) {
					v.insert(v.end(), vi.begin(), vi.end());
				}
			}
		auto vi = _router->gradient_biases();
		if (vi.size() > 0) {
			v.insert(v.end(), vi.begin(), vi.end());
		}
		return v;
	}
}