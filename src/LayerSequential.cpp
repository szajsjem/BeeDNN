#include "LayerSequential.h"

namespace beednn {
	LayerSequential::LayerSequential(std::vector<Layer*> mSequentialLayers)
		:Layer("LayerSequential")
	{
		for (auto layer : mSequentialLayers) {
			_Layers.push_back(layer);
		}
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
		return pLayer;
	}
	///////////////////////////////////////////////////////////////////////////////
	bool LayerSequential::init(size_t& in, size_t& out, bool debug)
	{
		size_t tmp=in,tmpo;
		for (auto l : _Layers) {
			tmpo = -1;
			if (!l->init(tmp, tmpo, debug))
				return false;
			if (in == -1)
				in = tmp;
			tmp = tmpo;
		}
		if (out != -1)
			if (tmp != out)
				return false;
		out = tmp;
		Layer::init(in, out, debug);
		return true;
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
	std::vector<MatrixFloat*> LayerSequential::weights() const
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
	std::vector<MatrixFloat*> LayerSequential::gradient_weights() const
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
	void LayerSequential::save(std::ostream& to) const {

	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSequential::load(std::istream& from) {
		return NULL;
	}
	///////////////////////////////////////////////////////////////
	Layer* LayerSequential::construct(std::initializer_list<float> fArgs, std::string sArg) {
		if (fArgs.size() != 0) return nullptr;

		// Parse layer pointers string - split on commas
		std::vector<Layer*> layers;
		std::string::size_type start = 0;
		std::string::size_type end;

		while ((end = sArg.find(',', start)) != std::string::npos) {
			Layer* layer = (Layer*)std::stoull(sArg.substr(start, end - start), nullptr, 16);
			layers.push_back(layer);
			start = end + 1;
		}
		// Get last layer
		if (start < sArg.length()) {
			Layer* layer = (Layer*)std::stoull(sArg.substr(start), nullptr, 16);
			layers.push_back(layer);
		}

		return new LayerSequential(layers);
	}
	///////////////////////////////////////////////////////////////
	std::string LayerSequential::constructUsage() {
		return "sequential layer container\nlayers\niNumLayers";
	}
	///////////////////////////////////////////////////////////////
}