#include "LayerBatchTo2D.h"

namespace beednn {
	LayerBatchTo2D::LayerBatchTo2D(const int incolsize, const int outcolsize, const int outrowsize, Layer* l2d)
	:Layer("LayerBatchTo2D"){
		_l2d = l2d;
		_incolsize = incolsize;
		_outcolsize = outcolsize;
		_outrowsize = outrowsize;
		LayerBatchTo2D::init();
	}

	LayerBatchTo2D::~LayerBatchTo2D()
	{
		delete _l2d;
	}

	Layer* LayerBatchTo2D::clone() const
	{
		return new LayerBatchTo2D(_incolsize, _outcolsize,_outrowsize, _l2d->clone());
	}

	void LayerBatchTo2D::init()
	{
		Layer::init();
	}

	void LayerBatchTo2D::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
	{
		mOut.resize(mIn.rows(), _outcolsize * _outrowsize);
		for (int i = 0; i < mIn.rows(); i++) {
			MatrixFloat out;
			_l2d->forward(fromRawBuffer(mIn.row(i).data(), mIn.cols() / _incolsize, _incolsize), out);
			if (out.cols() != _outcolsize || out.rows() != _outrowsize) {
				assert(i==0);
				_outcolsize = out.cols();
				_outrowsize = out.rows();
				mOut.resize(mIn.rows(), _outcolsize * _outrowsize);
			}
			mOut.row(i) = fromRawBuffer(out.data(), 1, out.size());
		}
	}

	void LayerBatchTo2D::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
	{
		mGradientIn.resizeLike(mIn);
		std::vector<MatrixFloat*> gradients;
		std::vector<MatrixFloat> sumgradient;
		if (_l2d->has_biases())
			for (auto& g : _l2d->biases())
				gradients.push_back(g);
		if (_l2d->has_weights())
			for (auto& g : _l2d->weights())
				gradients.push_back(g);
		assert(mGradientOut.cols() % _outcolsize == 0);
		assert(mIn.cols() % _incolsize == 0);
		const int inrowsize = mIn.cols() / _incolsize;
		for (int i = 0; i < mIn.rows(); i++) {
			MatrixFloat gradin,
				in = fromRawBuffer(mIn.row(i).data(), inrowsize, _incolsize),
				gradOut = fromRawBuffer(mGradientOut.row(i).data(), _outrowsize, _outcolsize);
			_l2d->backpropagation(in,gradOut, gradin);
			if(!_bFirstLayer)
				mGradientIn.row(i) = fromRawBuffer(gradin.data(), 1, gradin.size());
			if (i == 0) {
				for(auto& g:gradients)
					sumgradient.push_back(*g);
			}
			else {
				for (int i = 0; i < gradients.size(); i++)
					sumgradient[i] += *(gradients[i]);
			}
		}
		for (int i = 0; i < gradients.size(); i++)
			*gradients[i] = sumgradient[i]/(mIn.rows()* inrowsize);
	}
	bool LayerBatchTo2D::has_weights() const
	{
		return _l2d->has_weights();
	}
	std::vector<MatrixFloat*> LayerBatchTo2D::weights()
	{
		return _l2d->weights();
	}
	std::vector<MatrixFloat*> LayerBatchTo2D::gradient_weights()
	{
		return _l2d->gradient_weights();
	}
	bool LayerBatchTo2D::has_biases() const
	{
		return _l2d->has_biases();
	}
	std::vector<MatrixFloat*> LayerBatchTo2D::biases()
	{
		return _l2d->biases();
	}
	std::vector<MatrixFloat*> LayerBatchTo2D::gradient_biases()
	{
		return _l2d->gradient_biases();
	}
}