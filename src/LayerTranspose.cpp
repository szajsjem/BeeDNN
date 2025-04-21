#include "LayerTranspose.h"

beednn::LayerTranspose::LayerTranspose()
	:Layer("LayerTranspose")
{
}

beednn::LayerTranspose::~LayerTranspose()
{
}

beednn::Layer* beednn::LayerTranspose::clone() const
{
	return new LayerTranspose();
}

void beednn::LayerTranspose::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	mOut = mIn.transpose();
}

void beednn::LayerTranspose::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	mGradientIn = mGradientOut.transpose();
}

///////////////////////////////////////////////////////////////////////////////
void beednn::LayerTranspose::save(std::ostream& to) const {

}
///////////////////////////////////////////////////////////////
beednn::Layer* beednn::LayerTranspose::load(std::istream& from) {
	return NULL;
}
///////////////////////////////////////////////////////////////
beednn::Layer* beednn::LayerTranspose::construct(std::initializer_list<float> fArgs, std::string sArg) {
	if (fArgs.size() != 0) return nullptr;
	return new LayerTranspose();
}
///////////////////////////////////////////////////////////////
std::string beednn::LayerTranspose::constructUsage() {
	return "transposes input matrix\n \n ";
}
///////////////////////////////////////////////////////////////
bool beednn::LayerTranspose::has_weights() const
{
	return false;
}
///////////////////////////////////////////////////////////////////////////////
bool beednn::LayerTranspose::init(size_t& in, size_t& out, bool debug)
{
	//except l2d
	out = in;
	Layer::init(in, out, debug);
	return true;
}