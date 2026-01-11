/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Layer.h"
#include "LayerFactory.h"
#include <iostream>

using namespace std;
namespace beednn {

////////////////////////////////////////////////////////////////
Layer::Layer(const string &sType) : _sType(sType) {
  _bTrainMode = false;
  _bFirstLayer = false;
}
////////////////////////////////////////////////////////////////
Layer::~Layer() {}
///////////////////////////////////////////////////////////////
string Layer::type() const { return _sType; }
///////////////////////////////////////////////////////////////
void Layer::set_first_layer(bool bFirstLayer) { _bFirstLayer = bFirstLayer; }
///////////////////////////////////////////////////////////////
void Layer::set_train_mode(bool bTrainMode) { _bTrainMode = bTrainMode; }
////////////////////////////////////////////////////////////////
bool Layer::init(size_t &in, size_t &out,
                 vector<MatrixFloat> &internalCalculationMatrices, bool debug) {
  node_id = global_node_id++;

  if (debug) {
    std::cout << "Initializing " << _sType << " layer" << std::endl;
    std::cout << "Input size: " << in << std::endl;
    std::cout << "Output size: " << out << std::endl;
  }
  return false; // this function should be rewritten in
}
///////////////////////////////////////////////////////////////
bool Layer::has_weights() const { return false; }
///////////////////////////////////////////////////////////////
vector<MatrixFloat *> Layer::weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat *> Layer::gradient_weights() const {
  return std::vector<MatrixFloat *>();
}
///////////////////////////////////////////////////////////////
void Layer::set_initializer(const std::string &sWeightInitializer) {
  this->_sWeightInitializer = sWeightInitializer;
}
///////////////////////////////////////////////////////////////
std::string Layer::get_initializer() const { return _sWeightInitializer; }
///////////////////////////////////////////////////////////////
void Layer::save(std::ostream &to) const {}
///////////////////////////////////////////////////////////////
Layer *Layer::load(std::istream &from) { return LayerFactory::loadLayer(from); }
///////////////////////////////////////////////////////////////
Layer *Layer::construct(std::initializer_list<float> fArgs, std::string sArg) {
  return LayerFactory::construct(sArg, fArgs, sArg);
}
///////////////////////////////////////////////////////////////
std::string Layer::constructUsage() { return "error"; }
int Layer::global_node_id;
///////////////////////////////////////////////////////////////
} // namespace beednn