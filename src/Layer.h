/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerFactory.h"
#include "Matrix.h"

#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace beednn {
class Layer {
public:
  Layer(const std::string &sType);
  virtual ~Layer();

  virtual Layer *clone() const = 0;
  std::string type() const;
  void set_first_layer(bool bFirstLayer);

  virtual void forward(const MatrixFloat &mIn, MatrixFloat &mOut) = 0;

  virtual bool init(size_t &in, size_t &out,
                    std::vector<MatrixFloat> &internalCalculationMatrices,
                    bool debug = false); // true on initialization success
  virtual void
  backpropagation(const MatrixFloat &mIn, const MatrixFloat &mGradientOut,
                  MatrixFloat &mGradientIn,
                  std::vector<MatrixFloat> &internalCalculationMatrices,
                  int start) = 0;

  void set_train_mode(bool bTrainMode); // set to true to train, to false to
                                        // test

  void set_initializer(const std::string &_sWeightInitializer);
  std::string get_initializer() const;

  virtual bool has_weights() const;
  virtual std::vector<MatrixFloat *> weights() const;
  virtual std::vector<MatrixFloat *> gradient_weights() const;

  virtual void save(std::ostream &to) const = 0;
  static Layer *load(std::istream &from);
  static Layer *construct(std::initializer_list<float> fArgs, std::string sArg);
  static std::string constructUsage();

protected: /*
     MatrixFloat _weight,_gradientWeight;
         MatrixFloat _bias, _gradientBias;*/
  bool _bTrainMode;
  bool _bFirstLayer;

private:
  std::string _sType;
  std::string _sWeightInitializer;

protected:
  static int global_node_id; // For generating unique node IDs
  std::vector<Layer *> input_connections;

public:
  int node_id;

  virtual std::string get_layer_name() const {
    // Remove "Layer" prefix if present
    std::string name = _sType;
    if (name.substr(0, 5) == "Layer") {
      name = name.substr(5);
    }
    return name;
  }
  // Graph generation
  virtual std::string generate_mermaid_node() const {
    std::stringstream ss;
    ss << "    " << node_id << "[\"" << get_layer_name();
    // Add dimensions if available
    if (has_weights()) {
      auto w = this->weights();
      if (!w.empty() && w[0]) {
        ss << "<br/>(" << w[0]->rows() << "x" << w[0]->cols() << ")";
      }
    }
    ss << "\"]";
    return ss.str();
  }
  virtual std::string generate_mermaid_connections() const {
    std::stringstream ss;
    for (const auto *input : input_connections) {
      ss << "    " << input->node_id << " --> " << node_id << "\n";
    }
    return ss.str();
  }
  virtual void add_connection(Layer *input_layer) {
    input_connections.push_back(input_layer);
  }

  // New method to generate complete graph
  static std::string generate_network_graph(Layer *output_layer) {
    std::set<const Layer *> visited;
    std::stringstream ss;

    ss << "graph TD\n";

    // Helper function to recursively generate graph
    std::function<void(const Layer *)> generate_graph =
        [&](const Layer *layer) {
          if (visited.find(layer) != visited.end()) {
            return;
          }

          visited.insert(layer);

          // First generate nodes for all inputs
          for (const auto *input : layer->input_connections) {
            generate_graph(input);
          }

          // Generate this node
          ss << layer->generate_mermaid_node() << "\n";

          // Generate connections
          ss << layer->generate_mermaid_connections();
        };

    generate_graph(output_layer);
    return ss.str();
  }
};

// REGISTER_LAYER(Layer, "none");
} // namespace beednn