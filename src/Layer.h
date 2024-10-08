/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Matrix.h"

#include <string>
#include <vector>

namespace beednn {
class Layer
{
public:
    Layer(const std::string& sType);
    virtual ~Layer();

    virtual Layer* clone() const =0;
    std::string type() const;
	void set_first_layer(bool bFirstLayer);

    virtual void forward(const MatrixFloat& mIn,MatrixFloat& mOut) =0;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)=0;
	
	void set_train_mode(bool bTrainMode); //set to true to train, to false to test

    void set_weight_initializer(const std::string& _sWeightInitializer);
    std::string weight_initializer() const;
    virtual bool has_weights() const;
    virtual std::vector<MatrixFloat*> weights();
    virtual std::vector<MatrixFloat*> gradient_weights();

    void set_bias_initializer(const std::string& _sBiasInitializer);
    std::string bias_initializer() const;
    virtual bool has_biases() const;
    virtual std::vector<MatrixFloat*> biases();
    virtual std::vector<MatrixFloat*> gradient_biases();

protected:
    MatrixFloat _weight,_gradientWeight;
	MatrixFloat _bias, _gradientBias;
	bool _bTrainMode;
	bool _bFirstLayer;

private:
    std::string _sType;
    std::string _sWeightInitializer, _sBiasInitializer;
};
}
