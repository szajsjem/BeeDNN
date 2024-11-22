/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "LayerFactory.h"
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
	
    virtual bool init(size_t &in, size_t &out, bool debug=false);//true on initialization success
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)=0;
	
	void set_train_mode(bool bTrainMode); //set to true to train, to false to test

    void set_initializer(const std::string& _sWeightInitializer);
    std::string get_initializer() const;

    virtual bool has_weights() const;
    virtual std::vector<MatrixFloat*> weights();
    virtual std::vector<MatrixFloat*> gradient_weights();

    virtual void save(std::ostream& to)const = 0;
    static Layer* load(std::istream& from);
    static Layer* construct(std::initializer_list<float> fArgs, std::string sArg);
    static std::string constructUsage();

protected:/*
    MatrixFloat _weight,_gradientWeight;
	MatrixFloat _bias, _gradientBias;*/
	bool _bTrainMode;
	bool _bFirstLayer;

private:
    std::string _sType;
    std::string _sWeightInitializer;
};

//REGISTER_LAYER(Layer, "none");
}