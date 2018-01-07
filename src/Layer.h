#ifndef Layer_
#define Layer_

#include "Matrix.h"

class Layer
{
public:
    Layer(int iInSize,int iOutSize);
    virtual ~Layer();
    virtual void init() =0;

    virtual void forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const =0;
    virtual void forward_save(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) =0;

    virtual MatrixFloat get_weight_activation_derivation() const =0;
    virtual MatrixFloat& get_weight() =0;

    //learning variables, todo clean up / remove
    MatrixFloat in; //in
    MatrixFloat out; //fcn(in*weight)
    MatrixFloat outWeight; // in*weight
    MatrixFloat dE;

protected:
    int _iInSize, _iOutSize;
};

#endif
