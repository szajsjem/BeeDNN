// this sample shows how to do a simple classification
// the use case is to learn a XOR gate

#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"

int main()
{
    //construct network, 2 input, 1 output
    Net net;
    net.add_dense_layer(2,3);
	net.add_activation_layer("Relu");
	net.add_dense_layer(3, 1);

    //set the train data
    float dSamples[]={ 0,0 , 0,1 , 1,0 , 1,1 };
    float dTruths[]={ 0 , 1 , 1, 0 };
    const MatrixFloat mSamples=fromRawBuffer(dSamples,4,2);
    const MatrixFloat mTruth=fromRawBuffer(dTruths,4,1);

    //optimize network
	NetTrain netFit;
	netFit.set_epochs(1000);
	netFit.set_learning_data(mSamples, mTruth);
	netFit.fit(net);

    //predict and show results
    MatrixFloat mOut;
	net.predict(mSamples, mOut);
    cout << "0_xor_0=" << mOut(0) << endl << "0_xor_1=" << mOut(1) << endl << "1_xor_0=" << mOut(2) << endl << "1_xor_1=" << mOut(3) << endl;

    return 0;
}
