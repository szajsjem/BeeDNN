#include <iostream>


#include "Net.h"
#include "NetTrain.h"

#include "LayerDense.h"
#include "LayerActivation.h"
#include "LayerSequential.h"

using namespace beednn;
using namespace std;
/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 0.001)
{
	return fabs(a - b) < tolerancis;
}
void test(bool bTest, const string& sMessage = "")
{
	if (bTest) return;

	cout << "Test failed: " << sMessage << endl;
	exit(-1);
}
/////////////////////////////////////////////////////////////////////

int main()
{
	cout << "Test a simple xor classification using layer Sequential" << endl;
	cout << "The usecase is to learn a XOR gate" << endl;

	//construct network, 2 input, 1 output
	Net model;
	model.add(new LayerSequential({ 
		new LayerDense(2, 10),
		new LayerActivation("Relu"),
		new LayerDense(10, 1)
		}));

	//set the train data
	float dSamples[] = { 0,0 , 0,1 , 1,0 , 1,1 };
	float dTruths[] = { 0 , 1 , 1, 0 };
	const MatrixFloat mSamples = fromRawBuffer(dSamples, 4, 2);
	const MatrixFloat mTruth = fromRawBuffer(dTruths, 4, 1);

	//optimize network
	NetTrain netFit;
	netFit.set_epochs(500);
	netFit.set_train_data(mSamples, mTruth);

	//predict and show results
	netFit.fit(model);
	MatrixFloat mOut;
	model.predict_classes(mSamples, mOut);
	cout << "0_xor_0=" << mOut(0) << endl << "0_xor_1=" << mOut(1) << endl << "1_xor_0=" << mOut(2) << endl << "1_xor_1=" << mOut(3) << endl;

	//simple testU code
	test(is_near(mOut(0), 0.));
	test(is_near(mOut(1), 1.));
	test(is_near(mOut(2), 1.));
	test(is_near(mOut(3), 0.));

	cout << "Test succeded." << endl;
	return 0;
}