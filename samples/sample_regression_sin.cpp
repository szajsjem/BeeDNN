//This is a regression task sample.
//The goal is to approximate a sinus function and to evaluate the model error

#include <iostream>
#include <iomanip>
#include <cmath>

#include "Net.h"
#include "NetTrain.h"

#include "LayerDense.h"
#include "LayerActivation.h"

using namespace std;
using namespace beednn;

/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-10)
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
	cout << "This test a basic toy regression task." << endl;
	cout << "The task is to approximate a sinus function and to evaluate the model error" << endl;

	//build net
	Net model;
	model.add(new LayerDense(1, 10));
	model.add(new LayerActivation("Tanh"));
	model.add(new LayerDense(10, 1));

	model.set_classification_mode(false); //set regression mode
	model.init(1, true);

	//set train data
	MatrixFloat mTruth(128, 1), mTruth2(128, 1);
	MatrixFloat mSamples(128, 1), mSamples2(128, 1);
	for (int i = 0; i < 128; i++)
	{
		float x = i / 100.f;
		mTruth(i, 0) = sin(x);
		mSamples(i, 0) = x;
		mTruth2(i, 0) = sin(x + 0.005);
		mSamples2(i, 0) = x + 0.005;
	}

	//setup and train net
	cout << "Fitting..." << endl;
	NetTrain netfit;
	netfit.set_learningrate(0.001f);
	netfit.set_epochs(100000);
	netfit.set_train_data(mSamples, mTruth);
	netfit.set_validation_data(mSamples2, mTruth2);
	netfit.set_epoch_callback([&]() {
		static int epoch = 0;
		epoch++;
		if (epoch % 100 == 0)
		{
			float fLoss = netfit.compute_loss_accuracy(mSamples, mTruth);
			float fLoss2 = netfit.compute_loss_accuracy(mSamples2, mTruth2);
			cout << "Epoch " << epoch << " Loss=" << fLoss << " Validation=" << fLoss2 << endl;
		}
		});
	netfit.fit(model);

	//print truth and prediction
	MatrixFloat mPredict;
	model.predict(mSamples, mPredict);
	for (int i = 0; i < mSamples.size(); i += 8)
	{
		cout << setprecision(4) << "x=" << mSamples(i, 0) << "\ttruth=" << mTruth(i, 0) << "\tpredict=" << mPredict(i, 0) << endl;
		//test(is_near(mTruth(i, 0), mPredict(i, 0), 0.01));
	}
	//compute and print loss
	float fLoss = netfit.compute_loss_accuracy(mSamples, mTruth);
	cout << "Loss=" << fLoss << endl;
	test(fLoss < 1.e-5, "Loss must be < 1.e-5");

	float fMaxError = (mPredict - mTruth).cwiseAbs().maxCoeff();
	cout << "Max Error=" << fMaxError << endl;
	test(fMaxError < 0.01, "Max Error must be < 0.01");

	cout << "Test succeded." << endl;
	return 0;
}
