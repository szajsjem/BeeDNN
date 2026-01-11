#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

#include "Net.h"
#include "NetTrain.h"

#include "LayerDense.h"
#include "LayerGatedActivation.h"
#include "LayerParallel.h"
#include "LayerActivation.h"


using namespace beednn;


float func(float x) {
	return x;
}

float inversefunc(float y) {
	return y - sin(y);
}

#define sizz 512

int main()
{
	setlocale(LC_ALL, "fr-CA");

	//network structure
	Net model;
	model.add(new LayerDense(1, 20));
	model.add(new LayerActivation("Tanh"));
	model.add(new LayerDense(-1, 1));

	model.init(1);
	model.set_classification_mode(false); //set regression mode

	//collect the training data
	MatrixFloat mY(sizz, 1);
	MatrixFloat mX(sizz, 1);
	for (int i = 0; i < sizz; i++)
	{
		float x = i / 100.f;
		mY(i, 0) = func(x);
		mX(i, 0) = inversefunc(x);
	}
	//validation data
	MatrixFloat mYv(sizz, 1);
	MatrixFloat mXv(sizz, 1);
	for (int i = 0; i < sizz; i++)
	{
		float x = (- i) / 100.f;
		mYv(i, 0) = func(x);
		mXv(i, 0) = inversefunc(x);
	}

	FILE* f = NULL;
	fopen_s(&f,"out.csv", "w");
	fprintf(f, "epoch;train loss;validation loss\n");

	//training
	NetTrain netfit;
	netfit.set_epochs(5000);
	netfit.set_train_data(mX, mY);
	netfit.set_validation_data(mXv, mYv);
	netfit.set_batchsize(64);
	netfit.set_epoch_callback([&]() {
		static int epoch = 0;
		fprintf(f, "%d;%f;%f\n", epoch++, netfit.compute_loss_accuracy(mX, mY), netfit.compute_loss_accuracy(mXv, mYv));
		});
	netfit.fit(model);
	fclose(f);

	//print prediction
	MatrixFloat mPredict;
	model.predict(mX, mPredict);
	for (int i = 0; i < mX.size(); i += 8)
	{
		std::cout << std::setprecision(5) << "x=" << mX(i, 0) << "\ty=" << mY(i, 0) << "\tpredicted=" << mPredict(i, 0) << std::endl;
	}

	//compute and print loss
	float fLoss = netfit.compute_loss_accuracy(mX, mY);
	std::cout << "Loss=" << fLoss << std::endl;
	float fLossv = netfit.compute_loss_accuracy(mXv, mYv);
	std::cout << "validation loss=" << fLossv << std::endl;


	fopen_s(&f, "func.csv", "w");
	fprintf(f, "x;y;computed\n");

	for (int i = -1024; i < 1024; i++)
	{
		float t = i / 100.f;
		float x = inversefunc(t);
		float y = func(t);
		MatrixFloat in(1,1), out(1,1);
		in(0, 0) = x;
		model.predict(in, out);

		fprintf(f, "%f;%f;%f\n", x, y, out(0, 0));
	}
	fclose(f);

	return 0;
}










