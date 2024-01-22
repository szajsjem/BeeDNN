#include <iostream>

#include "Net.h"
#include "NetTrain.h"

#include "LayerEmbed.h"
#include "LayerDot.h"
#include "LayerDense.h"
#include "LayerActivation.h"
#include "LayerTransformerFeedForward.h"
#include "LayerTransformerHeads.h"
#include "LayerNormalize.h"

using namespace std;
using namespace beednn;


int main()
{

	const int dimmSize = 48;
	const int layers = 4;
	const int numheads = 4;
	const int vocabSize = 2;
	const int seqlen = 6;

	Net model;
	model.add(new LayerEmbed(vocabSize, dimmSize, seqlen,"Uniform"));
	for (int i = 0; i < layers; i++) {
		model.add(new LayerTransformerHeads(dimmSize, dimmSize / numheads, dimmSize / numheads, numheads));
		model.add(new LayerTransformerFeedForward(dimmSize, 4 * dimmSize, "Relu"));
	}
	model.add(new LayerNormalize());
	model.add(new LayerDot(dimmSize, vocabSize));
	model.add(new LayerSoftmax());

	model.set_classification_mode(false);

	//set the train data
	float dSamples[] = { 1,   0,   1,   0,   1,   0 };
	float dTruths[] = {  1,0, 0,1, 1,0, 0,1, 1,0, 0,1, };
	MatrixFloat mSamples = fromRawBuffer(dSamples, 6, 1);
	const MatrixFloat mTruth = fromRawBuffer(dTruths, 6, 2);

	//optimize network
	NetTrain netFit;
	netFit.set_epochs(500);
	netFit.set_train_data(mSamples, mTruth);

	//predict and show results
	netFit.fit(model);
	MatrixFloat mOut;

	mSamples(0, 0) = 1;
	mSamples(1, 0) = 0;
	mSamples(2, 0) = 1;
	mSamples(3, 0) = 0;
	mSamples(4, 0) = 1;
	mSamples(5, 0) = 0;

	model.predict(mSamples, mOut);
	cout << "next_tok= " << mOut(5,0)<<" "<< mOut(5, 1) << endl;

	cout << "Test succeded." << endl;
	return 0;
}