#include <iostream>

#include <fstream>

#include "Net.h"
#include "NetTrain.h"

#include "LayerEmbed.h"
#include "LayerDot.h"
#include "LayerDense.h"
#include "LayerActivation.h"
#include "LayerTransformerFeedForward.h"
#include "LayerTransformerHeads.h"
#include "LayerNormalize.h"
#include "Loss.h"
#include "Regularizer.h"
#include "LayerGatedActivation.h"
#include "LayerBatchTo2D.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace beednn;

int main() {//train loss 270
#ifdef linux
	freopen("/dev/pts/2", "w", stdout);
#endif
	const int dimmSize = 16;
	const int layers = 16;
	const int numheads = 4;
	const int vocabSize = 4;//max,min,close,vol, time
	const int FFmem =  4*dimmSize;
	const int QKVmem = dimmSize/4;
	const int seqlen = 1024;//equal to batch size
	cout << "started" << endl;

	Net model;
	model.add(new LayerDense(vocabSize, dimmSize, "Ones", "Zeros"));
	for (int i = 0; i < layers; i++) {
		//model.add(new LayerTransformerHeads(dimmSize, QKVmem, QKVmem, numheads, "Normal", "Normal"));
		model.add(new LayerParallel({
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerStacked({
					new LayerParallel({
						new LayerSequential({
							new LayerStacked({
								new LayerDense(dimmSize,QKVmem, "Normal", "Normal")
							},ROWSTACK,2),
							new LayerSelfAttention(),
							new LayerSoftmax()
						}),
						new LayerDense(dimmSize,QKVmem, "Normal", "Normal"),
					},DOT)
				},COLSTACK,numheads),
				new LayerDense(numheads * QKVmem, dimmSize, "Normal", "Normal")
			})
			}, SUM));
		//model.add(new LayerTransformerFeedForward(dimmSize, FFmem, "Relu", "Uniform", "Uniform"));//modified version below
		model.add(new LayerParallel({
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(dimmSize, FFmem * 2,"Normal","Normal"),
				new LayerGatedActivation("Sigmoid","LeakyRelu"),//added gated activation for tests
				new LayerDense(FFmem, dimmSize,"Normal","Normal")}),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(dimmSize, FFmem * 2,"Normal","Normal"),
				new LayerGatedActivation("Sigmoid","Sin"),//added gated activation for tests
				new LayerDense(FFmem, dimmSize,"Normal","Normal")}),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(dimmSize, FFmem * 2,"Normal","Normal"),
				new LayerGatedActivation("Sigmoid","Absolute"),//added gated activation for tests
				new LayerDense(FFmem, dimmSize,"Normal","Normal")})
			}, SUM));
	}
	//model.add(new LayerNormalize());
	model.add(new LayerDense(dimmSize, vocabSize, "Ones", "Zeros"));

	model.set_classification_mode(false);
	cout << "model created" << endl;
#ifdef linux
	//load csv
	fstream f("/home/pi/US30M1.10k.csv", ios::in);
#else
	fstream f("../../../../US30M1.10k.csv", ios::in);
#endif




	const int iSamples = 9900;//- seqlen

	MatrixFloat mSamples(iSamples, vocabSize);
	MatrixFloat mTruth(iSamples, vocabSize);
	//MatrixFloat mSamples2(iSamples/2, vocabSize);
	//MatrixFloat mTruth2(iSamples/2, vocabSize);
	
	float trash, prvclose;
	
	
	f >> mSamples(0, 0);
	f >> mSamples(0, 1);
	f >> mSamples(0, 2);
	f >> mSamples(0, 3);
	f >> trash;// mSamples(0, 4);

	prvclose = mSamples(0, 2);

	for (int i = 1; i < iSamples; i++) {
		for (int j = 0; j < 3; j++) {
			f >> mSamples(i, j);
			mTruth(i - 1, j) = mSamples(i, j)-prvclose;
		}
		f >> mSamples(i, 3);
		mTruth(i - 1, 3) = mSamples(i, 3);
		f >> trash;
		prvclose = mSamples(i, 2);
	}
	cout << "data loaded" << endl;
	//f >> mSamples2(0, 0);
	//f >> mSamples2(0, 1);
	//f >> mSamples2(0, 2);
	//f >> mSamples2(0, 3);
	//f >> trash;// mSamples2(0, 4);
	//prvclose = mSamples2(0, 2);

	//for (int i = 1; i < iSamples/2; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		f >> mSamples2(i, j);
	//		mTruth2(i - 1, j) = mSamples2(i, j) - prvclose;
	//	}
	//	f >> mSamples2(i, 3);
	//	mTruth2(i - 1, 3) = mSamples2(i, 3);
	//	f >> trash;
	//}

	//optimize network
	NetTrain netFit;
	//netFit.set_optimizer("Adagrad");
	//netFit.set_regularizer("GradientNormClip");
	netFit.set_loss("AdjustedMeanSquaredError");
	netFit.set_train_data(mSamples, mTruth);
	netFit.set_learningrate(0.0001);
	netFit.set_RandomBatchOrder(false);
	netFit.set_epochs(seqlen+10);
	int bsize = 1;
	netFit.set_validation_batchsize(bsize);
	netFit.set_batchsize(bsize++);
	netFit.set_batchstepsize(1);
	//netFit.set_validation_data(mSamples2, mTruth2);
	netFit.set_keepbest(true);
	float prevloss = 1e10;
	bsize = 1;
	netFit.set_epoch_callback([&]() {
		if (netFit.get_current_train_loss() > prevloss)
			netFit.set_learningrate(netFit.get_learningrate() * 0.99);
		prevloss = netFit.get_current_train_loss();

		cout <<"epoch:"<<bsize << " train_loss=" << prevloss << " selection_loss=" << netFit.get_current_validation_loss() << endl;

		if (bsize > seqlen)bsize = seqlen;
		netFit.set_validation_batchsize(bsize);
		netFit.set_batchsize(bsize++);
		});

	cout << "starting training" << endl;
	netFit.fit(model);
	cout << "slowfit" << endl;
	while(1)netFit.slowfit(model);
}