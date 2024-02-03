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
#include "Loss.h"
#include "Regularizer.h"
#include "LayerGatedActivation.h"
#include "LayerBatchTo2D.h"

using namespace std;
using namespace beednn;

MatrixFloat sampleAllTokens(MatrixFloat mPredict) {
	MatrixFloat mOut(mPredict.rows(), 1);
	for (int i = 0; i < mOut.rows(); i++) {
		int m = 0;
		float max = mPredict(i,0);
		for (int j = 1; j < mPredict.cols(); j++) {
			if (max < mPredict(i, j)) {
				max = mPredict(i, j);
				m = j;
			}
		}
		mOut(i, 0) = m;
	}
	return mOut;
}

int sampleNextToken(MatrixFloat mPredict) {
	auto o = mPredict.row(mPredict.rows() - 1);
	int m = 0;
	float max = o(0);
	for (int j = 1; j < mPredict.cols(); j++) {
		if (max < o(j)) {
			max = o(j);
			m = j;
		}
	}
	return m;
}

int sampleNextToken(MatrixFloat mPredict, const int skipfirst) {
	auto o = mPredict.row(0);
	int m = skipfirst;
	float max = o(skipfirst);
	for (int j = skipfirst+1; j < mPredict.cols(); j++) {
		if (max < o(j)) {
			max = o(j);
			m = j;
		}
	}
	return m- skipfirst;
}

int nextToken(std::string str, int& start, int end, std::vector<std::string> vocabulary) {
	for (int i = 0; i < vocabulary.size(); i++) {
		int c = strcmp(str.substr(start,vocabulary[i].size()).data(), vocabulary[i].data());
		if (c==0) {
			start += vocabulary[i].size();
			return i;
		}
	}
	return -1;
}

std::vector<int> tokenizeString(std::string data, std::vector<std::string> vocabulary) {
	std::vector<int> tokenized;
	int start = 0, t;
	while ((t = nextToken(data, start, data.size(), vocabulary)) >= 0) {
		tokenized.push_back(t);
	}
	return tokenized;
}

void testLayerGradientImpl(Layer* ln, MatrixFloat inputData) {
	Loss* l = create_loss("MeanSquaredError");
	MatrixFloat mF, mB, mL, mG;
	ln->forward(inputData, mF);
	MatrixFloat mb = mF;
	for (int i = 0; i < mb.size(); i++)
		mb(i / mb.cols(), i % mb.cols()) += rand() / (float)RAND_MAX - 0.5;
	l->compute(mF, mb, mL);
	float loss = mL.mean();
	l->compute_gradient(mF, mb, mG);
	ln->backpropagation(inputData, mG, mB);
	MatrixFloat maa = inputData - mB / 1000;
	ln->forward(maa, mF);
	l->compute(mF, mb, mL);
	float loss2 = mL.mean();
	if (loss2 < loss) {
		printf("pased gradient passthrough:%f\n", loss - loss2);
	}
	else {
		printf("backpropagation is wrong\n");
	}

	auto bs = ln->biases();
	auto bg = ln->gradient_biases();
	for (int i = 0; i < bs.size(); i++) {
		*bs[i] -= (*bg[i]) / 1000;
	}
	auto ws = ln->weights();
	auto wg = ln->gradient_weights();
	for (int i = 0; i < ws.size(); i++) {
		*ws[i] -= (*wg[i]) / 1000;
	}

	ln->forward(inputData, mF);
	l->compute(mF, mb, mL);
	float loss3 = mL.mean();
	if (loss3 < loss) {
		printf("pased gradient apply:%f\n", loss - loss3);
	}
	else {
		printf("backpropagation is wrong\n");
	}
	delete l;
}

void test() {
	float a[] = { 1.1, 0.9,   1,   1, 0.9,   1, 1.1,   1, 1.1, 0.9,   1,   1,   1, 1.1,   1,   1 };
	MatrixFloat ma = fromRawBuffer(a, 4, 4);


	Layer* ln = new LayerTransformerHeads(4, 4, 4, 4, "Uniform", "Uniform"); //new LayerTransformerFeedForward(4, 16, "Relu", "Uniform", "Uniform");//
	//new LayerNormalize();
	//new LayerParallel({}, DOT);
	//new LayerEmbed();
	//new LayerSelfAttention();
	//new LayerSelfDot();
	//new LayerTransformerFeedForward();
	//new LayerTransformerHeads();

	testLayerGradientImpl(ln, ma);
}

char* rndstr(int l) {
	static const char* sl = "qwertyuiopasdfghjklzxcvbnm";// QWERTYUIOPASDFGHJKLZXCVBNM";//,./<>?;':\"[] {}-=_+!@#$%^&*()~`\\|
	char* out = new char[l + 1L];
	out[l] = 0;
	for (int i = 0; i < l; i++) {
		out[i] = sl[rand() % 26];
	}
	return out;
}

int main()
{
	//test();
	/*std::vector<std::string> vocabulary = {"\n",":hi ","dad"," ",":","!","a","b","c","d","e" ,"f" ,"g" ,"h" ,"i" ,"j" ,"k" ,"l" ,"m","n","o","p","q","r","s","t","u","w","x","y","z","v"};
	
	std::string data = "szajsjem:hi dad!\ndad:hi szajsjem";
	for (int i = 0; i < 10; i++) {
		string name = rndstr(3 + rand() % 7);
		data += "!\n";
		data += name;
		data += ":hi dad!\ndad:hi ";
		data += name;
	}
	data += "!\n";*/
	std::vector<std::string> vocabulary = { "a","b","c","d" };
	std::string data = "abcdcbabcdcba";
	std::vector<int> tokenized = tokenizeString(data, vocabulary);

	const int dimmSize = 8;
	const int layers = 1;
	const int numheads = 3;
	const int vocabSize = vocabulary.size();
	const int FFmem = 4 * dimmSize;
	const int QKVmem = dimmSize;
	const int seqlen = 2;//equal to batch size

	Net model;
	model.add(new LayerEmbed(vocabSize, dimmSize, seqlen,"Uniform"));//new LayerBatchTo2D(1, dimmSize, seqlen, 
	for (int i = 0; i < layers; i++) {
		model.add(new LayerTransformerHeads(dimmSize, QKVmem, QKVmem, numheads, "Uniform", "Uniform"));
		model.add(new LayerTransformerFeedForward(dimmSize, FFmem, "Relu", "Uniform", "Uniform"));//modified version below
		/*model.add(new LayerParallel({
			new LayerActivation("Identity"),
			new LayerSequential({
				new LayerNormalize(),
				new LayerDense(dimmSize, FFmem * 2, "Uniform", "Uniform"),
				new LayerGatedActivation("Tanh","Relu"),//added gated activation for tests
				new LayerDense(FFmem, dimmSize, "Uniform", "Uniform")})
			}, SUM));*/
	}
	model.add(new LayerNormalize());//new LayerBatchTo2D(dimmSize, dimmSize, dimmSize, 
	model.add(new LayerDense(dimmSize, vocabSize, "Uniform", "Uniform"));//new LayerBatchTo2D(dimmSize, vocabSize, vocabSize, 
	model.add(new LayerSoftmax());//new LayerBatchTo2D(vocabSize, vocabSize, vocabSize, 

	model.set_classification_mode(false);

	//set the train data
	const int iSamples = tokenized.size()-1;//- seqlen
	MatrixFloat mSamples(iSamples, 1);//seqlen
	MatrixFloat mTruth(iSamples, vocabSize);//*seqlen

	for (int i = 0; i < iSamples; i++) {
		for (int s = 0; s < 1; s++) {//seqlen
			mSamples(i, s) = tokenized[i+s];
			for (int j = 0; j < vocabSize; j++) {
				if (j == (int)tokenized[i +s + 1]) mTruth(i, j + s* vocabSize) = 1.0f;
				else mTruth(i, j + s * vocabSize) = 0.0f;
			}
		}
	}

	//optimize network
	NetTrain netFit;
	netFit.set_optimizer("Step");
	//netFit.set_regularizer("GradientNormClip", 0.01);
	netFit.set_loss("AdjustedMeanSquaredError");
	netFit.set_train_data(mSamples, mTruth);
	netFit.set_validation_batchsize(seqlen);
	netFit.set_learningrate(0.02);
	netFit.set_RandomBatchOrder(false);
	netFit.set_epochs(2000);
	netFit.set_batchsize(seqlen);
	netFit.set_batchstepsize(1);





	netFit.fit(model);

	//predict and show results
	while (1) {
		MatrixFloat mOut;
		MatrixFloat mTest(2,1);
		cout << "x x -> x";
		for (int kk = 0; kk < vocabSize; kk++) {
			cout << "      " << vocabulary[kk];
		}
		cout << "  Loss: " << netFit.get_current_train_loss() << " " << netFit.get_current_train_accuracy() << " " << netFit.get_current_validation_loss() << " " << netFit.get_current_validation_accuracy()  << " " << endl;
		if (vocabSize < 10) {
			for (int i = 0; i < vocabSize; i++)
				for (int j = 0; j < vocabSize; j++) {

					mTest(0, 0) = i;
					mTest(1, 0) = j;

					model.predict(mTest, mOut);
					cout << vocabulary[i] << " " << vocabulary[j] << " -> " << vocabulary[sampleNextToken(mOut)];//,vocabSize
					for (int kk = 0; kk < vocabSize; kk++) {
						printf(" %1.4f", mOut(1, kk));
						//cout << " " << mOut(1, kk);//+vocabSize
					}
					cout << "   " << vocabulary[i] << vocabulary[j];
					for (int kk = 0; kk < 10; kk++) {
						int nt = sampleNextToken(mOut);//,vocabSize
						cout << vocabulary[nt];
						mTest(0, 0) = mTest(1, 0);
						mTest(1, 0) = nt;
						model.predict(mTest, mOut);
					}
					cout << endl;
				}
			cout << endl;
		}
		else{
			cout << data;
			for (int i = 1; i < (tokenized.size()); i++) {
				MatrixFloat mOut;
				MatrixFloat mTest(seqlen, 1);
				for (int j = i - seqlen; j < i; j++)
					if (j > 0)
						mTest(seqlen + j - i, 0) = tokenized[j];
					else
						mTest(seqlen + j - i, 0) = 0;
				model.predict(mTest, mOut);
				cout << vocabulary[sampleNextToken(mOut)];
			}
		}
		netFit.set_epochs(1);
		netFit.slowfit(model);
		//netFit.fit(model);
	}
	cout << "Test succeded." << endl;
	return 0;
}