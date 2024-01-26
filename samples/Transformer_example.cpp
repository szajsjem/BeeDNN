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


int main()
{

	std::vector<std::string> vocabulary = {"a","b","c","d"};
	std::string data = "abcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcbabcdcba";
	std::vector<int> tokenized = tokenizeString(data, vocabulary);

	const int dimmSize = 8;
	const int layers = 8;
	const int numheads = 8;
	const int vocabSize = vocabulary.size();
	const int seqlen = 2;//equal to batch size

	Net model;
	model.add(new LayerEmbed(vocabSize, dimmSize, seqlen,"Uniform"));
	for (int i = 0; i < layers; i++) {
		model.add(new LayerTransformerHeads(dimmSize, dimmSize / numheads, dimmSize / numheads, numheads));
		model.add(new LayerTransformerFeedForward(dimmSize, 4 * dimmSize, "Relu"));
	}
	model.add(new LayerNormalize());
	model.add(new LayerDense(dimmSize, vocabSize));
	model.add(new LayerSoftmax());

	model.set_classification_mode(false);

	//set the train data
	const int iSamples = tokenized.size()-1;
	MatrixFloat mSamples(iSamples, 1);
	MatrixFloat mTruth(iSamples, vocabSize);

	for (int i = 0; i < iSamples; i++) {
		mSamples(i, 0) = tokenized[i];
		for (int j = 0; j < vocabSize; j++) {
			if(j == (int)tokenized[i+1]) mTruth(i,j) = 1.0f;
			else mTruth(i, j) = 0.0f;
		}
	}

	//optimize network
	NetTrain netFit;
	netFit.set_epochs(10);
	netFit.set_batchsize(seqlen);
	netFit.set_validation_batchsize(seqlen);
	netFit.set_batchstepsize(1);
	//todo add keep_data_order=true, keep_data_no_hole=1
	netFit.set_train_data(mSamples, mTruth);

	//predict and show results
	while (1) {
		netFit.fit(model);
		MatrixFloat mOut;
		MatrixFloat mTest(seqlen,1);
		cout << "x x -> x";
		for (int kk = 0; kk < vocabSize; kk++) {
			cout << " " << vocabulary[kk];
		}
		cout << "  Loss: " << netFit.get_current_train_loss() << " " << netFit.get_current_train_accuracy() << " " << netFit.get_current_validation_loss() << " " << netFit.get_current_validation_accuracy()  << " " << endl;
		for (int i = 0; i < vocabSize; i++)
			for (int j = 0; j < vocabSize; j++) {

				mTest(0, 0) = i;
				mTest(1, 0) = j;

				model.predict(mTest, mOut);
				cout << vocabulary[i] << " " << vocabulary[j] << " -> " << vocabulary[sampleNextToken(mOut)];
				for (int kk = 0; kk < vocabSize; kk++) {
					cout << " " << mOut(1, kk);
				}
				cout << "   " << vocabulary[i] << vocabulary[j];
				for (int kk = 0; kk < 10; kk++) {
					int nt = sampleNextToken(mOut);
					cout << vocabulary[nt];
					mTest(0, 0) = mTest(1, 0);
					mTest(1, 0) = nt;
					model.predict(mTest, mOut);
				}
				cout << endl;
			}
		cout << endl;
	}
	cout << "Test succeded." << endl;
	return 0;
}