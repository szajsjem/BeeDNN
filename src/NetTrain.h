/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Matrix.h"

#include <vector>
#include <functional>
#include <string>

namespace beednn {

class Optimizer;
class Loss;
class Regularizer;
class Net;
class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();
	NetTrain& operator=(const NetTrain& other);

    void clear();
	
	const Net& model() const;
    void set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth);
	void set_train_data_copy(const MatrixFloat& mSamples, const MatrixFloat& mTruth);
	void set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation);

	void set_epochs(int iEpochs); //100 by default
	int get_epochs() const;

    /// reboost helps the optimizer to have a new fresh start every N epochs, improve Adamax for example
	void set_reboost_every_epochs(int iReboostEveryEpochs); //-1 by default -> disabled
	int get_reboost_every_epochs() const;

	void set_epoch_callback(std::function<void()> epochCallBack);

    void set_optimizer(const std::string& sOptimizer); //"Adam by default, ex "SGD" "Adam" "Nadam" "Nesterov" "iRPROP-" ...
    std::string get_optimizer() const;

	void set_regularizer(const std::string& sRegularizer, float fParameter=-1.f); //"None" by default , -1 is default paremeter, can be also "L2" ...
	std::string get_regularizer() const;
	float get_regularizer_parameter() const;

	void set_learningrate(float fLearningRate=-1.f ); // -1.f is for default settings
    float get_learningrate() const;

	void set_patience(int iPatience); //divide by two the learning rate if no progress during iPatience epochs , -1 is no patience
	int get_patience() const;

    void set_decay( float fDecay = -1.f); // -1.f is for default settings
    float get_decay() const;

    void set_momentum( float fMomentum = -1.f); //" -1.f is for default settings
    float get_momentum() const;

	void set_batchsize(Index iBatchSize); //32 by default
	Index get_batchsize() const;

	void set_RandomBatchOrder(bool bRandomBatchOrder); //true by default
	bool get_RandomBatchOrder() const;

	void set_batchstepsize(Index iBatchSize); //0 = batchsize by default
	Index get_batchstepsize() const;

	void set_classbalancing(bool bBalancing); //true by default //use weight loss algorithm
	bool get_classbalancing() const;

	void set_keepbest(bool bKeepBest); //true by default: keep the best model of all epochs (evaluated on the test database)
	bool get_keepbest() const;

	void set_loss(const std::string&  sLoss); // "MeanSquareError" by default, ex "MeanSquareError" "CategoricalCrossEntropy"
	void set_loss(Loss* loss){
		_pLoss=loss;
	}
	std::string get_loss() const;

	void set_validation_batchsize(Index iValBatchSize);
	Index get_validation_batchsize() const;

	void fit(Net& rNet);
	void slowfit(Net& rNet);

	float compute_loss_accuracy(const MatrixFloat & mSamples, const MatrixFloat& mTruth,float* pfAccuracy = nullptr) const;

	const std::vector<float>& get_train_loss() const;
	const std::vector<float>& get_validation_loss() const;
	const std::vector<float>& get_train_accuracy() const;
	const std::vector<float>& get_validation_accuracy() const;

	float get_current_train_loss() const;
	float get_current_train_accuracy() const;
	float get_current_validation_loss() const;
	float get_current_validation_accuracy() const;

	virtual void train_batch(const MatrixFloat& mSample, const MatrixFloat& mTruth); //all the backprop is here	

protected:
	virtual void train_one_epoch(const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled);
	void add_online_statistics(const MatrixFloat&mPredicted, const MatrixFloat&mTruth);	//online statistics, i.e. loss, accuracy ..
	Index _iBatchSize,_iBatchSizeAdjusted, _iBatchStepSize, _iBatchStepAdjusted;
	bool _bRandomBatchOrder;
	Loss* _pLoss;
	Regularizer* _pRegularizer;
	std::vector<Optimizer*> _optimizers;
	std::vector<MatrixFloat> _inOut;
	std::vector<MatrixFloat> _gradient;
	size_t _iNbLayers;

private:
	void set_net(Net& model);
	void collect_all_weights_biases();
	void update_class_weight(); // compute balanced class weight loss (if asked) and update loss
	void clear_optimizers();

	Net* _pNet;
	int _iOnlineAccuracyGood;
	float _fOnlineLoss;

	bool _bKeepBest;
	Index _iValidationBatchSize;
	int _iEpochs;
	bool _bClassBalancingWeightLoss;
	int _iReboostEveryEpochs;

    std::string _sOptimizer;
    float _fLearningRate;
	float _fDecay;
	float _fMomentum;
	int _iPatience;
	int _iCurrentPatience;

    const MatrixFloat* _pmSamplesTrain;
    const MatrixFloat* _pmTruthTrain;

	const MatrixFloat* _pmSamplesValidation;
	const MatrixFloat* _pmTruthValidation;

	std::function<void()> _epochCallBack;

    std::vector<float> _trainLoss;
    std::vector<float> _trainAccuracy;
	
	std::vector<float> _validationLoss;
	std::vector<float> _validationAccuracy;

	std::vector<MatrixFloat*> _pWeights;
	std::vector<MatrixFloat*> _pGradWeights;

	float _fTrainLoss;
	float _fTrainAccuracy;

	float _fValidationLoss;
	float _fValidationAccuracy;
};
}
