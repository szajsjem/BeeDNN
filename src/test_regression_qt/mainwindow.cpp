#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include "SimpleCurve.h"

#include "DNNEngine.h"
#include "DNNEngineTestDnn.h"
#include "DNNEngineTinyDnn.h"

#include "LayerActivation.h"

/*
//////////////////////////////////////////////////////////////////////////
// callback class to observe loss evolution
class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)
    {
        vdLoss.push_back(tr.loss);
        vdMaxError.push_back(tr.maxError);
    }

    vector<double> vdLoss,vdMaxError;
};
//////////////////////////////////////////////////////////////////////////
*/
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    vector<string> vsActivations;

    list_activations_available( vsActivations);
    /*
    for(unsigned int i=0;i<vsActivations.size();i++)
    {
        ui->cbActivationLayer1->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer2->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer3->addItem(vsActivations[i].c_str());
    }
    ui->cbActivationLayer1->setCurrentText("Tanh");
    ui->cbActivationLayer2->setCurrentText("Tanh");
    ui->cbActivationLayer3->setCurrentText("Linear");

    ui->cbFunction->addItem("Sin");
    ui->cbFunction->addItem("Abs");
    ui->cbFunction->addItem("Parabolic");
    ui->cbFunction->addItem("Gamma");
    ui->cbFunction->addItem("Exp");
    ui->cbFunction->addItem("Sqrt");
    ui->cbFunction->addItem("Ln");
    ui->cbFunction->addItem("Gauss");
    ui->cbFunction->addItem("Inverse");
    ui->cbFunction->addItem("Rectangular");
*/
    QStringList qsl;
    qsl+="LayerType";
    qsl+="InSize";
    qsl+="OutSize";

    ui->twNetwork->setHorizontalHeaderLabels(qsl);

    for(int i=0;i<10;i++)
    {
        QComboBox*  qcbType=new QComboBox;
        qcbType->addItem("");
        qcbType->addItem("DenseAndBias");
        qcbType->addItem("DenseNoBias");

        qcbType->insertSeparator(3);

        for(unsigned int a=0;a<vsActivations.size();a++)
            qcbType->addItem(vsActivations[a].c_str());

        ui->twNetwork->setCellWidget(i,0,qcbType);
    }

    resizeDocks({ui->dockWidget},{1},Qt::Horizontal);

    _pEngine=new DNNEngineTestDnn;
}
//////////////////////////////////////////////////////////////////////////
MainWindow::~MainWindow()
{
    delete ui;
    delete _pEngine;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_clicked()
{
    train_and_test(true);
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::train_and_test(bool bReset)
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    //LossObserver lossCB;

    if(bReset)
        parse_net();

    int iNbPoint=ui->leNbPointsLearn->text().toInt();
    float dInputMin=ui->leInputMin->text().toFloat();
    float dInputMax=ui->leInputMax->text().toFloat();
    float dStep=(dInputMax-dInputMin)/(iNbPoint-1.f);

    //create ref sample
    MatrixFloat mTruth(iNbPoint,1);
    MatrixFloat mSamples(iNbPoint,1);
    float dVal=dInputMin;

    for( int i=0;i<iNbPoint;i++)
    {
        mTruth(i,0)=compute_truth(dVal);
        mSamples(i,0)=dVal;
        dVal+=dStep;
    }

    DNNTrainOption dto;
    dto.epochs=ui->leEpochs->text().toInt();
    dto.earlyAbortMaxError=ui->leEarlyAbortMaxError->text().toDouble();
    dto.earlyAbortMeanError=ui->leEarlyAbortMeanError->text().toDouble(); //same as loss?
    dto.learningRate=ui->leLearningRate->text().toFloat();
    dto.batchSize=ui->leBatchSize->text().toInt();
    dto.momentum=ui->leMomentum->text().toFloat();
    dto.observer=0;//&lossCB;
    dto.initWeight=bReset;

    DNNTrainResult dtr =_pEngine->train(mSamples,mTruth,dto);

    ui->leMSE->setText(QString::number(dtr.loss));
    ui->leMaxError->setText(QString::number(dtr.maxError));
    ui->leComputedEpochs->setText(QString::number(dtr.computedEpochs));
    ui->leTimeByEpoch->setText(QString::number(dtr.epochDuration));

    //drawLoss(lossCB.vdLoss,lossCB.vdMaxError);
    drawRegression();
    resizeEvent(0);

    update_details();
    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<double> vdLoss,vector<double> vdMaxError)
{
    SimpleCurve* qs=new SimpleCurve;

    vector<double> x,loss,maxError;

    for(unsigned int i=0;i<vdLoss.size();i++)
    {
        x.push_back(i);
        loss.push_back(-vdLoss[i]*1000.); // //up side down *1000
        maxError.push_back(-vdMaxError[i]); //up side down
    }

    qs->addCurve(x,loss,Qt::red);
    qs->addCurve(x,maxError,Qt::black);

    ui->gvLearningCurve->setScene(qs); //take ownership
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawRegression()
{
    SimpleCurve* qs=new SimpleCurve;

    //create ref sample hi-res and net output
    unsigned int iNbPoint=(unsigned int)(ui->leNbPointsTest->text().toInt());
    float fInputMin=ui->leInputMin->text().toFloat();
    float fInputMax=ui->leInputMax->text().toFloat();
    bool bExtrapole=ui->cbExtrapole->isChecked();
    vector<double> vTruth;
    vector<double> vSamples;
    vector<double> vRegression;
    MatrixFloat mIn(1,1),mOut;

    if(bExtrapole)
    {
        float fBorder=(fInputMax-fInputMin)/2.f;
        fInputMin-=fBorder;
        fInputMax+=fBorder;
        iNbPoint*=2;
    }

    float fVal=fInputMin;
    float fStep=(fInputMax-fInputMin)/(iNbPoint-1.f);

    for(unsigned int i=0;i<iNbPoint;i++)
    {
        mIn(0,0)=fVal;
        vTruth.push_back(-compute_truth(fVal));
        vSamples.push_back(fVal);
        _pEngine->predict(mIn,mOut);
        vRegression.push_back(-mOut(0));
        fVal+=fStep;
    }

    qs->addCurve(vSamples,vTruth,Qt::red);
    qs->addCurve(vSamples,vRegression,Qt::blue);

    QPen penBlack(Qt::black);
    penBlack.setCosmetic(true);
    qs->addXAxis();
    qs->addYAxis();

    ui->gvRegression->setScene(qs); //take ownership
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionQuit_triggered()
{
    close();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_actionAbout_triggered()
{
    QMessageBox mb;
    QString qsText="Regression Net Demo";
    qsText+= "\n";
    qsText+= "\n GitHub: https://github.com/edeforas/test_DNN";
    qsText+= "\n by Etienne de Foras";
    qsText+="\n email: etienne.deforas@gmail.com";

    mb.setText(qsText);
    mb.exec();
}
//////////////////////////////////////////////////////////////////////////
float MainWindow::compute_truth(float x)
{
    //function not optimized but not mandatory

    string sFunction=ui->cbFunction->currentText().toStdString();

    if(sFunction=="Sin")
        return sinf(x);

    if(sFunction=="Abs")
        return fabs(x);

    if(sFunction=="Parabolic")
        return x*x;

    if(sFunction=="Gamma")
        return tgammaf(x);

    if(sFunction=="Exp")
        return expf(x);

    if(sFunction=="Sqrt")
        return sqrtf(x);

    if(sFunction=="Ln")
        return logf(x);

    if(sFunction=="Gauss")
        return expf(-x*x);

    if(sFunction=="Inverse")
        return 1.f/x;

    if(sFunction=="Rectangular")
        return ((((int)x)+(x<0.))+1) & 1 ;

    return 0.f;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::resizeEvent( QResizeEvent *e )
{
    (void)e;

    QGraphicsScene* qsr=ui->gvRegression->scene();
    if(qsr)
    {
        ui->gvRegression->fitInView(qsr->itemsBoundingRect());
        ui->gvRegression->scale(0.9,0.9);
    }

    QGraphicsScene* qsl=ui->gvLearningCurve->scene();
    if(qsl)
    {
        ui->gvLearningCurve->fitInView(qsl->itemsBoundingRect());
        ui->gvLearningCurve->scale(0.9,0.9);
    }
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::update_details()
{
    if(_pEngine==0)
    {
        ui->peDetails->clear();
        return;
    }

    ui->peDetails->setPlainText(_pEngine->to_string().c_str());
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_cbEngine_currentTextChanged(const QString &arg1)
{
    delete _pEngine;

    if(arg1=="testDNN")
        _pEngine=new DNNEngineTestDnn;

    if(arg1=="tiny-dnn")
        _pEngine=new DNNEngineTinyDnn;
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::on_btnTrainMore_clicked()
{
    train_and_test(false);
}
//////////////////////////////////////////////////////////////////////////////
void MainWindow::parse_net()
{
    _pEngine->clear();
    for(int iRow=0;iRow<10;iRow++) //todo dynamic size
    {
        QTableWidgetItem* pwiType=ui->twNetwork->item(iRow,0);
        if(!pwiType)
            continue;
        string sType=pwiType->text().toStdString();

        QTableWidgetItem* pwiInSize=ui->twNetwork->item(iRow,1); //todo not used in activation
        if(!pwiInSize)
            continue;
        int iInSize=pwiInSize->text().toInt();

        QTableWidgetItem* pwiOutSize=ui->twNetwork->item(iRow,2); //todo not used in activation
        if(!pwiOutSize)
            continue;
        int iOutSize=pwiOutSize->text().toInt();

        if(!sType.empty())
            _pEngine->add_layer(iInSize,iOutSize,sType);

        _pEngine->init();
    }
}
//////////////////////////////////////////////////////////////////////////////
