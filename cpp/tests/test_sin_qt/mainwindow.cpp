#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include <QGraphicsPolygonItem>

#include "Net.h"
#include "DenseLayer.h"

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
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    vector<string> vsActivations;
    _activ.list_all(vsActivations);

    for(unsigned int i=0;i<vsActivations.size();i++)
    {
        ui->cbActivationLayer1->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer2->addItem(vsActivations[i].c_str());
        ui->cbActivationLayer3->addItem(vsActivations[i].c_str());
    }
    ui->cbActivationLayer1->setCurrentText("Gauss");
    ui->cbActivationLayer2->setCurrentText("Gauss");
    ui->cbActivationLayer3->setCurrentText("Linear");

}
//////////////////////////////////////////////////////////////////////////
MainWindow::~MainWindow()
{
    delete ui;
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::on_pushButton_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);

    LossObserver lossCB;

    Net n;

    int iNbHiddenNeurons2=ui->sbNbNeurons2->value();
    int iNbHiddenNeurons3=ui->sbNbNeurons3->value();
    Activation* pActivLayer1=_activ.get_activation(ui->cbActivationLayer1->currentText().toStdString());
    Activation* pActivLayer2=_activ.get_activation(ui->cbActivationLayer2->currentText().toStdString());
    Activation* pActivLayer3=_activ.get_activation(ui->cbActivationLayer3->currentText().toStdString());
    DenseLayer l1(1,iNbHiddenNeurons2,pActivLayer1);
    DenseLayer l2(iNbHiddenNeurons2,iNbHiddenNeurons3,pActivLayer2);
    DenseLayer l3(iNbHiddenNeurons3,1,pActivLayer3);

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    //create ref sample
    Matrix mTruth(64);
    Matrix mSamples(64);
    for( int i=0;i<64;i++)
    {
        double x=(double)i/10.;
        mTruth(i)=sin(x);
        mSamples(i)=x;
    }

    TrainOption tOpt;
    tOpt.epochs=ui->leEpochs->text().toInt();
    tOpt.earlyAbortMaxError=ui->leEarlyAbortMaxError->text().toDouble();
    tOpt.earlyAbortMeanError=ui->leEarlyAbortMeanError->text().toDouble(); //same as loss?
    tOpt.learningRate=ui->leLearningRate->text().toDouble();;
    tOpt.batchSize=ui->leBatchSize->text().toInt();
    tOpt.momentum=ui->leMomentum->text().toDouble();
    tOpt.observer=&lossCB;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);

    ui->leMSE->setText(QString::number(tr.loss));
    ui->leMaxError->setText(QString::number(tr.maxError));
    ui->leComputedEpochs->setText(QString::number(tr.computedEpochs));

    drawLoss(lossCB.vdLoss,lossCB.vdMaxError);

    QApplication::restoreOverrideCursor();
}
//////////////////////////////////////////////////////////////////////////
void MainWindow::drawLoss(vector<double> vdLoss,vector<double> vdMaxError)
{
    QGraphicsScene* qs=new QGraphicsScene;

    ui->gvLearningCurve->setScene(qs);   //gives ownership

    QPainterPath painterLoss;
    QPainterPath painterMax;
    QPainterPath painterZero;

    for(unsigned int i=0;i<vdLoss.size();i++)
    {
        painterLoss.lineTo(QPointF(i,-vdLoss[i]*1000)); //up side down
        painterMax.lineTo(QPointF(i,-vdMaxError[i])); //up side down *1000
    }

    painterZero.moveTo(QPointF(0,0));
    painterZero.lineTo(QPointF(vdLoss.size()-1,0));

    QPen penBlack(Qt::black);
    QPen penRed(Qt::red);
    QPen penBlue(Qt::blue);

    penBlack.setCosmetic(true);
    penRed.setCosmetic(true);
    penBlue.setCosmetic(true);

    qs->addPath(painterLoss,penBlue);
    qs->addPath(painterMax,penRed);
    qs->addPath(painterZero,penBlack);

    ui->gvLearningCurve->fitInView(qs->itemsBoundingRect());
    ui->gvLearningCurve->scale(0.8,0.8);

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
    QString qsText="Sin Net Demo";
    qsText+= "\n";
    qsText+= "\n GitHub: https://github.com/edeforas/test_DNN";
    qsText+= "\n by Etienne de Foras";
    qsText+="\n email: etienne.deforas@gmail.com";

    mb.setText(qsText);
    mb.exec();
}
//////////////////////////////////////////////////////////////////////////
