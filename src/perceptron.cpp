#include <iostream>
#include <cmath>
#include <string>

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>

using namespace mlpack;

static const char testNumber[] = "1";


int main(int argc, char** argv){
    //Enumerate Training and Test Image Numbers
        std::string trainingImages[37] = {
                                                         "0080",
                                                         "0160",
                                                         "0180",
                                                         "0200",
                                                         "0580",
                                                         "0700",
                                                         "0760",
                                                         "0800",
                                                         "0820",
                                                         "1020",
                                                         "1060",
                                                         "1080",
                                                         "1100",
                                                         "1160",
                                                         "1220",
                                                         "1320",
                                                         "1440",
                                                         "1480",
                                                         "1580",
                                                         "1600",
                                                         "1660",
                                                         "1720",
                                                         "1940",
                                                         "2280",
                                                         "2300",
                                                         "2340",
                                                         "2420",
                                                         "2580",
                                                         "2600",
                                                         "2860",
                                                         "3140",
                                                         "3220",
                                                         "3700",
                                                         "3920",
                                                         "4100",
                                                         "4200",
                                                         "4260"
                                                        };

            std::string testingImages[13] = {
                                                        "4280",
                                                        "4720",
                                                        "4740",
                                                        "4780",
                                                        "5000",
                                                        "5220",
                                                        "5440",
                                                        "5600",
                                                        "5640",
                                                        "5660",
                                                        "5900",
                                                        "6100",
                                                        "6120"
                                                   };


    //Import Training FVs and GTVs
        //trainingFVs_combined
        arma::mat trainingFVs_combined, trainingFVs[37];
        std::stringstream firstFV;
        firstFV << "../data/feature_vectors/" << trainingImages[0].c_str() << "_fv.csv";
        mlpack::data::Load(firstFV.str().c_str(),trainingFVs[0],true);
        mlpack::data::Load(firstFV.str().c_str(),trainingFVs_combined,true);
        //trainingGTVs_float (later converted to arma::Col<size_t>
        arma::mat GTV_first;
        std::stringstream firstGTV;
        firstGTV << "../data/GT_vectors/" << trainingImages[0].c_str() << "_GT.csv";
        mlpack::data::Load(firstGTV.str().c_str(),GTV_first,true);
        arma::rowvec trainingGTVs_float(GTV_first);
        //Fill in the rest
        for(int trainingImageNumber=1; trainingImageNumber<37; ++trainingImageNumber){
            //FV (arma::mat)
            std::stringstream currentFV;
            currentFV << "../data/feature_vectors/" << trainingImages[trainingImageNumber].c_str() << "_fv.csv";
            mlpack::data::Load(currentFV.str().c_str(),trainingFVs[trainingImageNumber],true);
            trainingFVs_combined = join_rows(trainingFVs_combined,trainingFVs[trainingImageNumber]);
            //GTV (arma::Col<int>)
            arma::mat GTV_currentArray;
            std::stringstream currentGTV;
            currentGTV << "../data/GT_vectors/" << trainingImages[trainingImageNumber].c_str() << "_GT.csv";
            mlpack::data::Load(currentGTV.str().c_str(),GTV_currentArray,true);
            arma::rowvec GTV_currentCol(GTV_currentArray);
            trainingGTVs_float = join_rows(trainingGTVs_float,GTV_currentCol);
          }
        arma::Row<std::size_t> trainingGTVs(trainingGTVs_float.n_cols);
        for(int FinalGTVIndex=0;FinalGTVIndex<trainingGTVs.n_cols;++FinalGTVIndex){
            trainingGTVs(FinalGTVIndex) = (std::size_t)trainingGTVs_float(FinalGTVIndex);
          }

    //Import Testing FVs
        arma::mat testingFVs[13];
        for(int testingImageNumber=0; testingImageNumber<13;++testingImageNumber){
            std::stringstream currentFV;
            currentFV << "../data/feature_vectors/" << testingImages[testingImageNumber].c_str() << "_fv.csv";
            mlpack::data::Load(currentFV.str().c_str(),testingFVs[testingImageNumber],true);
          }


    //Train NBC
        perceptron::Perceptron<> perceptron(trainingFVs_combined,trainingGTVs,50);

    //Predict and Export Vectors
       for(int trainingIndex=0; trainingIndex<37; ++trainingIndex){
            arma::Row<std::size_t> predictionOutputRow(trainingFVs[trainingIndex].n_cols);
            perceptron.Classify(trainingFVs[trainingIndex],predictionOutputRow);
            std::stringstream prediction;
            prediction << "../data/Predictions_" << testNumber << "/Perceptron_Prediction_Vectors/" << trainingImages[trainingIndex] << "_predictions.csv";
            arma::mat predictionOutput(1,predictionOutputRow.n_cols);
            for(int i=0; i<predictionOutputRow.n_cols; ++i){
                predictionOutput(0,i) = predictionOutputRow(i);
              }
            mlpack::data::Save(prediction.str().c_str(),predictionOutput,true);
          }
        for(int testingIndex=0; testingIndex<13; ++testingIndex){
            arma::Row<std::size_t> predictionOutputRow(testingFVs[testingIndex].n_cols);
            perceptron.Classify(testingFVs[testingIndex],predictionOutputRow);
            std::stringstream prediction;
            prediction << "../data/Predictions_" << testNumber << "/Perceptron_Prediction_Vectors/" << testingImages[testingIndex] << "_predictions.csv";
            arma::mat predictionOutput(1,predictionOutputRow.n_cols);
            for(int i=0; i<predictionOutputRow.n_cols; ++i){
                predictionOutput(0,i) = predictionOutputRow(i);
              }
            mlpack::data::Save(prediction.str().c_str(),predictionOutput,true);
          }

    return 0;
}
