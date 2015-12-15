#include <iostream>
#include <cmath>
#include <string>

#include <mlpack/core.hpp>

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


    //Import Training PVs(NBC) and PVs(DecisionStumps) and GTVs
        arma::mat predictionVectors_NBC[37]; //!!Column-Vector!!
        for(int trainingImageNumber=0; trainingImageNumber<37; ++trainingImageNumber){
            std::stringstream currentNBCPV;
            currentNBCPV << "../data/Predictions_" << testNumber << "/NBC_Prediction_Vectors/" << trainingImages[trainingImageNumber] << "_predictions.csv";
            mlpack::data::Load(currentNBCPV.str().c_str(),predictionVectors_NBC[trainingImageNumber],true);
          }
        arma::mat predictionVectors_DS[37]; //!!Row-Vector!!
        for(int trainingImageNumber=0; trainingImageNumber<37; ++trainingImageNumber){
            std::stringstream currentDSPV;
            currentDSPV << "../data/Predictions_" << testNumber << "/DecisionStump_Prediction_Vectors/" << trainingImages[trainingImageNumber] << "_predictions.csv";
            mlpack::data::Load(currentDSPV.str().c_str(),predictionVectors_DS[trainingImageNumber],true);
          }
        arma::mat groundTruthVectors[37]; //!!Row-Vector!!
        for(int trainingImageNumber=0; trainingImageNumber<37; ++trainingImageNumber){
            std::stringstream currentGTV;
            currentGTV << "../data/GT_vectors/" << trainingImages[trainingImageNumber] << "_GT.csv";
            mlpack::data::Load(currentGTV.str().c_str(),groundTruthVectors[trainingImageNumber],true);
          }

      //Construct NBC Confusion Matrix
        arma::mat NBC_Confusion(11,11); NBC_Confusion.fill(0.0);
        for(int trainingImageNumber=0; trainingImageNumber<37; ++trainingImageNumber){

          }


      //Construct DS Confusion Matrix














    return 0;
}

