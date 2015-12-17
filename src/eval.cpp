#include <iostream>
#include <cmath>
#include <string>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <vigra/multi_array.hxx>
#include <vigra/stdimagefunctions.hxx>
#include <vigra/impex.hxx>

using namespace mlpack;
using namespace vigra;


std::size_t determineIndex(std::size_t value, arma::Row<std::size_t> rowVector){
    std::size_t index=0;
    for(int i=0; i<rowVector.n_cols; i++){
            if(rowVector(i)==value){
                    index=i;
                }
        }
    return index;
}


static const char testNumber[] = "4";

int main(int argc, char** argv){
    //Enumerate Training and Test Image Numbers
  std::string trainingImages[37] = {
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
                                                           "6120",
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
                                                     };

  arma::mat confusionMatrix(11,11); confusionMatrix.fill(0.0);

    for(int testImageIndex=0; testImageIndex<13; ++testImageIndex)
    {
      //Import GroundTruth IMAGES
        std::stringstream GTImage_file;
        GTImage_file << "../data/ml_task/labeled_gif/" << testingImages[testImageIndex] << "_left.gif";
        ImageImportInfo GTImageInfo(GTImage_file.str().c_str());
        MultiArray<2, unsigned int> GTImage(GTImageInfo.shape());
        importImage(GTImageInfo, GTImage);

      //Import Prediction Vectors
        std::stringstream PV_file;
        PV_file << "../data/Predictions_" << testNumber << "/FinalPredictionVectors/" << testingImages[testImageIndex] << "_predictionVector.csv";
        arma::Mat<std::size_t> PV;
        mlpack::data::Load(PV_file.str().c_str(),PV,true); //!!Row Vector!!
        mlpack::data::Save("../test.csv",PV,true);

      //Import SLIC label vectors
        std::stringstream SLICLabels_file;
        SLICLabels_file << "../data/" << "/SLICLabel_vectors/" << testingImages[testImageIndex] << "SLICLabels.csv";
        arma::Mat<std::size_t> SLICLabels;
        mlpack::data::Load(SLICLabels_file.str().c_str(),SLICLabels,true); //!!Row Vector!!

      //Import SLIC Segmentation Arrays
        std::stringstream SLICImage_file;
        SLICImage_file << "../data/SLICSegmentation_arrays/" << testingImages[testImageIndex] << "_segmented.csv";
        arma::Mat<std::size_t> SLICImage;
        mlpack::data::Load(SLICImage_file.str().c_str(),SLICImage,true);

      //Begin Calculating Confusion Matrix
        for(int x=0; x<GTImage.width(); ++x){
            for(int y=0; y<GTImage.height(); ++y){
                confusionMatrix(PV(0,determineIndex(SLICImage(y,x),SLICLabels.row(0))),GTImage[Shape2(x,y)]) += 1.0;
              }
          }

      //Create Prediction Image
        MultiArray<2,float> predicted(GTImageInfo.shape()); predicted.init(0);
        for(int x=0; x<GTImage.width(); ++x){
                for(int y=0; y<GTImage.height(); ++y){
                        predicted[Shape2(x,y)] = (float)PV(0,determineIndex(SLICImage(y,x),SLICLabels.row(0)));
                    }
            }
        //std::stringstream predicted_file;
        //predicted_file << "../data/Prediction_" << testNumber << "/PredictedImages/" << testingImages[testImageIndex] << "_predicted.gif";
        exportImage(predicted, ImageExportInfo("output.gif"));
      }
    std::stringstream finalConfMat_file;;
    finalConfMat_file << "../data/Predictions_" << testNumber << "/Performance_Evaluation/confusionMatrix.csv";
    mlpack::data::Save(finalConfMat_file.str().c_str(),confusionMatrix,true);

    //Calculate other metrics (Precision, Recall, F1)
        double correct = 0.0;
        for(int i=0; i<confusionMatrix.n_cols; ++i){
            correct += confusionMatrix(i,i);
          }
        std::cout << "Accuracy: " << correct/(640*512) << "%" << std::endl;

    return 0;
}

