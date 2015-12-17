#include <iostream>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <initializer_list>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <vigra/multi_array.hxx>
#include <vigra/stdimagefunctions.hxx>
#include <vigra/impex.hxx>

using namespace mlpack;
using namespace vigra;


std::size_t determineIndex(std::size_t value, arma::Row<std::size_t> rowVector){
    std::size_t index=0;
    for(unsigned int i=0; i<rowVector.n_cols; i++){
            if(rowVector(i)==value){
                    index=i;
                }
        }
    return index;
}


int main(int argc, char** argv){
    //Enumerate Training and Test Image Numbers
      static const int testNumber = std::atoi(argv[1]);
      std::string trainingImages[37];
      std::string testingImages[13];

      switch(testNumber){
          case 1 :
          {
              std::string trainingImages_[37] =                {"0080",
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
                                                                 "4260"};
             for(int i=0; i<37; ++i) trainingImages[i] = trainingImages_[i];

	     std::string testingImages_[13] =		       {"4280",
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
								"6120"};
	      for(int i=0; i<13; ++i) testingImages[i] = testingImages_[i];

	      break;
	    }
	  case 2 :
	  {
	      std::string trainingImages_[37] = {
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
	      for(int i=0; i<37; ++i) trainingImages[i] = trainingImages_[i];

              std::string testingImages_[13] = {
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
              for(int i=0; i<13; ++i) testingImages[i] = testingImages_[i];

              break;
            }
          case 3 :
          {
              std::string trainingImages_[37] = {
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
              for(int i=0; i<37; ++i) trainingImages[i] = trainingImages_[i];

              std::string testingImages_[13] = {
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
                                                               };
              for(int i=0; i<13; ++i) testingImages[i] = testingImages_[i];

              break;
            }
          case 4 :
          {
              std::string trainingImages_[37] = {
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
              for(int i=0; i<37; ++i) trainingImages[i] = trainingImages_[i];

              std::string testingImages_[13] = {
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
              for(int i=0; i<13; ++i) testingImages[i] = testingImages_[i];

              break;
            }
        }

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
        PV_file << "../data/Predictions_" << argv[1] << "/FinalPredictionVectors/" << testingImages[testImageIndex] << "_predictionVector.csv";
        arma::Mat<std::size_t> PV;
        mlpack::data::Load(PV_file.str().c_str(),PV,true); //!!Row Vector!!

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
/*
      //Create Prediction Image
        MultiArray<2,UInt8> predicted(GTImageInfo.shape()); predicted.init(0);
        for(int x=0; x<GTImage.width(); ++x){
                for(int y=0; y<GTImage.height(); ++y){
                        predicted[Shape2(x,y)] = (UInt8)PV(0,determineIndex(SLICImage(y,x),SLICLabels.row(0)));
                    }
            }
        std::stringstream predicted_file;
        predicted_file << "../data/Prediction_" << argv[1] << "/PredictedImages/" << testingImages[testImageIndex] << "_predicted.gif";
        exportImage(predicted, ImageExportInfo(predicted_file.str().c_str()));*/
      }
    std::stringstream finalConfMat_file;;
    finalConfMat_file << "../data/Predictions_" << argv[1] << "/Performance_Evaluation/confusionMatrix.csv";
    mlpack::data::Save(finalConfMat_file.str().c_str(),confusionMatrix,true);

    //Calculate other metrics (Accuracy, Precision, Recall, F1)
        //Accuracy
          double accuracy = 0.0;
          for(unsigned int i=0; i<confusionMatrix.n_cols; ++i){
              accuracy += confusionMatrix(i,i);
            }
          accuracy = (accuracy*100)/(640*512);
          std::cout << "Accuracy: " << accuracy << "%" << std::endl;

        //Precision
          arma::mat Precision(1,confusionMatrix.n_cols); Precision.fill(0.0);
          for(unsigned int i=0; i<Precision.n_cols; ++i){
              Precision(0,i) = confusionMatrix(i,i);
              double denominator = 0.0;
              for(unsigned int j=0; j<confusionMatrix.n_cols; ++j){
                  denominator += confusionMatrix(i,j);
                }
              if(denominator!=0.0) Precision(0,i) /= denominator;
              else Precision(0,i) = 0.0;
            }
          std::stringstream precision_file;
          precision_file << "../data/Predictions_" << argv[1] << "/Performance_Evaluation/Precision.csv";
          mlpack::data::Save(precision_file.str().c_str(),Precision,true);

        //Recall
          arma::mat Recall(1,confusionMatrix.n_cols); Recall.fill(0.0);
          for(unsigned int i=0; i<Recall.n_cols; ++i){
              Recall(0,i) = confusionMatrix(i,i);
              double denominator = 0.0;
              for(unsigned int j=0; j<confusionMatrix.n_cols; ++j){
                  denominator += confusionMatrix(j,i);
                }
              if(denominator!=0.0) Recall(0,i) /= denominator;
              else Recall(0,i) = 0.0;
            }
          std::stringstream recall_file;
          recall_file << "../data/Predictions_" << argv[1] << "/Performance_Evaluation/Recall.csv";
          mlpack::data::Save(recall_file.str().c_str(),Recall,true);

        //F1 Score
          arma::mat F1(1,confusionMatrix.n_cols); F1.fill(0.0);
          for(unsigned int i=0; i<F1.n_cols; ++i){
              F1(0,i) = 2*Precision(0,i)*Recall(0,i);
              double denominator = Precision(0,i) + Recall(0,i);
              if(denominator!=0.0) F1(0,i) /= denominator;
              else F1(0,i) = 0.0;
            }
          std::stringstream f1_file;
          f1_file << "../data/Predictions_" << argv[1] << "/Performance_Evaluation/F1.csv";
          mlpack::data::Save(f1_file.str().c_str(),F1,true);


    return 0;
}

