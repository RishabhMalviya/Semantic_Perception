#include <iostream>
#include <cmath>
#include <string>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;


double colSum(arma::vec input){
  double sum=0.0;
  for(int i=0; i<input.n_rows; ++i){
      sum += input(i);
    }
  return sum;
}

double rowSum(arma::rowvec input){
  double sum=0.0;
  for(int i=0; i<input.n_cols; ++i){
      sum += input(i);
    }
  return sum;
}

std::size_t maxValueIndex(arma::vec input){
  double current_highest=0;
  std::size_t maxValueIndex_=0;
  for(std::size_t i=0; i<input.n_rows; ++i){
      if(input(i)>current_highest){
          current_highest = input(i);
          maxValueIndex_ = i;
        }
    }
  return maxValueIndex_;
}

std::size_t inverseDistancePredictor_Training(arma::vec Distances, arma::Col<std::size_t> Results, unsigned int numberOfClasses){
  arma::vec probabilities(numberOfClasses);
  probabilities.fill(0.0);

  if(Distances.n_rows<=1) std::cout << "Consider more neighbours. First neighbour will be the training point itself!" << std::endl;

  double denominator = 0;
  for(int i=1; i<Distances.n_rows; ++i){
      denominator += 1.0/Distances(i);
    }

  const double self_weighting = 0.3;
  for(int i=1; i<Distances.n_rows; ++i){
      probabilities(Results(i)) += (1-self_weighting)*1.0/(Distances(i)*denominator);
    }
  probabilities(Results(0)) += self_weighting;

  return maxValueIndex(probabilities);
}

arma::vec inverseDistancePredictor_Testing(arma::vec Distances, arma::Col<std::size_t> Results, unsigned int numberOfClasses){
   arma::vec probabilities(numberOfClasses);
   probabilities.fill(0.0);

  double denominator = 0;
  for(int i=0; i<Distances.n_rows; ++i){
      denominator += 1.0/Distances(i);
    }
  for(int i=0; i<Distances.n_rows; ++i){
      probabilities(Results(i)) += 1.0/(Distances(i)*denominator);
    }

  return probabilities;
}


static const char testNumber[] = "4";
static const std::size_t TRAIN_NEIGHBORHOOD = 5;
static const std::size_t TEST_NEIGHBORHOOD = 3;

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
        arma::vec trainingGTVs_float(GTV_first.t());
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
            arma::vec GTV_currentCol(GTV_currentArray.t());
            trainingGTVs_float = join_cols(trainingGTVs_float,GTV_currentCol);
          }
        arma::Col<std::size_t> trainingGTVs(trainingGTVs_float.n_rows);
        for(int FinalGTVIndex=0;FinalGTVIndex<trainingGTVs.n_rows;++FinalGTVIndex){
            trainingGTVs(FinalGTVIndex) = (std::size_t)trainingGTVs_float(FinalGTVIndex);
          }

    //Import Testing FVs
        arma::mat testingFVs[13];
        for(int testingImageNumber=0; testingImageNumber<13;++testingImageNumber){
            std::stringstream currentFV;
            currentFV << "../data/feature_vectors/" << testingImages[testingImageNumber].c_str() << "_fv.csv";
            mlpack::data::Load(currentFV.str().c_str(),testingFVs[testingImageNumber],true);
          }


    //Create Confusion Matrices from Training Data (1st Nearest Neighbour)
        neighbor::NeighborSearch<> kNN(trainingFVs_combined,trainingFVs_combined,true,false);
        //Search!
          arma::Mat<std::size_t> Results;
          arma::mat Distances;
          kNN.Search(TRAIN_NEIGHBORHOOD,Results,Distances);
        //Make distances easier to work with
          arma::mat divider(Distances.n_rows,Distances.n_cols); divider.fill(100.0);
          Distances = Distances/divider;
        //Make the results matrix hold predicted label values
          for(int x=0; x<Results.n_cols; ++x){
              for(int y=0; y<Results.n_rows; ++y){
                  Results(y,x) = trainingGTVs(Results(y,x));
                }
            }
        //Determine final predictions for query points using all neighbours (weighted by inverse distances)
          for(int i=0; i<Results.n_cols; ++i){
              Results(0,i) = inverseDistancePredictor_Training(Distances.col(i),Results.col(i),11);
            }
          arma::Row<std::size_t> trainingResults = Results.row(0);
        //Create confusion matrix
          arma::mat confusionMatrix(11,11); confusionMatrix.fill(0.0);
          for(int i=0; i<Results.n_cols; ++i){
              confusionMatrix(trainingGTVs(i),Results(0,i)) += 1.0;
            }
        //Export Confusion Matrix
          std::stringstream confMat_file;
          confMat_file << "../data/Predictions_" << testNumber << "/intermediateConfusionMatrix.csv";
          mlpack::data::Save(confMat_file.str().c_str(),confusionMatrix,true);


      //Predict labels for each test case (5 neighbours based confidence vector)
        //First, Convert ConfMat into Conditional Probabilities Matrix
          arma::mat condProb(confusionMatrix.n_rows,confusionMatrix.n_cols);
          /*
           * condProb(i,j) will contain the probability that a
           * superpixel actually belongs to category 'i' given
           * that it has been predicted to belong to category
           * 'j' by the above nearest neighbour classifier.
           */
          for(int x=0; x<condProb.n_cols; ++x){
              for(int y=0; y<condProb.n_rows; ++y){
                  if(colSum(confusionMatrix.col(x))==0){
                      if(x==y) condProb(x,y)=1;
                      else condProb(y,x) = 0;
                    }
                  else condProb(y,x) = confusionMatrix(y,x)/colSum(confusionMatrix.col(x));
                }
            }
          std::stringstream condProb_file;
          condProb_file << "../data/Predictions_" << testNumber << "/condProb.csv";
          mlpack::data::Save(condProb_file.str().c_str(),condProb,true);

        //For each test image, calculate prediction vectors
          for(int testImageNumber=0; testImageNumber<13; ++testImageNumber){
              neighbor::NeighborSearch<> kNN(trainingFVs_combined,testingFVs[testImageNumber],true,false);
              //Search!
                arma::Mat<std::size_t> Results_;
                arma::mat Distances_;
                kNN.Search(TEST_NEIGHBORHOOD,Results_,Distances_);
                std::cout << "Nearest neighbor search for test image: " << testImageNumber+1 << " completed." << std::endl;
                std::cout << "Number of superpixels in image: " << Results_.n_cols << std::endl << std::endl;
              //Make distances easier to work with
                arma::mat divider(Distances_.n_rows,Distances_.n_cols); divider.fill(100.0);
                Distances_ = Distances_/divider;
              //Make the Results_ matrix hold predicted label values
                for(int x=0; x<Results_.n_cols; ++x){
                    for(int y=0; y<Results_.n_rows; ++y){
                        Results_(y,x) = trainingGTVs(Results_(y,x));
                      }
                  }
              //Determine Confidence Vectors for Each superpixel
                arma::mat finalConfidenceVectors(11,Results_.n_cols);
                for(int superpixel=0; superpixel<Results_.n_cols; ++superpixel){
                    arma::vec confidenceVector_prediction; confidenceVector_prediction.fill(0.0);
                    confidenceVector_prediction = inverseDistancePredictor_Testing(Distances_.col(superpixel),Results_.col(superpixel),11);
                    for(int i=0; i<11; ++i){
                        finalConfidenceVectors(i,superpixel) = dot(confidenceVector_prediction,condProb.row(i).t());
                      }
                  }
                std::stringstream confidenceVectors_view;
                confidenceVectors_view << "../data/Predictions_" << testNumber << "/FinalConfidenceVectors/" << testingImages[testImageNumber] << "_confVects.csv";
                mlpack::data::Save(confidenceVectors_view.str().c_str(),finalConfidenceVectors,true);

              //Determine final predictions from the maximums in the final confidence vectors
                arma::Mat<std::size_t> finalPredictions(1,Results_.n_cols);
                for(int superpixel=0; superpixel<Results_.n_cols; ++superpixel){
                    finalPredictions(0,superpixel) = maxValueIndex(finalConfidenceVectors.col(superpixel));
                  }
                std::stringstream predictionVector_file;
                predictionVector_file << "../data/Predictions_" << testNumber << "/FinalPredictionVectors/" << testingImages[testImageNumber] << "_predictionVector.csv";
                mlpack::data::Save(predictionVector_file.str().c_str(),finalPredictions,true);

            }













    return 0;
}
