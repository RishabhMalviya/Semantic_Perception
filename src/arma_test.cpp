#include <mlpack/core.hpp>

int main(int argc, char** argv){
  arma::mat featureVectors = arma::randu<arma::mat>(18,100);
  featureVectors.resize(18,featureVectors.n_cols+1);

  arma::vec a(18); a.fill(32.1);
  featureVectors.insert_cols(featureVectors.n_cols,a);
   
  //arma::mat featureVectorsFinal = featureVectors.submat(arma::span(0,16),arma::span(0,99));

  mlpack::data::Save("../arma_test.csv",featureVectors,true);

  return 0;
}
