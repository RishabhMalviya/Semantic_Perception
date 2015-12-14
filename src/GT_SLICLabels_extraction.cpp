#include <iostream>
#include <string>

#include <vigra/multi_array.hxx>
#include <vigra/stdimagefunctions.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_watersheds.hxx>
#include <vigra/impex.hxx>
#include <vigra/slic.hxx>
#include <vigra/colorconversions.hxx>

#include <mlpack/core.hpp>

#define ARMA_64BIT_WORD

using namespace vigra;


template <class inImage, class outImage>
void extractSLICSuperpixels(inImage in, outImage& labels, int maxSize, double intensityScaling, unsigned int &superpixelCount)
{
  transformMultiArray(srcMultiArrayRange(in), destMultiArray(in), RGBPrime2LabFunctor<float>());

  slicSuperpixels(in, labels, intensityScaling, maxSize, SlicOptions().iterations(40).minSize(70));

  FindMinMax<unsigned int> superpixelCountStatistic;
  inspectImage(labels, superpixelCountStatistic);
  superpixelCount = superpixelCountStatistic.max;
  std::cout << "Number of Superpixels: " << superpixelCount << std::endl;

}

template<class inImage>
ArrayOfRegionStatistics<FindAverage<typename inImage::value_type>, unsigned int>
extractMeans(inImage transformedImage, MultiArray<2,unsigned int> labels, unsigned int superpixelCount)
{
  //populate statistics array
  ArrayOfRegionStatistics<FindAverage<typename inImage::value_type>, unsigned int> means_array(superpixelCount);
  inspectTwoImages(srcImageRange(transformedImage), srcImage(labels), means_array);

  //output visualization
  MultiArray<2, RGBValue<float> > outputImageAverages(transformedImage.shape());
  transformImage(srcImageRange(labels), destImage(outputImageAverages), means_array);
  //exportImage(outputImageAverages, ImageExportInfo("Means.gif"));

  return means_array;
}


int main(int argc, char ** argv)
{
    try
    {
      //Import image
        std::stringstream ss_inp;
        ss_inp << "../data/ml_task/left_gif/" << argv[1] << "_left.gif";
        ImageImportInfo inputImageInfo(ss_inp.str().c_str());
        MultiArray<2, RGBValue<float> > image(inputImageInfo.shape());
        importImage(inputImageInfo, image);

      //Extracting SLIC superpixels
        MultiArray<2, unsigned int> labels(inputImageInfo.shape());
        unsigned int superpixelCount;
        extractSLICSuperpixels(image, labels, 100, 20.0, superpixelCount);
        //save as image
        std::stringstream segmentedImage_filename;
        segmentedImage_filename << "../data/SLICSegmentation_images/" << argv[1] << "_segmented.gif";
        exportImage(labels, ImageExportInfo(segmentedImage_filename.str().c_str()));
        //save as .csv
        std::stringstream segmentedArray_filename;
        segmentedArray_filename << "../data/SLICSegmentation_arrays/" << argv[1] << "_segmented.csv";
        arma::Mat<arma::uword> segmentedImageArray(labels.height(),labels.width());
        for(int x=0; x<labels.width(); x++){
            for(int y=0; y<labels.height(); y++){
                segmentedImageArray(y,x) = labels[Shape2(x,y)];
              }
          }
        mlpack::data::Save(segmentedArray_filename.str().c_str(),segmentedImageArray,true);


      //Import Ground Truth Image
        std::stringstream GTImage_filename;
        GTImage_filename << "../data/ml_task/labeled_gif/" << argv[1] << "_left.gif";
        ImageImportInfo GTImageInfo(GTImage_filename.str().c_str());
        MultiArray<2, int> groundTruths(GTImageInfo.shape());
        importImage(GTImageInfo,groundTruths);

      //Computing Corresponding GroundTruth Vector
        const unsigned int maximumGTLabel = 10;
        const unsigned int minimumGTLabel = 0;
        arma::mat tallyTable(superpixelCount,maximumGTLabel-minimumGTLabel+1); tallyTable.fill(0.0);
        for(int x=0; x<inputImageInfo.width(); x++){
            for(int y=0; y<inputImageInfo.height(); y++){
                tallyTable(labels[Shape2(x,y)]-1,groundTruths[Shape2(x,y)]-minimumGTLabel) += 1;
              }
          }
        arma::vec GTVector(superpixelCount);
        for(unsigned int current_SLIClabel=1;current_SLIClabel<=superpixelCount;current_SLIClabel++){
          int winner_in_label = 0;
          for(unsigned int current_groundTruthBin=minimumGTLabel; current_groundTruthBin<=maximumGTLabel; current_groundTruthBin++){
            int current_highest = 0, new_contender = 0;
            new_contender = tallyTable(current_SLIClabel-1,current_groundTruthBin-minimumGTLabel);
            if(new_contender>current_highest){
               current_highest = new_contender;
               winner_in_label = current_groundTruthBin;
              }
            }
          GTVector(current_SLIClabel-1) = winner_in_label;
          }
        //mlpack::data::Save("../mode_test.csv",GTVector,true);

      //Computing Average of Labels Vector
        ArrayOfRegionStatistics<FindAverage<unsigned int>, unsigned int> LabelMeans = extractMeans(labels, labels, superpixelCount);

      //Pruning Out nans From Averages Vector
        arma::mat prunedSLICLabelsVector(1,1);
        arma::mat prunedGTVector(1,1);
        for(int superpixelIndex=0; superpixelIndex<superpixelCount; superpixelIndex++){
          if(LabelMeans(superpixelIndex+1)>0.00001){
              //SLIC Labels Vector
              arma::vec tempLV(1);
              tempLV(0) = superpixelIndex+1; //LabelMeans(superpixelIndex+1);
              if(prunedSLICLabelsVector.n_cols==1) prunedSLICLabelsVector.insert_cols(0,tempLV);
              else prunedSLICLabelsVector.insert_cols(prunedSLICLabelsVector.n_cols-1,tempLV);
              //GT Vector Cleaning
              arma::vec tempGTV(1);
              tempGTV(0) = GTVector(superpixelIndex);
              if(prunedGTVector.n_cols==1) prunedGTVector.insert_cols(0,tempGTV);
              else prunedGTVector.insert_cols(prunedGTVector.n_cols-1,tempGTV);
            }
          }
        prunedSLICLabelsVector.shed_col(prunedSLICLabelsVector.n_cols-1);
        prunedGTVector.shed_col(prunedGTVector.n_cols-1);

        std::stringstream GTVector_filename;
        GTVector_filename << "../data/GT_vectors/" << argv[1] << "_GT.csv";
        mlpack::data::Save(GTVector_filename.str().c_str(),prunedGTVector,true);

        std::stringstream SLICLabelsVector_filename;
        SLICLabelsVector_filename << "../data/SLICLabel_vectors/" << argv[1] << "SLICLabels.csv";
        mlpack::data::Save(SLICLabelsVector_filename.str().c_str(),prunedSLICLabelsVector,true);








    }
    catch (std::exception & e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
