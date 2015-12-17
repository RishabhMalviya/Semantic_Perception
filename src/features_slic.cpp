#include <iostream>
#include <cmath>
#include <string>

#include <vigra/multi_array.hxx>
#include <vigra/stdimagefunctions.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_watersheds.hxx>
#include <vigra/impex.hxx>
#include <vigra/slic.hxx>
#include <vigra/colorconversions.hxx>
#include <vigra/inspectimage.hxx>
#include <vigra/symmetry.hxx>
#include <vigra/convolution.hxx>
#include <vigra/localminmax.hxx>
#include <vigra/cornerdetection.hxx>
#include <vigra/boundarytensor.hxx>
#include <vigra/noise_normalization.hxx>

#include <mlpack/core.hpp>

using namespace vigra;


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

template<class inImage>
ArrayOfRegionStatistics<FindAverageAndVariance<typename inImage::value_type>, unsigned int>
extractStatistics(inImage transformedImage, MultiArray<2,unsigned int> labels, unsigned int superpixelCount)
{
  //populate statistics array
  ArrayOfRegionStatistics<FindAverageAndVariance<typename inImage::value_type>, unsigned int> statistics_array(superpixelCount);
  inspectTwoImages(srcImageRange(transformedImage), srcImage(labels), statistics_array);

  //output visualization
  MultiArray<2, RGBValue<float> > outputImageVariances(transformedImage.shape());
  transformImage(srcImageRange(labels), destImage(outputImageVariances), statistics_array);
  //exportImage(outputImageVariances, ImageExportInfo("Variances.gif"));

  return statistics_array;
}

template <class inImage, class outImage>
void
extractSLICSuperpixels(inImage in, outImage& labels, int maxSize, double intensityScaling, unsigned int &superpixelCount)
{
  transformMultiArray(srcMultiArrayRange(in), destMultiArray(in), RGBPrime2LabFunctor<float>());

  slicSuperpixels(in, labels, intensityScaling, maxSize, SlicOptions().iterations(40).minSize(70));

  FindMinMax<unsigned int> superpixelCountStatistic;
  inspectImage(labels, superpixelCountStatistic);
  superpixelCount = superpixelCountStatistic.max;
  std::cout << "Number of Superpixels: " << superpixelCount << std::endl;

}


int main(int argc, char ** argv)
{
    try
    {
      //Import image
        std::stringstream ss_left;
        ss_left << "../data/ml_task/left_gif/" << argv[1] << "_left.gif";
        ImageImportInfo inputImageInfo(ss_left.str().c_str());
        MultiArray<2, RGBValue<float> > image(inputImageInfo.shape());
        importImage(inputImageInfo, image);

      //Extracting SLIC superpixels
        MultiArray<2, unsigned int> labels(inputImageInfo.shape());
        unsigned int superpixelCount;
        extractSLICSuperpixels(image, labels, 100, 20.0, superpixelCount);

      /*FEATURE EXTRACTION*/
      //RGB Values
        ArrayOfRegionStatistics<FindAverage<RGBValue<float> >, unsigned int> RGBMeans = extractMeans(image, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<RGBValue<float> >, unsigned int> RGBStatistics = extractStatistics(image, labels, superpixelCount);
      //Vector Norm
        MultiArray<2, float> vectorNormImage(inputImageInfo.shape()); transformImage(srcImageRange(image), destImage(vectorNormImage), VectorNormFunctor<RGBValue<float> >());
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> NormMeans = extractMeans(vectorNormImage, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<float>, unsigned int> NormStatistics = extractStatistics(vectorNormImage, labels, superpixelCount);
      //Radial Symmetry
        MultiArray<2,float> symmetry(inputImageInfo.shape());
        for(double scale=2.0; scale<=8.0; scale+=2.0){
            MultiArray<2,float> temp(inputImageInfo.shape());
            radialSymmetryTransform(vectorNormImage, temp, scale);
            symmetry += temp;
         }
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> symmetryMeans = extractMeans(symmetry, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<float>, unsigned int> symmetryStatistics = extractStatistics(symmetry, labels, superpixelCount);
      //Gaussian Gradient Magnitude
        MultiArray<2,float> grad(inputImageInfo.shape()); gaussianGradientMagnitude(image, grad, 5.0);
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> gradMeans = extractMeans(grad, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<float>, unsigned int> gradStatistics = extractStatistics(grad, labels, superpixelCount);
      //Laplacian of Gaussian
        MultiArray<2, float> laplacian(inputImageInfo.shape()); laplacianOfGaussian(vectorNormImage, laplacian, 5.0);
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> laplacianMeans = extractMeans(laplacian, labels, superpixelCount);
      //Corner Detection
        MultiArray<2, float> corner(inputImageInfo.shape()); beaudetCornerDetector(vectorNormImage, corner, 5.0);
          MultiArray<2, float> mins(inputImageInfo.shape()); mins.init(0);
          MultiArray<2, float> maxs(inputImageInfo.shape()); maxs.init(0);
          localMinima(corner, mins, LocalMinmaxOptions().allowAtBorder(true).allowPlateaus(true).markWith(255));
          localMaxima(corner, maxs, LocalMinmaxOptions().allowAtBorder(true).allowPlateaus(true).markWith(255));
          corner += mins; corner += maxs;
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> cornerMeans = extractMeans(corner, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<float>, unsigned int> cornerStatistics = extractStatistics(corner, labels, superpixelCount);
      //Disparity Values
        std::stringstream ss_disparity;
        ss_disparity << "../data/ml_task/disp_gif/" << argv[1] << "_disp.gif";
        ImageImportInfo inputDisparityImageInfo(ss_disparity.str().c_str());
        MultiArray<2, float> disparity(inputDisparityImageInfo.shape());
        importImage(inputDisparityImageInfo, disparity);
        ArrayOfRegionStatistics<FindAverage<float>, unsigned int> disparityMeans = extractMeans(disparity, labels, superpixelCount);
        ArrayOfRegionStatistics<FindAverageAndVariance<float>, unsigned int> disparityStatistics = extractStatistics(disparity, labels, superpixelCount);

      /*FEATURE VECTORS*/
        arma::mat featureVectors(17,1), FV_full(17,superpixelCount);
        for(unsigned int superpixelIndex=0; superpixelIndex<superpixelCount; superpixelIndex++){
          if(RGBMeans(superpixelIndex+1).blue()>0.00001){
              arma::vec tempFV(17);
              tempFV(0) = RGBMeans(superpixelIndex+1).red();
              tempFV(1) = RGBMeans(superpixelIndex+1).green();
              tempFV(2) = RGBMeans(superpixelIndex+1).blue();
              tempFV(3) = RGBStatistics(superpixelIndex+1).red();
              tempFV(4) = RGBStatistics(superpixelIndex+1).green();
              tempFV(5) = RGBStatistics(superpixelIndex+1).blue();
              tempFV(6) = NormMeans(superpixelIndex+1);
              tempFV(7) = NormStatistics(superpixelIndex+1);
              tempFV(8) = symmetryMeans(superpixelIndex+1);
              tempFV(9) = symmetryStatistics(superpixelIndex+1);
              tempFV(10) = gradMeans(superpixelIndex+1);
              tempFV(11) = gradStatistics(superpixelIndex+1);
              tempFV(12) = laplacianMeans(superpixelIndex+1);
              tempFV(13) = cornerMeans(superpixelIndex+1);
              tempFV(14) = cornerStatistics(superpixelIndex+1);
              tempFV(15) = disparityMeans(superpixelIndex+1);
              tempFV(16) = disparityStatistics(superpixelIndex+1);

              if(featureVectors.n_cols==1) featureVectors.insert_cols(0,tempFV);
              else featureVectors.insert_cols(featureVectors.n_cols-1,tempFV);
          }
          FV_full(0,superpixelIndex) = RGBMeans(superpixelIndex+1).red();
          FV_full(1,superpixelIndex) = RGBMeans(superpixelIndex+1).green();
          FV_full(2,superpixelIndex) = RGBMeans(superpixelIndex+1).blue();
          FV_full(3,superpixelIndex) = RGBStatistics(superpixelIndex+1).red();
          FV_full(4,superpixelIndex) = RGBStatistics(superpixelIndex+1).green();
          FV_full(5,superpixelIndex) = RGBStatistics(superpixelIndex+1).blue();
          FV_full(6,superpixelIndex) = NormMeans(superpixelIndex+1);
          FV_full(7,superpixelIndex) = NormStatistics(superpixelIndex+1);
          FV_full(8,superpixelIndex) = symmetryMeans(superpixelIndex+1);
          FV_full(9,superpixelIndex) = symmetryStatistics(superpixelIndex+1);
          FV_full(10,superpixelIndex) = gradMeans(superpixelIndex+1);
          FV_full(11,superpixelIndex) = gradStatistics(superpixelIndex+1);
          FV_full(12,superpixelIndex) = laplacianMeans(superpixelIndex+1);
          FV_full(13,superpixelIndex) = cornerMeans(superpixelIndex+1);
          FV_full(14,superpixelIndex) = cornerStatistics(superpixelIndex+1);
          FV_full(15,superpixelIndex) = disparityMeans(superpixelIndex+1);
          FV_full(16,superpixelIndex) = disparityStatistics(superpixelIndex+1);
        }

        featureVectors.shed_col(featureVectors.n_cols-1);

        std::stringstream ss_fv, ss_fv_full;
        ss_fv << "../data/feature_vectors/" << argv[1] << "_fv.csv";
        ss_fv_full << "../data/feature_vectors_full/" << argv[1] << "_fv_full.csv";
        mlpack::data::Save(ss_fv.str().c_str(),featureVectors,true);
        mlpack::data::Save(ss_fv_full.str().c_str(),FV_full,true);
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
