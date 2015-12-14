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


int main(int argc, char ** argv)
{
/*    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " infile outfile" << std::endl;
        std::cout << "(supported formats: " << impexListFormats() << ")" << std::endl;

        return 1;
    }
*/
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
          std::stringstream label_import_file;
          label_import_file << argv[1] << "_labels.gif";
          ImageImportInfo labelsImportInfo(label_import_file.str().c_str());
          importImage(labelsImportInfo, labels);
        //extractSLICSuperpixels(image, labels, 100, 20.0, superpixelCount);
/*        std::stringstream ss_label;
        ss_label << argv[1] << "_labels.gif";
        exportImage(labels, ImageExportInfo(ss_label.str().c_str()));
*/
/*      //Writing output for visualization
        MultiArray<2, RGBValue<float> > outputImage(inputImageInfo.shape());
        outputImage = image;
        regionImageToEdgeImage(labels,outputImage, NumericTraits<RGBValue<float> >::zero());
        std::stringstream ss_out;
        ss_out << "../data/" << argv[1] << "_outSLIC.gif";;
        exportImage(outputImage, ImageExportInfo(ss_out.str().c_str()));
*/
      //Computing Corresponding GroundTruth Vector
        const unsigned int maximumGTLabel = superpixelCount;
        const unsigned int minimumGTLabel = 1;
        const int superpixelCount_ = superpixelCount;
        //Algorithm for computing superpixel-wise modes from a transformed image (here, labels)
//        arma::mat tallyTable(superpixelCount,maximumGTLabel); tallyTable.fill(0.0);
        unsigned int tallyTable[superpixelCount_][maximumGTLabel-minimumGTLabel+1];
        for(int x=0; x<superpixelCount; x++){
            for(int y=0; y<maximumGTLabel-minimumGTLabel+1; y++){
                tallyTable[x][y]=0;
              }
          }
        for(int x=0; x<inputImageInfo.height(); x++){
            for(int y=0; y<inputImageInfo.width(); y++){
                tallyTable[labels[Shape2(x,y)]-1][labels[Shape2(x,y)]-minimumGTLabel] += 1.0;
              }
          }
        unsigned int LabelsFV[superpixelCount_];
        for(unsigned int current_label=1;current_label<=superpixelCount;current_label++){
          static int winner_in_label = 100000;
          for(unsigned int current_groundTruthBin=minimumGTLabel; current_groundTruthBin<=maximumGTLabel; current_groundTruthBin++){
            static int current_highest = 0, new_contender = 0;
            new_contender = tallyTable[current_label-1][current_groundTruthBin-minimumGTLabel];
            if(new_contender>current_highest){
               current_highest = new_contender;
               winner_in_label = current_groundTruthBin;
              }
            }
          LabelsFV[current_label-1] = winner_in_label;
          }


/*        for(int x=0; x<inputImageInfo.height(); x++){
          for(int y=0; y<inputImageInfo.width(); y++){
              tallyTable(labels[Shape2(x,y)]-1,labels[Shape2(x,y)]-minimumGTLabel) += 1.0;
            }
          }
        arma::vec LabelsFV(maximumGTLabel-minimumGTLabel+1);
        for(unsigned int current_label=1;current_label<=superpixelCount;current_label++){
          static int winner_in_label = 100000;
          for(unsigned int current_groundTruthBin=minimumGTLabel; current_groundTruthBin<=maximumGTLabel; current_groundTruthBin++){
            static int current_highest = 0, new_contender = 0;
            new_contender = tallyTable(current_label-1,current_groundTruthBin-minimumGTLabel);
            if(new_contender>current_highest){
               current_highest = new_contender;
               winner_in_label = current_groundTruthBin;
              }
            }
          LabelsFV(current_label-1) = winner_in_label;
          }*/
        //mlpack::data::Save("../mode_test.csv",LabelsFV,true);
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
