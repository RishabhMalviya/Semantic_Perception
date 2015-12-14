#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/random.hxx>
#include <vigra/distancetransform.hxx>
#include <vigra/labelimage.hxx>
#include <vigra/seededregiongrowing.hxx>
#include <vigra/impex.hxx>
#include <vigra/transformimage.hxx>

using namespace vigra; 

int main(int argc, char ** argv)
{
    try
    {
        int number_of_points = 100;
        std::cout << "Number of superpixels?" << std::endl;
        std::cin >> number_of_points;

        ImageImportInfo inpInfo(argv[1]);
        MultiArray<2, RGBValue<UInt8> > inputImage(inpInfo.shape());
        importImage(argv[1],inputImage);

        MultiArray<2, UInt8> seeds(inpInfo.shape()), labels(inpInfo.shape());
        MultiArray<2, float> gradient(inpInfo.shape());

        MersenneTwister random;
        for(int i=1; i<=number_of_points; ++i)
        {
            // mark a number of points 
            int x = random.uniformInt(inpInfo.width());
            int y = random.uniformInt(inpInfo.height());
            
            // label each point with a unique number
            seeds(x,y) = i;
        }

        //transformImage(inputImage, gradient, VectorNormFunctor<RGBValue<UInt8> >());
        gradientBasedTransform(inputImage, gradient, RGBGradientMagnitudeFunctor<UInt8>());

        ArrayOfRegionStatistics<SeedRgDirectValueFunctor<float> > statistics(number_of_points);
        seededRegionGrowing(gradient, seeds, labels, statistics);
        
        MultiArray<2, RGBValue<UInt8> > outputImage = inputImage;
        regionImageToEdgeImage(labels, outputImage, 0);
        exportImage(outputImage, ImageExportInfo(argv[2]));
        std::cout << "Wrote segmented image" << argv[2] << std::endl;
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
