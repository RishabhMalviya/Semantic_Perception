#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/stdimagefunctions.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_watersheds.hxx>
#include <vigra/impex.hxx>

using namespace vigra; 

template <class InImage, class OutImage>
void watershedSegmentation(InImage & in, OutImage & out, double scale)
{
    MultiArray<2, float> gradient(in.shape());
    gaussianGradientMagnitude(in, gradient, scale);

    MultiArray<2, unsigned int> labeling(in.shape());
    unsigned int max_region_label = watershedsMultiArray(gradient, labeling, DirectNeighborhood, WatershedOptions().regionGrowing());

    std::cout << "Number of regions segmented: " << max_region_label << std::endl;
    out = in;
    regionImageToEdgeImage(labeling, out,
                           NumericTraits<typename OutImage::value_type>::zero());
}


int main(int argc, char ** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " infile outfile" << std::endl;
        std::cout << "(supported formats: " << impexListFormats() << ")" << std::endl;

        return 1;
    }

    try
    {
        ImageImportInfo info(argv[1]);

        // input width of gradient filter
        double scale = 1.0;
        std::cout << "Scale for gradient calculation ? ";
        std::cin >> scale;

        if(info.isGrayscale())
        {
            int w = info.width();
            int h = info.height();

            MultiArray<2, UInt8> in(w, h), out(w, h);
            
            importImage(info, in);

            watershedSegmentation(in, out, scale);

            std::cout << "Writing " << argv[2] << std::endl;
            exportImage(out, ImageExportInfo(argv[2]));
        }
        else
        {
            int w = info.width();
            int h = info.height();

            std::cout << std::endl << "Image Width: " << info.width() << std::endl;
            std::cout << "Image Height " << info.height() << std::endl << std::endl;

            MultiArray<2, RGBValue<UInt8> > in(w, h),  out(w, h);
            importImage(info, in);

            watershedSegmentation(in, out, scale);

            std::cout << "Writing " << argv[2] << std::endl;
            exportImage(out, ImageExportInfo(argv[2]));
        }
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
