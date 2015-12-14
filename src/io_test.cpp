#include <iostream>
#include <vigra/multi_array.hxx>
#include <vigra/stdimage.hxx>
#include "vigra/impex.hxx"
#include "vigra/imageinfo.hxx"


int main(int argc, char ** argv) 
{
    try 
    {
        char* in_filename  = argv[1];
        char* out_filename = argv[2];
        
        // read from input file
        vigra::ImageImportInfo inputImageInfo(in_filename);
        std::cout << "Image information:" << std::endl;
        std::cout << "Pixel type:  " << inputImageInfo.getPixelType() << std::endl;

        // instantiate array for image data
        vigra::MultiArray<2, vigra::RGBValue<vigra::UInt8> > imageArrayInput(inputImageInfo.shape());
        vigra::MultiArray<2, vigra::UInt8> imageArrayOutput(inputImageInfo.shape());

        // copy image data from file into array
        importImage(inputImageInfo, imageArrayInput);

        // create output image
        for(int x=0; x<imageArrayInput.width(); x++){
        	for(int y=0; y<imageArrayInput.height(); y++){
                        imageArrayOutput[vigra::Shape2(x,y)] = ((x+y)%256);
        	}
        }

        // write to output file
        vigra::ImageExportInfo outputImageInfo(out_filename);
        exportImage(imageArrayOutput, outputImageInfo);
    }
    catch (std::exception & e) 
    {
        // catch any errors that might have occurred and print their reason
        std::cout << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
