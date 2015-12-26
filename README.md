# Semantic_Perception
Semantic Perception internship assignment for CMU 2016 summer internships





## Prerequisites
You'll need to have the libraries **mlpack** and **Vigra** installed on your system: 

1. To install Vigra, follow the instructions given [Vigra installation instructions](http://ukoethe.github.io/vigra/doc-release/vigra/Installation.html) and ensure that `make install` executes properly.
2. To install mlpack, follow the instructions given [mlpack installation instructions](http://www.mlpack.org/doxygen.php?doc=build.html). Again, ensure that `make install` executed properly.


## Notes

1. The project was developed in QtCreator, which is why there is a *CMakeLists.txt.user* file in the repository.
2. All the images in the original dataset were converted into `.gif` files for ease of use with the methods in Vigra.
3. All executables are meant to be run from the *build/* folder itself. Likewise, the scripts must be run from the *scripts/* folder.
4. An explanation of the pipeline and the algorithms used can be found in the [docs/explanation.md](https://github.com/RishabhMalviya/Semantic_Perception/blob/master/Write-Up/explanation.md) file. It is nearly identical to the text in the [docs/CMU_Assignment_final.pdf](https://github.com/RishabhMalviya/Semantic_Perception/blob/master/Write-Up/CMU_Assignment_final.pdf) file (in fact, the pdf file is more complete, with figures as well).
