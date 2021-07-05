// opencv_demo.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.hpp"

using namespace cv;
using namespace std;
/*
1.Read lena.jpg as cv::Mat and then transfer it to class Matrix;
2.Use the methods defined in class Matrix to handel the matrix (slice or convolution or etc).
3.Transfer the resulted matrix to cv::Mat and then save it as a picture.
*/
int main(int args, char **argv)
{
cout << "OpenCV Version: " << CV_VERSION << endl;
Mat img = imread("../img/lena.jpg",0);
Matrix<uchar> m = img;
Mat img1 = m.slice(0,0,m.getRow()/2,m.getColumn()/2).Matrix2Mat(0);
vector<vector<uchar>> vecken = {{1,0},{2,0}}; 
Matrix<uchar> ker(vecken);
Mat img2 = m.convolution(ker).Matrix2Mat(0);
imwrite("../img/lena_read1channel.jpg",img);
imwrite("../img/lena_slice_out.jpg",img1);
imwrite("../img/lena_conv_out.jpg",img2);
return 0;
}
