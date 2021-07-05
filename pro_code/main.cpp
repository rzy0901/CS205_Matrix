#include <iostream>
#include <opencv2/core.hpp>
#include "Matrix.hpp"
#include <complex>
#include <ctime>

using namespace std;
using namespace cv;

int main()
{
    // Q1,Q2
    cout << "==================================Q1,Q2==================================" << endl;
    vector<vector<double>> vec1 = {{1.1, 2.2, 3, 4}, {5, 6, 7, 8}, {9, 1, 1, 2}, {3, 4, 5, 6}};
    Matrix<double> m1(vec1);
    vector<vector<double>> vec2 = {{1, 1, 8, 1}, {2, 2, 1, 4}, {1, 1, 8, 1}, {3, 2, 1, 9}};
    Matrix<double> m2(vec2);
    vector<vector<complex<double>>> com1 = {{complex<double>(1, 2), 3, 4}, {5, 6, complex<double>(7, 8)}, {complex<double>(9, 2), complex<double>(3, 6), 4}};
    Matrix<complex<double>> com_m(com1);
    vector<vector<double>> row1 = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    Matrix<double> row_vec1(row1);
    vector<vector<double>> row2 = {{9, 8, 7, 6, 5, 4, 3, 2, 1}};
    Matrix<double> row_vec2(row2);
    vector<vector<double>> col1 = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
    Matrix<double> col_vec1(col1);
    vector<vector<double>> col2 = {{9}, {8}, {7}, {6}, {5}, {4}, {3}, {2}, {1}};
    Matrix<double> col_vec2(col2);
    vector<vector<double>> col3 = {{9}, {8}, {7}, {6}};
    Matrix<double> col_vec3(col3);
    cout << "Matrix double m1 = \n"
         << m1 << endl;
    cout << "Matrix int m2 = \n"
         << m2 << endl;
    cout << "complex matrix com_m =\n"
         << com_m << endl;
    cout << "row_vec1 = \n"
         << row_vec1 << endl;
    cout << "row_vec2 = \n"
         << row_vec2 << endl;
    cout << "col_vec1 = \n"
         << col_vec1 << endl;
    cout << "col_vec2 = \n"
         << col_vec2 << endl;
    cout << "col_vec3 = \n"
         << col_vec3 << endl;


    // Q3
    cout << "==================================Q3==================================" << endl;
    // Addition
    cout << "Matrix addition: m1 + m2 = \n"
         << (m1 + m2) << endl;
    // Substraction
    cout << "Matrix substraction: m1 - m2 = \n"
         << (m1 - m2) << endl;
    // scalar multiplication
    cout << "Scalar multiplication: m1 * 2.0 = \n"
         << (m1 * 2.0) << endl;
    // cout <<"2.0 * m1 = \n" << (2.0 * m1) << endl;
    // scalar division
    cout << "Scalar division: m1 / 2.0 = \n"
         << (m1 / 2.0) << endl;
    // tranposition
    cout << "Transposition of m1 = \n"
         << m1.transposition() << endl;
    cout << "Transposition of row_vec1 = \n"
         << row_vec1.transposition() << endl;
    // conjugation
    cout << "Conjugation of complex matrix com_m = \n"
         << com_m.conjugation() << endl;
    //element-wise multiplication

    cout << "Element wise multiplication of m1 and m2 = \n"
         << m1.element_wise_multiplication(m2) << endl;
    cout << "Element wise multiplication of row_vec1 and row_vec2 = \n"
         << row_vec1.element_wise_multiplication(row_vec2) << endl;
    // matrix matrix multiplication
    cout << "Matrix-matrix multiplictaion of m1 and m2 = \n"
         << (m1 * m2) << endl;
    // matrix vector multiplication
    cout << "Matrix-vector multiplictaion of m1 and col_vec3 = \n"
         << (m1 * col_vec3) << endl;
    // column vector and row vector multiplication
    cout << "Column-row vector multiplication between col_vec1 and row_vec1 = \n"
         << (col_vec1 * row_vec1) << endl;


    // Sparse Matrix
    cout << "==================================Sparse matrix==================================" << endl;
    SMatrix<double> sm1(4, 4);
    cout << "The spare matrix sm1 is: " << sm1 << endl;
    sm1.insert(1, 1, 2);
    cout << "Insert a element (1,1), with value 2 : " << sm1 << endl;
    sm1.insert(1, 1, 1); // 去除重复
    cout << "Update a element (1,1), with value 1 : " << sm1 << endl;
    // 排序
    sm1.insert(0, 3, 5);
    sm1.insert(2, 1, 7);
    sm1.insert(2, 2, 4);
    cout << "The order of the matrix element will not depend on the insert: " << sm1 << endl;

    sm1.insert(Trituple<double>{3, 1, 4}); //也可以插入定义的Struct Trituple
    sm1.insert(100, 100, 1);               //越界插入失败，返回false
    cout << "sm1 = \n"
         << sm1 << endl;

    // SMatrix与Matrix互转
    //使用Matrix声明SMatrix
    vector<vector<double>> vecs = {{0, 0, 3, 4}, {5, -1, 0, 0}, {9, 0, 0, 2}, {3, 0, 0, 6}};
    Matrix<double> ms(vecs);
    SMatrix<double> sm2(ms);
    cout << "Using a matrix to delcare the sparse matrix: \n";
    cout << "sm2 = \n"
         << sm2 << endl;
    //使用SMatrix声明Matrix
    Matrix<double> m_from_sparse(sm2);
    cout << "SMatrix to Matrix: \n" << m_from_sparse << endl;
    // SMatrix + SMatrix
    cout << "Addition of sparse matrix:\n" << (sm1 + sm2) << endl;
    //Compare the running time.
    cout << "Compare the time of sm1+sm2 and m1+m2" << endl;
    clock_t start_time, end_time;
    start_time = clock();
    SMatrix<double> sm1_add_sm2 = sm1 + sm2;
    end_time = clock();
    double t_sp = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Under sparse representation the calculating time is " << t_sp << "s" << endl;
    clock_t start_time2, end_time2;
    start_time2 = clock();
    Matrix<double> m1_add_m2 = m1 + m2;
    end_time2 = clock();
    double t_dense = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
    cout << "Under denses representation the calculating time is " << t_dense << "s" << endl;


    // Q4
    cout << "==================================Q4==================================" << endl;
    // Max a matrix.
    cout << "The second parameter's default value of max,min,sum and average is false, which means the default is row." << endl;
    cout << "Max matrix m1 = " << m1.max() << ", max row 0 of matrix m1 = " << m1.max(0) << ", max column 2 of matrix m1 = " << m1.max(2, true) << endl;
    // Min a matrix.
    cout << "Min matrix m1 = " << m1.min() << ", min row 0 of matrix m1 = " << m1.min(0) << ", min column 2 of matrix m1 = " << m1.min(2, true) << endl;
    // Sum a matrix.
    cout << "Sum matrix m1 = " << m1.sum() << ", sum row 0 of matrix m1 = " << m1.sum(0) << ", sum column 2 of matrix m1 = " << m1.sum(2, true) << endl;
    // Average a matrix.
    cout << "Average matrix m1 = " << m1.average() << ", average row 0 of matrix m1 = " << m1.average(0) << ", average column 2 of matrix m1 = " << m1.average(2, true) << endl;


    // Q5
    cout << "==================================Q5==================================" << endl;

    vector<vector<double>> vec3 = {{1, 2, 3}, {4, 5, 6}, {7, 9, 6}};
    Matrix<double> m3(vec3);
    cout << "The matrix to test Q5 is m3: " << m3 << endl;

    //Eigen
    cout << "Eigenvalues of m3 = \n"
         << m3.eigenvalue() << endl;
    cout << "Eigenvectors of m3 = \n"
         << m3.eigenvector() << endl;
    //Trace
    cout << "Trace of matrix m3 = " << m3.trace() << endl;
    //Inverse
    cout << "Inverse of matrix m3 = \n " << m3.inverse() << endl;
    //Det
    cout << "Det of matrix m3 = " << m3.det() << endl;


    // Q6
    cout << "==================================Q6==================================" << endl;
    // reshape
    cout << "The matrix m1 is: " << m1 << endl;
    cout << "m1.reshape(8,2), to change the 4 by 4 matrix to 8 by 2: = \n"
         << m1.reshape(8, 2) << endl; // 指定8行2列
    cout << "m1.reshape(8) = \n"
         << m1.reshape(8) << endl; // 指定8行
    cout << "m1.reshape(2,true) = \n"
         << m1.reshape(2, true) << endl; // 指定2列
    //slice
    cout << "To slice the matrix with index slice (start_row,start_col,end_row,end_col), and the index is included. m1.slice(1,0,3,2) = \n"
         << m1.slice(1, 0, 3, 2) << endl; // 左闭右闭 起始坐标(1,0) 结束坐标(3,2)


    // Q7
    cout << "==================================Q7==================================" << endl;
    // conv
    vector<vector<double>> test = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    Matrix<double> mat(test);
    vector<vector<double>> vecken = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    Matrix<double> ker(vecken);
    cout << "mat = \n"
         << mat << endl;
    cout << "ker = \n"
         << ker << endl;
    cout << "mat.convolution(ker) = \n"
         << mat.convolution(ker) << endl;


    //Q8
    cout << "==================================Q8==================================" << endl;
    int temp[4][2] = {1, 2, 3, 4, 5, 6, 7, 8};
    Mat cvmat(4, 2, CV_32F, temp);
    cout << "cvmat = \n"
         << cvmat << endl;
    //cvmat to matrix
    cout << "cvmat to matrix, Matrix<float>(cvmat) = \n"
         << Matrix<float>(cvmat) << endl;
    //matrix to cv
    cout << "Matrix to cv, m1.Matrix2Mat(CV_8U) = " << endl;
    cout << m1.Matrix2Mat(CV_8U) << endl;
    cout << "Run TestCV.cpp to explore more." << endl;

    
    //Q9
    cout << "==================================Q9==================================" << endl;
    // Exceptions
    //cout << "m1[1000][1] = " << m1[1000][1] << endl;
    cout << "Some examples of the Exceptions" << endl;
    cout << "m1 + row_vec1 = ";
    cout << (m1 + row_vec1) << endl;
    cout << "m1 - row_vec1 = ";
    cout << (m1 - row_vec1) << endl;
    cout << "m1/0 = ";
    cout << (m1 / 0) << endl;
    cout << "m1.element_wise_multiplication(ker) = ";
    cout << (m1.element_wise_multiplication(ker)) << endl;
    cout << "m1*ker = ";
    cout << (m1 * ker) << endl;
    cout << "m1.max(100) = ";
    cout << m1.max(100) << endl;
    cout << "m1.min(100) = ";
    cout << m1.min(100) << endl;
    cout << "m1.sum(100) = ";
    cout << m1.sum(100) << endl;
    cout << "m1.average(100) = ";
    cout << m1.average(100) << endl;
    cout << "row_vec1.eigenvalue() = ";
    cout << row_vec1.eigenvalue() << endl;
    cout << "row_vec1.eigenvector() = ";
    cout << row_vec1.eigenvector() << endl;
    cout << "row_vec1.trace() = ";
    cout << row_vec1.trace() << endl;
    cout << "row_vec1.inverse() = ";
    cout << row_vec1.inverse() << endl;
    Matrix<double> m_det0(2, 2);
    cout << "m_det0.inverse() = ";
    cout << m_det0.inverse() << endl;
    cout << "row_vec1.det() = ";
    cout << row_vec1.det() << endl;
    cout << "m1.reshape(10000,10000) = ";
    cout << m1.reshape(10000, 10000) << endl;
    cout << "m1.reshape(10) = ";
    cout << m1.reshape(10) << endl;
    cout << "m1.slice(10,1,3,1) = ";
    cout << m1.slice(10, 1, 3, 1) << endl;

    return 0;
}