#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

template <class T>
struct Trituple //Element of Sparse Matrix
{
    int x, y;
    T val;
    bool operator<(Trituple &other) // Use it to sort the elemnets in the order of xth row, yth column;
    {
        if (x != other.x)
        {
            return x < other.x;
        }
        else
        {
            return y < other.y;
        }
    }
};

template <typename T>
bool operator!=(complex<T> com, int w)
{
    if (com.real() == w && com.imag() == 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}

template <class T>
class SMatrix;

template <class T>
class Matrix
{
private:
    vector<vector<T>> matrix;
    int row, column;

public:
    Matrix() : row(0), column(0) { matrix.resize(0); }
    Matrix(int row, int column)
    {
        this->row = row;
        this->column = column;
        this->matrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            this->matrix[i].resize(column);
        }
    }
    // Use a 2-D vector to initialize the matrix;
    Matrix(vector<vector<T>> vec)
    {
        this->row = vec.size();
        this->column = vec[0].size();
        this->matrix = vec;
    }
    // Copy constructor;
    Matrix(Matrix const &other)
    {
        this->row = other.row;
        this->column = other.column;
        this->matrix = other.matrix;
    }
    Matrix(SMatrix<T> &other)
    {
        row = other.getRow();
        column = other.getColumn();
        matrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            this->matrix[i].resize(column);
        }
        for (Trituple<T> &t : other.getSMatrixList())
        {
            matrix[t.x][t.y] = t.val;
        }
    }
    // Override operator [] to retrieve the vector;
    vector<T> &operator[](int i)
    {
        if (i >= row)
        {
            cerr << "\033[31;1mIndex out of bound for row.\033[0m" << endl;
            abort();
        }
        return matrix[i];
    }
    int getRow()
    {
        return row;
    }
    int getColumn()
    {
        return column;
    }
    // Matrix addition
    Matrix operator+(Matrix other)
    {
        if (other.getColumn() != column || other.getRow() != row)
        {
            cerr << "\033[31;1mError, the columns and rows should be equal.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix answer(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] + other.matrix[i][j];
            }
        }
        return answer;
    }
    // Matrix substraction
    Matrix operator-(Matrix other)
    {
        return (*this + (other*(-1.0)));
    }
    // Scalar multiplication
    Matrix operator*(T other)
    {
        Matrix answer(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] * other;
            }
        }
        return answer;
    }
    friend Matrix operator*(T other1, Matrix other2)
    {
        Matrix answer(other2.row, other2.column);
        for (int i = 0; i < other2.row; i++)
        {
            for (int j = 0; j < other2.column; j++)
            {
                answer[i][j] = other2[i][j] * other1;
            }
        }
        return answer;
    }
    // Scalar division
    template <class T_other>
    Matrix operator/(T_other other)
    {
        if (other == 0)
        {
            cerr << "\033[31;1mThe divisor should not be 0.\033[0m" << endl;
            return Matrix(0, 0);
        }
        return (*this * (1.0 / other));
    }
    // Tranpostion
    Matrix transposition()
    {
        Matrix answer(column, row);
        for (int i = 0; i < column; i++)
        {
            for (int j = 0; j < row; j++)
            {
                answer[i][j] = matrix[j][i];
            }
        }
        return answer;
    }
    // Conjugation
    Matrix conjugation()
    {
        // Only applicable for complex matrix.
        Matrix answer(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = conj(matrix[i][j]);
            }
        }
        return answer;
    }
    // Element-wise multiplication
    Matrix element_wise_multiplication(Matrix other)
    {
        if (other.column != column || other.row != other.row)
        {
            cerr << "\033[31;1mError, matrixes shoud be equal-sized.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix answer(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[i][j] = matrix[i][j] * other[i][j];
            }
        }
        return answer;
    }
    // Matrix-matrix multiplication
    Matrix operator*(Matrix other)
    {
        if (column != other.row)
        {
            cerr << "\033[31;1mError, the number of columns of the first matrix/vector must euqal the number of rows of the second matrix/vector.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix answer(row, other.column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < other.column; j++)
            {
                for (int count = 0; count < column; count++)
                {
                    answer[i][j] += matrix[i][count] * other[count][j];
                }
            }
        }
        return answer;
    }
    // Max
    T max() // max of all matrix
    {
        T max = matrix[0][0];
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (matrix[i][j] > max)
                {
                    max = matrix[i][j];
                }
            }
        }
        return max;
    }
    T max(int col_or_row, bool iscol = false)
    {
        T max;
        if (iscol) // Max a column
        {
            if (col_or_row >= column)
            {
                cerr << "\033[31;1mInput column is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            max = matrix[0][col_or_row];
            for (int i = 0; i < row; i++)
            {
                if (matrix[i][col_or_row] > max)
                {
                    max = matrix[i][col_or_row];
                }
            }
        }
        else // Max a row
        {
            if (col_or_row >= row)
            {
                cerr << "\033[31;1mInput row is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            max = matrix[col_or_row][0];
            for (int i = 0; i < column; i++)
            {
                if (matrix[col_or_row][i] > max)
                {
                    max = matrix[col_or_row][i];
                }
            }
        }
        return max;
    }
    // Min
    T min() // min of all matrix
    {
        T min = matrix[0][0];
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (matrix[i][j] < min)
                {
                    min = matrix[i][j];
                }
            }
        }
        return min;
    }
    T min(int col_or_row, bool iscol = false)
    {
        T min;
        if (iscol) // Min a column
        {
            if (col_or_row >= column)
            {
                cerr << "\033[31;1mInput column is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            min = matrix[0][col_or_row];
            for (int i = 0; i < row; i++)
            {
                if (matrix[i][col_or_row] < min)
                {
                    min = matrix[i][col_or_row];
                }
            }
        }
        else // Min a row
        {
            if (col_or_row >= row)
            {
                cerr << "\033[31;1mInput row is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            min = matrix[col_or_row][0];
            for (int i = 0; i < column; i++)
            {
                if (matrix[col_or_row][i] < min)
                {
                    min = matrix[col_or_row][i];
                }
            }
        }
        return min;
    }
    // Sum
    T sum() // Sum all items.
    {
        T answer = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer += matrix[i][j];
            }
        }
        return answer;
    }
    T sum(int col_or_row, bool iscol = false)
    {
        T answer = 0;
        if (iscol) // Sum a column
        {
            if (col_or_row >= column)
            {
                cerr << "\033[31;1mInput column is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            for (int i = 0; i < row; i++)
            {
                answer += matrix[i][col_or_row];
            }
        }
        else // sum a row
        {
            if (col_or_row >= row)
            {
                cerr << "\033[31;1mInput row is greater than the size of the matrix.\033[0m" << endl;
                return NAN;
            }
            for (int i = 0; i < column; i++)
            {
                answer += matrix[col_or_row][i];
            }
        }
        return answer;
    }
    // Avg
    T average() // Average all.
    {
        return max() / (row * column);
    }
    T average(int col_or_rol, bool iscol = false)
    {
        return iscol ? sum(col_or_rol, iscol) / row : sum(col_or_rol, iscol) / column;
    }
    // Eigenvalue
    Matrix eigenvalue()
    {
        if (row != column)
        {
            cerr << "\033[31;1mInput matrix must be a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Mat myMat = Matrix2Mat(CV_64F);
        Mat eValuesMat;
        Mat eVectorsMat;
        eigen(myMat, eValuesMat, eVectorsMat);
        return Matrix(eValuesMat);
    }
    // Eigenvector
    Matrix eigenvector()
    {
        if (row != column)
        {
            cerr << "\033[31;1mInput matrix must be a square matrix.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Mat myMat = Matrix2Mat(CV_64F);
        Mat eValuesMat;
        Mat eVectorsMat;
        eigen(myMat, eValuesMat, eVectorsMat);
        return Matrix(eVectorsMat);
    }
    // Traces
    T trace()
    {
        if (column != row)
        {
            cerr << "\033[31;1mThe input matrix should be a square matrix.\033[0m" << endl;
            return NAN;
        }
        T answer = 0;
        for (int i = 0; i < column; i++)
        {
            answer += matrix[i][i];
        }
        return answer;
    }
    // Inverse
    Matrix inverse()
    {
        if (column != row)
        {
            cerr << "\033[31;1mThis matrix is irreversible (Not a square matrix).\033[0m" << endl;
            return Matrix(0, 0);
        }
        T deter = det();
        if (deter == 0)
        {
            cerr << "\033[31;1mThis matrix is irreversible (determinant equals zero).\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix answer(column, column);
        for (int i = 0; i < column; i++)
        {
            for (int j = 0; j < column; j++)
            {
                answer[j][i] = pow(-1, i + j) * subMatrix(i, j).det() / deter;
            }
        }
        return answer;
    }
    // Determinant
    T det()
    {
        T sum = 0;
        if (row != column)
        {
            cout << "\033[31;1mThe input matrix should be a square matrix.\033[0m" << endl;
            return NAN;
        }
        if (column == 1)
        {
            return matrix[0][0];
        }
        else
        {
            for (int i = 0; i < column; i++)
            {
                Matrix subMat = subMatrix(0, i);
                sum += matrix[0][i] * pow(-1, i) * subMat.det();
            }
        }
        return sum;
    }
    Matrix subMatrix(int m, int n) // Use it to calculate determinant. (m,n)的代数余子式矩阵
    {
        Matrix answer(column - 1, column - 1);
        for (int i = 0; i < column - 1; i++)
        {
            for (int j = 0; j < column - 1; j++)
            {
                if (i < m)
                {
                    if (j < n)
                    {
                        answer[i][j] = matrix[i][j];
                    }
                    else
                    {
                        answer[i][j] = matrix[i][j + 1];
                    }
                }
                else
                {
                    if (j < n)
                    {
                        answer[i][j] = matrix[i + 1][j];
                    }
                    else
                    {
                        answer[i][j] = matrix[i + 1][j + 1];
                    }
                }
            }
        }
        return answer;
    }
    // Reshape
    // reshape(m,n)指定m行n列; reshape(m)指定m行; reshape(m,true)指定m列。
    Matrix reshape(int m, int n)
    {
        // Reshape the matrix;
        if (m * n != row * column)
        {
            cerr << "\033[31;1mError, the size of the matrix should not change.\033[0m" << endl;
            return Matrix(0, 0);
        }
        Matrix answer(m, n);
        int count = 0;
        while (count < row * column)
        {
            int j = count / row;
            int i = count % row;
            int ansColumn = count / m;
            int ansRow = count % m;
            answer[ansRow][ansColumn] = matrix[i][j];
            count++;
        }
        return answer;
    }
    Matrix reshape(int col_or_row, bool iscol = false)
    {
        int m, n;
        if (iscol == false)
        {
            m = col_or_row;
            if (row * column % m == 0)
            {
                n = row * column / m;
            }
            else
            {
                cerr << "\033[31;1mError, the size of the matrix should not change.\033[0m" << endl;
                return Matrix(0, 0);
            }
        }
        else
        {
            n = col_or_row;
            if (row * column % n == 0)
            {
                m = row * column / n;
            }
            else
            {
                cerr << "\033[31;1mError, the size of the matrix should not change.\033[0m" << endl;
                return Matrix(0, 0);
            }
        }
        // Reshape the matrix;
        Matrix answer(m, n);
        int count = 0;
        while (count < row * column)
        {
            int j = count / row;
            int i = count % row;
            int ansColumn = count / m;
            int ansRow = count % m;
            answer[ansRow][ansColumn] = matrix[i][j];
            count++;
        }
        return answer;
    }
    // Slice 左闭右闭
    Matrix slice(int start_row, int start_col, int end_row, int end_col)
    {
        if (start_row < 0 || start_col < 0 || start_row >= row || start_col >= column || end_row < 0 || end_col < 0 || end_row >= row || end_col >= column)
        {
            cerr << "\033[31;1mError, invalid input (index out of bound).\033[0m" << endl;
            return Matrix(0, 0);
        }
        int x1, y1, x2, y2;
        if (start_row > end_row)
        {
            x1 = end_row;
            x2 = start_row;
        }
        else
        {
            x2 = end_row;
            x1 = start_row;
        }
        if (start_col > end_col)
        {
            y1 = end_col;
            y2 = start_col;
        }
        else
        {
            y2 = end_col;
            y1 = start_col;
        }
        Matrix answer(x2 - x1 + 1, y2 - y1 + 1);
        for (int i = 0; i < answer.row; i++)
        {
            for (int j = 0; j < answer.column; j++)
            {
                answer[i][j] = matrix[x1 + i][y1 + j];
            }
        }
        return answer;
    }

    // Convolution
    // Reference for convolution: https://blog.csdn.net/qq_32846595/article/details/79053277
    // Use zeros to complete the origin matrix
    Matrix convolution(Matrix kernel)
    {
        kernel = kernel.transposition().transposition(); // 翻转180度
        // center
        int x = kernel.row / 2;
        int y = kernel.column / 2;
        Matrix ans = Matrix(row, column);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                for (int m = 0; m < kernel.row; m++) // kernel rows
                {
                    for (int n = 0; n < kernel.column; n++) // kernel columns
                    {
                        int ii = i + (m - x);
                        int jj = j + (n - y);
                        // ignore input samples which are out of bound
                        if (ii >= 0 && ii < row && jj >= 0 && jj < column)
                        {
                            ans[i][j] += matrix[ii][jj] * kernel[m][n];
                            // cout << i <<" " <<j <<" "<< matrix[ii][jj] << " " << kernel[m][n] << endl;
                        }
                    }
                }
            }
        }
        return ans;
    }
    // Opencv mat to Matrix
    /*
    CV的宏定义 
    #define CV_8U   0
    #define CV_8S   1
    #define CV_16U  2
    #define CV_16S  3
    #define CV_32S  4
    #define CV_32F  5
    #define CV_64F  6
    #define CV_16F  7 */
    Matrix(cv::Mat other) //For simplicity, only consider the case other.channels() == 1.
    {
        row = other.rows;
        column = other.cols * other.channels(); //Opencv mat element could have several channels. For example, RGB represents 3 channels;
        this->matrix.resize(row);
        for (int i = 0; i < row; i++)
        {
            this->matrix[i].resize(column);
        }
        for (int i = 0; i < other.rows; i++) // cv row
        {
            for (int j = 0; j < other.cols; j++) // cv col
            {
                for (int k = 0; k < other.channels(); k++) //cv channel
                {
                    switch (other.type() % 8)
                    {
                    case 0:
                        matrix[i][j + k] = other.at<uchar>(i, j);
                        break;
                    case 1:
                        matrix[i][j + k] = other.at<char>(i, j);
                        break;
                    case 2:
                        matrix[i][j + k] = other.at<ushort>(i, j);
                        break;
                    case 3:
                        matrix[i][j + k] = other.at<short>(i, j);
                        break;
                    case 4:
                        matrix[i][j + k] = other.at<int>(i, j);
                        break;
                    case 5:
                        matrix[i][j + k] = other.at<float>(i, j);
                        break;
                    case 6:
                        matrix[i][j + k] = other.at<double>(i, j);
                        break;
                    case 7:
                        matrix[i][j + k] = other.at<double>(i, j);
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }
    // Matrix to mat
    Mat Matrix2Mat(int type = 0) //For simplicity, only consider ans.channels() == 1
    {
        Mat answer = Mat::zeros(row, column, type);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                switch (type)
                {
                case 0:
                    answer.at<uchar>(i, j) = matrix[i][j];
                    break;
                case 1:
                    answer.at<char>(i, j) = matrix[i][j];
                case 2:
                    answer.at<ushort>(i, j) = matrix[i][j];
                    break;
                case 3:
                    answer.at<short>(i, j) = matrix[i][j];
                case 4:
                    answer.at<int>(i, j) = matrix[i][j];
                case 5:
                    answer.at<float>(i, j) = matrix[i][j];
                case 6:
                    answer.at<double>(i, j) = matrix[i][j];
                case 7:
                    answer.at<double>(i, j) = matrix[i][j];
                default:
                    break;
                }
            }
        }
        return answer;
    }
    // Override << to show the matrix;
    friend ostream &operator<<(ostream &os, Matrix other)
    {
        os << "[";
        for (int i = 0; i < other.row; i++)
        {
            if (i != 0)
            {
                os << " ";
            }
            for (int j = 0; j < other.column; j++)
            {
                os << other.matrix[i][j];
                if (j != other.column - 1)
                {
                    os << ", ";
                }
            }
            if (i != other.row - 1)
            {
                os << endl;
            }
        }
        os << "]";
        return os;
    }
};

template <class T>
class SMatrix
{
private:
    int row, column;
    int terms, maxTerms;
    vector<Trituple<T>> smatrix;

public:
    SMatrix() : row(0), column(0), terms(0), maxTerms(0) {}
    SMatrix(int row = 0, int column = 0) : terms(0), maxTerms(0)
    {
        this->row = row;
        this->column = column;
        this->maxTerms = row * column;
    }
    // 使用自己定义的Matirx声明
    SMatrix(Matrix<T> &other) : terms(0)
    {
        row = other.getRow();
        column = other.getColumn();
        maxTerms = row * column;
        for (int i = 0; i < other.getRow(); i++)
        {
            for (int j = 0; j < other.getColumn(); j++)
            {
                if (other[i][j] != 0)
                {
                    smatrix.push_back(Trituple<T>{i, j, other[i][j]});
                    terms++;
                }
            }
        }
    }
    vector<Trituple<T>> getSMatrixList()
    {
        return smatrix;
    }
    int getRow()
    {
        return row;
    }
    int getColumn()
    {
        return column;
    }
    int getTerms()
    {
        return terms;
    }
    bool insert(int x, int y, T val) // 插入元素
    {
        Trituple<T> other;
        other.x = x;
        other.y = y;
        other.val = val;
        return insert(other);
    }
    bool insert(Trituple<T> other) // 插入元素
    {
        if (other.x < 0 || other.x >= row || other.y < 0 || other.y >= column || terms >= maxTerms)
        {
            return false;
        }
        for (Trituple<T> &t : smatrix) // 去除重复元素
        {
            if (t.x == other.x && t.y == other.y)
            {
                t.val = other.val;
                return true;
            }
        }
        terms++;
        smatrix.push_back(other);
        sort(smatrix.begin(), smatrix.end());
        return true;
    }
    SMatrix operator+(SMatrix other)
    {
        if (row != other.row || column != other.column)
        {
            cerr << "\033[31;1mError, the columns and rows should be equal.\033[0m" << endl;
            return SMatrix(0, 0);
        }
        SMatrix answer(other.row, other.column);
        auto it1 = smatrix.begin();
        auto it2 = other.smatrix.begin();
        while (it1 < smatrix.end() || it2 < other.smatrix.end())
        {
            if (it1 < smatrix.end() && it2 < other.smatrix.end())
            {
                if (it1->x == it2->x && it1->y == it2->y)
                {
                    if(it1->val + it2->val != 0)
                    {
                        answer.smatrix.push_back(Trituple<T>{it1->x, it1->y, it1->val + it2->val});
                        answer.terms++;
                    }
                    it1++;
                    it2++;
                }
                else
                {
                    if (*it1 < *it2)
                    {
                        answer.smatrix.push_back(*it1);
                        answer.terms++;
                        it1++;
                    }
                    else
                    {
                        answer.smatrix.push_back(*it2);
                        answer.terms++;
                        it2++;
                    }
                }
            }
            else
            {
                if(it1 < smatrix.end())
                {
                    answer.smatrix.push_back(*it1);
                    answer.terms++;
                    it1++;
                }
                if(it2 < other.smatrix.end())
                {
                    answer.smatrix.push_back(*it2);
                    answer.terms++;
                    it2++;
                }
            }
        }
        return answer;
    }
    friend ostream &operator<<(ostream &os, SMatrix other)
    {
        os << "INFORMATION: Row = " << other.row << ", column = " << other.column << ", number of terms = " << other.terms << ". Items are listed below: " << endl;
        if (other.terms == 0)
        {
            os << "[]";
            return os;
        }
        for (int i = 0; i < other.terms - 1; i++)
        {
            os << i + 1 << ": "
               << "SMatrix[" << other.smatrix[i].x << "][" << other.smatrix[i].y << "] = " << other.smatrix[i].val << endl;
        }
        os << other.terms << ": "
           << "SMatrix[" << other.smatrix[other.terms - 1].x << "][" << other.smatrix[other.terms - 1].y << "] = " << other.smatrix[other.terms - 1].val;
        return os;
    }
};
#endif