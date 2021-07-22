## CS205 C/ C++ Programming Project: Building a library for matrix computation.

>11813211 陈倚天
>
>11813219 吴宇闻
>
>11812214 任振裕

### Description

  Matrix is an important concept introduced in linear algebra. Matrix calculation is widely used in many practical applications, including image processing, machine learning and deep learning. In this project, we do not rely on any existing libraries, instead building a matrix calculation library, which providing serial methods like basic calculation, eigenvalues and eigenvectors, reshape and slicing, and convolution which widely used in image processing (Computer vision).

   Our work can be summarized into nine main inceptions, and the report first illustrates the methods analyses and respective codes, then comprehensive test cases are presented.  

### Usage

`rm -rf build/`

`mkdir build`

`cd build`

`cmake ..`

`make`

`./main`

`./TestCV`

### Part 1: Analysis

+ Two classes and one struct are defined:
  + One class named `Matrix` represents the dense matrix, using a 2-D `vector` to store the elements.
  
    ```c++
    template <class T>
    class Matrix
    {
    private:
        vector<vector<T>> matrix;
        int row, column;  
    };
    ```
  
  + One struct named `Trituple` represents the element of `SMatrix`, storing the indexes and values of the non-zero element for the sparse matrix.
  
    ```c++
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
    ```
  
  + One class named `SMatrix` represents the sparse matrix, using one `vector` to store the `Trituple` before.
  
    ```c++
    template <class T>
    class SMatrix
    {
        int row, column;
        int terms, maxTerms;
        vector<Trituple<T>> smatrix;
    };
    ```
  
  + All of the above use `template` to deal with all types of  data.
  
+ For class `Matrix`, all the methods required are implemented.  

+ For class `SMatrix`, we implement the method of addition, insertion, automatic sorting in the constructor and after insertion.

  ```c++
  SMatrix operator+(SMatrix other); // To achieve faster speed, directly handle the vector insted of using method insert. 
  bool insert(int x, int y, T val); // Sorting while also automatically override the existing element.
  bool insert(Trituple<T> other);
  ```

+ We provide the methods of transformations between classes `Matrix` and `SMatrix` using the constructors and between class `Matrix` and `cv::mat`.\

  All the constructors and transform methods are listed below:

  For class `Matrix`:

  ```c++
  Matrix(int row, int column);
  Matrix(vector<vector<T>> vec);
  Matrix(Matrix const &other);
  Matrix(SMatrix<T> &other); // Copy constructor using SMatrix.
  // For simplicity, only consider ans.channels() == 1 for below methods.
  Matrix(cv::Mat other); // Copy constructor using cv::mat.
  Mat Matrix2Mat(int type = 0); // Transfer matrix to cv::mat.
  ```

  For class `SMatrix`:

  ```c++
  SMatrix(int row = 0, int column = 0);
  SMatrix(Matrix<T> &other); // Copy constructor using Matrix.

+ Supported methods for class `Matrix`:
  
  + Matrix and vector arithmetic are supported. (+, -, \*, .\*, /,  $\times$, transposition, conjugation)
  
    ```c++
    Matrix operator+(Matrix other);
    Matrix operator-(Matrix other);
    Matrix operator*(T other);
    friend Matrix operator*(T other1, Matrix other2);
    Matrix operator/(T_other other);
    Matrix transposition();
    Matrix conjugation();
    Matrix element_wise_multiplication(Matrix other);
    Matrix operator*(Matrix other);
    ```
  
  + Descriptive statistics are supported. (max. min. sum, average, all supporting axis-specific and slicing.)
  
    ```c++
    T max(); // max of all matrix
    T max(int col_or_row, bool iscol = false); // axis-wise, default by row
    T min(); // min of all matrix
    T min(int col_or_row, bool iscol = false); // axis-wise, default by row
    T sum(); // Sum all items.
    T sum(int col_or_row, bool iscol = false); // axis-wise, default by row
    T average(); // Average all.
    T average(int col_or_rol, bool iscol = false); // axis-wise, default by row
    ```
  
  + Calculating eigenvalues and eigenvectors are supported.
  
    ```c++
    Matrix eigenvalue();
    Matrix eigenvector();
    ```
  
  + Calculating traces, inverse and determinant are supported.
  
    ```c++
    T trace();
    Matrix inverse();
    T det();
    Matrix subMatrix(int m, int n) // Use it to calculate determinant. (m,n)的代数余子式矩阵
    ```
  
  + Operations of reshape, slicing, and convolution are supported.
  
    ```c++
    Matrix reshape(int m, int n);
    Matrix reshape(int col_or_row, bool iscol = false); // axis-wise, default by row
    // Slice 左闭右闭
    Matrix slice(int start_row, int start_col, int end_row, int end_col);
    Matrix convolution(Matrix kernel);
    ```
  
+ For both classes, override the operator `<<` to show the matrix.
  
+ Methods all of above are well defined with comprehensive exception handling.

  ```c++
  demo:
  cerr << "\033[31;1mError, invalid input (index out of bound).\033[0m" << endl;
  ```

### Part 2: Codes

+ See [pro_code](./pro_code).

### Part 3: Testing Results

#### **Our new features are highlighted by ==example==.**  

+ (Q1 and Q2) It supports all matrix sizes, from small fixed-size matrices to arbitrarily large dense matrices, and even sparse matrices (Add: try to use efficient ways to store the sparse matrices). (10 points)

  It supports all standard numeric types, including std::complex, integers, and is easily extensible to custom numeric types. (10 points)

  + For dense matrix:

    ```
    Matrix double m1 = 
    [1.1, 2.2, 3, 4
     5, 6, 7, 8
     9, 1, 1, 2
     3, 4, 5, 6]
    Matrix int m2 = 
    [1, 1, 8, 1
     2, 2, 1, 4
     1, 1, 8, 1
     3, 2, 1, 9]
    complex matrix com_m =
    [(1,2), (3,0), (4,0)
     (5,0), (6,0), (7,8)
     (9,2), (3,6), (4,0)]
    row_vec1 = 
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    row_vec2 = 
    [9, 8, 7, 6, 5, 4, 3, 2, 1]
    col_vec1 = 
    [1
     2
     3
     4
     5
     6
     7
     8
     9]
    col_vec2 = 
    [9
     8
     7
     6
     5
     4
     3
     2
     1]
    col_vec3 = 
    [9
     8
     7
     6]
    ```

  + For sparse matrix:

    1. **==Using method `insert` to declare a `SMatrix`:(The sparse matrix is automatic sorting by the indexes of the matrix.)==**

    ```
    The spare matrix sm1 is: INFORMATION: Row = 4, column = 4, number of terms = 0. Items are listed below: 
    []
    Insert a element (1,1), with value 2 : INFORMATION: Row = 4, column = 4, number of terms = 1. Items are listed below: 
    1: SMatrix[1][1] = 2
    Update a element (1,1), with value 1 : INFORMATION: Row = 4, column = 4, number of terms = 1. Items are listed below: 
    1: SMatrix[1][1] = 1
    The order of the matrix element will not depend on the insert: INFORMATION: Row = 4, column = 4, number of terms = 4. Items are listed below: 
    1: SMatrix[0][3] = 5
    2: SMatrix[1][1] = 1
    3: SMatrix[2][1] = 7
    4: SMatrix[2][2] = 4
    sm1 = 
    INFORMATION: Row = 4, column = 4, number of terms = 5. Items are listed below: 
    1: SMatrix[0][3] = 5
    2: SMatrix[1][1] = 1
    3: SMatrix[2][1] = 7
    4: SMatrix[2][2] = 4
    5: SMatrix[3][1] = 4
    ```

    2. `Matrix` to `SMatrix`:

    ```
    Using a matrix to delcare the sparse matrix: 
    sm2 = 
    INFORMATION: Row = 4, column = 4, number of terms = 8. Items are listed below: 
    1: SMatrix[0][2] = 3
    2: SMatrix[0][3] = 4
    3: SMatrix[1][0] = 5
    4: SMatrix[1][1] = -1
    5: SMatrix[2][0] = 9
    6: SMatrix[2][3] = 2
    7: SMatrix[3][0] = 3
    8: SMatrix[3][3] = 6
    ```

    3. `SMatrix` to `Matrix`:

    ```
    SMatrix to Matrix: 
    [0, 0, 3, 4
     5, -1, 0, 0
     9, 0, 0, 2
     3, 0, 0, 6]
    ```

    4. Addition of `SMatrix`:

    ```
    Addition of sparse matrix:
    INFORMATION: Row = 4, column = 4, number of terms = 10. Items are listed below: 
    1: SMatrix[0][2] = 3
    2: SMatrix[0][3] = 9
    3: SMatrix[1][0] = 5
    4: SMatrix[2][0] = 9
    5: SMatrix[2][1] = 7
    6: SMatrix[2][2] = 4
    7: SMatrix[2][3] = 2
    8: SMatrix[3][0] = 3
    9: SMatrix[3][1] = 4
    10: SMatrix[3][3] = 6
    ```

    5. **==Compare the time of $sm_1+ sm_2$ and $m_1+m_2$:==**

    ```
    Compare the time of sm1+sm2 and m1+m2
    Under sparse representation the calculating time is 5e-06s
    Under denses representation the calculating time is 6e-06s
    ```

+ (Q3) It supports matrix and vector arithmetic, including addition, subtraction, scalar multiplication, scalar division, transposition, conjugation, element-wise multiplication, matrix-matrix multiplication, matrix-vector multiplication, dot product and cross product. (20 points)

  ```
  Matrix addition: m1 + m2 = 
  [2.1, 3.2, 11, 5
   7, 8, 8, 12
   10, 2, 9, 3
   6, 6, 6, 15]
  Matrix substraction: m1 - m2 = 
  [0.1, 1.2, -5, 3
   3, 4, 6, 4
   8, 0, -7, 1
   0, 2, 4, -3]
  Scalar multiplication: m1 * 2.0 = 
  [2.2, 4.4, 6, 8
   10, 12, 14, 16
   18, 2, 2, 4
   6, 8, 10, 12]
  Scalar division: m1 / 2.0 = 
  [0.55, 1.1, 1.5, 2
   2.5, 3, 3.5, 4
   4.5, 0.5, 0.5, 1
   1.5, 2, 2.5, 3]
  Transposition of m1 = 
  [1.1, 5, 9, 3
   2.2, 6, 1, 4
   3, 7, 1, 5
   4, 8, 2, 6]
  Transposition of row_vec1 = 
  [1
   2
   3
   4
   5
   6
   7
   8
   9]
  Conjugation of complex matrix com_m = 
  [(1,-2), (3,-0), (4,-0)
   (5,-0), (6,-0), (7,-8)
   (9,-2), (3,-6), (4,-0)]
  Element wise multiplication of m1 and m2 = 
  [1.1, 2.2, 24, 4
   10, 12, 7, 32
   9, 1, 8, 2
   9, 8, 5, 54]
  Element wise multiplication of row_vec1 and row_vec2 = 
  [9, 16, 21, 24, 25, 24, 21, 16, 9]
  Matrix-matrix multiplictaion of m1 and m2 = 
  [20.5, 16.5, 39, 48.9
   48, 40, 110, 108
   18, 16, 83, 32
   34, 28, 74, 78]
  Matrix-vector multiplictaion of m1 and col_vec3 = 
  [72.5
   190
   108
   130]
  Column-row vector multiplication between col_vec1 and row_vec1 = 
  [1, 2, 3, 4, 5, 6, 7, 8, 9
   2, 4, 6, 8, 10, 12, 14, 16, 18
   3, 6, 9, 12, 15, 18, 21, 24, 27
   4, 8, 12, 16, 20, 24, 28, 32, 36
   5, 10, 15, 20, 25, 30, 35, 40, 45
   6, 12, 18, 24, 30, 36, 42, 48, 54
   7, 14, 21, 28, 35, 42, 49, 56, 63
   8, 16, 24, 32, 40, 48, 56, 64, 72
   9, 18, 27, 36, 45, 54, 63, 72, 81]
  ```

+ (Q4) It supports basic arithmetic reduction operations, including finding the maximum value, finding the minimum value, summing all items, calculating the average value (all supporting axis-specific and all items). (10 points)

  ```
  The second parameter's default value of max,min,sum and average is false, which means the default is row.
  Max matrix m1 = 9, max row 0 of matrix m1 = 4, max column 2 of matrix m1 = 7
  Min matrix m1 = 1, min row 0 of matrix m1 = 1.1, min column 2 of matrix m1 = 1
  Sum matrix m1 = 67.3, sum row 0 of matrix m1 = 10.3, sum column 2 of matrix m1 = 16
  Average matrix m1 = 0.5625, average row 0 of matrix m1 = 2.575, average column 2 of matrix m1 = 4
  ```

+ (Q5) It supports computing eigenvalues and eigenvectors, calculating traces, computing inverse and computing determinant. (10 points)

  ```
  The matrix to test Q5 is m3: [1, 2, 3
   4, 5, 6
   7, 9, 6]
  Eigenvalues of m3 = 
  [12.6153
   0.268939
   -0.884239]
  Eigenvectors of m3 = 
  [0.293706, 0.637984, 0.711838
   0.790015, -0.581254, 0.194986
   0.538157, 0.505094, -0.674735]
  Trace of matrix m3 = 12
  Inverse of matrix m3 = 
   [-1.6, 1, -0.2
   1.2, -1, 0.4
   0.0666667, 0.333333, -0.2]
  Det of matrix m3 = 15
  ```

+ (Q6) It supports the operations of reshape and slicing. (10 points)
  
  ```
  The matrix m1 is: [1.1, 2.2, 3, 4
   5, 6, 7, 8
   9, 1, 1, 2
   3, 4, 5, 6]
  m1.reshape(8,2), to change the 4 by 4 matrix to 8 by 2: = 
  [1.1, 3
   5, 7
   9, 1
   3, 5
   2.2, 4
   6, 8
   1, 2
   4, 6]
  m1.reshape(8) = 
  [1.1, 3
   5, 7
   9, 1
   3, 5
   2.2, 4
   6, 8
   1, 2
   4, 6]
  m1.reshape(2,true) = 
  [1.1, 3
   5, 7
   9, 1
   3, 5
   2.2, 4
   6, 8
   1, 2
   4, 6]
  To slice the matrix with index slice (start_row,start_col,end_row,end_col), and the index is included. m1.slice(1,0,3,2) = 
  [5, 6, 7
   9, 1, 1
   3, 4, 5]
  ```
  
+ (Q7) It supports convolutional operations of two matrices. (10 points)
  
  ```
  mat = 
  [1, 2, 3, 4
   5, 6, 7, 8
   9, 10, 11, 12
   13, 14, 15, 16]
  ker = 
  [1, 2, 1
   0, 0, 0
   -1, -2, -1]
  mat.convolution(ker) = 
  [-16, -24, -28, -23
   -24, -32, -32, -24
   -24, -32, -32, -24
   28, 40, 44, 35]
  ```
  
+ (Q8) It supports to transfer the matrix from OpenCV to the matrix of this library and vice versa. (10 points)
  
  \# Test 1:
  
  ```
  cvmat = 
  [1, 2;
   3, 4;
   5, 6;
   7, 8]
  cvmat to matrix, Matrix<float>(cvmat) = 
  [1, 2
   3, 4
   5, 6
   7, 8]
  Matrix to cv, m1.Matrix2Mat(CV_8U) = 
  [  1,   2,   3,   4;
     5,   6,   7,   8;
     9,   1,   1,   2;
     3,   4,   5,   6]
  Run TestCV.cpp to explore more.
  ```
  
  \# ==**Test 2 (Image processing test using our defined methods `slice` and `convolution`):**==
  
  1. Test imgae `lena.jpg`:
  
     <img src="11813211+11813219+11812214+陈倚天+吴宇闻+任振裕.assets/lena.jpg" alt="lena" style="zoom: 33%;" />
  
  2. Read the image by type `CV_8U`: 
  
     `Mat img = imread("../img/lena.jpg",0);`
  
     <img src="11813211+11813219+11812214+陈倚天+吴宇闻+任振裕.assets/lena_read1channel.jpg" alt="lena_read1channel" style="zoom:33%;" />
  
     Transfer it to class `Matrix`:
  
     `Matrix<uchar> m = img;`
  
  3. Slice the image and then transfer it to `cv::mat`:
  
     `Mat img1 = m.slice(0,0,m.getRow()/2,m.getColumn()/2).Matrix2Mat(0);`
  
     <img src="11813211+11813219+11812214+陈倚天+吴宇闻+任振裕.assets/lena_slice_out.jpg" alt="lena_slice_out" style="zoom:33%;" />
  
  4. Convolution with kernel {{1,0},{2,0}} and then transfer it to `cv::mat`:
  
     `Mat img2 = m.convolution(ker).Matrix2Mat(0);`
  
     <img src="11813211+11813219+11812214+陈倚天+吴宇闻+任振裕.assets/lena_conv_out.jpg" alt="lena_conv_out" style="zoom:33%;" />
+ (Q9) It should process likely exceptions as much as possible. (10 points)
  
  ![image-20210610123546382](11813211+11813219+11812214+陈倚天+吴宇闻+任振裕.assets/image-20210610123546382.png)

### Part 4: Difficulties & Solutions

+ null

