#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp> // imread
#include <opencv2\highgui.hpp> // imshow, waitKey
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

void PrintMatrix(CvMat *Matrix, int Rows, int Cols);

int main() {
	
	/*
	Mat_<uchar> imgWrp(img);
	Mat_<uchar> smallImgWrp(img.size() / 2);
	
	for (int rowIndex = 0; rowIndex != smallImgWrp.rows; ++rowIndex)
		for (int colIndex = 0; colIndex != smallImgWrp.cols; ++colIndex)
			smallImgWrp(rowIndex, colIndex) =
			imgWrp(rowIndex * 2, colIndex * 2);
	
	Mat result(img.rows + smallImgWrp.rows, img.cols, CV_8U, Scalar(0));
	imgWrp.copyTo(result(Rect(0, 0, imgWrp.cols, imgWrp.rows)));
	
	smallImgWrp.copyTo(
		result(Rect(0, imgWrp.rows, smallImgWrp.cols, smallImgWrp.rows)));
	*/


	Mat img_a = imread("pic1.bmp", IMREAD_GRAYSCALE);
	Mat img_b = imread("pic2.bmp", IMREAD_GRAYSCALE);
	Mat img_c = imread("pic3.bmp", IMREAD_GRAYSCALE);
	Mat img_d = imread("pic4.bmp", IMREAD_GRAYSCALE);
	Mat img_e = imread("pic5.bmp", IMREAD_GRAYSCALE);
	Mat img_f  = imread("pic6.bmp", IMREAD_GRAYSCALE);

	Mat_<uchar> img1(img_a);
	Mat_<uchar> img2(img_b);
	Mat_<uchar> img3(img_c);
	Mat_<uchar> img4(img_d);
	Mat_<uchar> img5(img_e);
	Mat_<uchar> img6(img_f );

	ifstream infile;
	infile.open("LightSource.txt");

	float source[18];
	char trash[10];

	for (int i = 0; i < 18; i = i + 3) {
		infile.getline(trash, 10, '(');
		infile >> source[i];
		infile >> trash[0];
		infile >> source[i+1];
		infile >> trash[0];
		infile >> source[i+2];
		infile >> trash;
	}
	
	float temp_sum =0;

	for (int i = 0; i < 18; i++) {
		temp_sum = source[i] * source[i] + temp_sum;	
		if (i % 3 == 2) {
			temp_sum = sqrt(temp_sum);
			source[i - 2] = source[i - 2] / temp_sum;
			source[i - 1] = source[i - 1] / temp_sum;
			source[i] = source[i] / temp_sum;
			temp_sum = 0;
		}
	}


	CvMat *S = cvCreateMat(6, 3, CV_32FC1);
	cvSetData(S, source, S->step);
//PrintMatrix(S, S->rows, S->cols);
//cout <<cvmGet(S,1,2)<<endl;
//system("pause");
	
	CvMat *St = cvCreateMat(3, 6, CV_32FC1);
	CvMat *inv = cvCreateMat(3, 3, CV_32FC1);
	CvMat *StS = cvCreateMat(3, 3, CV_32FC1);
	CvMat *result = cvCreateMat(3, 6, CV_32FC1);

	cvTranspose(S, St);

	cvMatMul(St, S,StS);

	cvInvert(StS, inv, CV_LU);

	cvMatMul(inv, St, result);
//PrintMatrix(result, result->rows, result->cols);
//system("pause");
	float intensity[6];
	CvMat *intensity_mat = cvCreateMat(6, 1, CV_32FC1);
	CvMat ***b;
	b = new CvMat**[img1.rows];
	for (int i = 0; i < img1.rows; i++) {
		b[i] = new CvMat*[img1.cols];
	}

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			b[i][j] = cvCreateMat(3, 1, CV_32FC1);
		}
	}
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			intensity[0] = img1.at<uchar>(i, j);
			intensity[1] = img2.at<uchar>(i, j);
			intensity[2] = img3.at<uchar>(i, j);
			intensity[3] = img4.at<uchar>(i, j);
			intensity[4] = img5.at<uchar>(i, j);
			intensity[5] = img6.at<uchar>(i, j);
			cvSetData(intensity_mat, intensity, intensity_mat->step);
			cvMatMul(result, intensity_mat, b[i][j]);
		}
	}

	float **a1;
	a1 = new float* [img1.rows];
	for (int i = 0; i < img1.rows; i++) {
		a1[i] = new float [img1.cols];
	}

	float **a2;
	a2 = new float* [img1.rows];
	for (int i = 0; i < img1.rows; i++) {
		a2[i] = new float [img1.cols];
	}

	float n1, n2, n3;
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			n1 = cvGet2D(b[i][j], 0, 0).val[0];
			n2 = cvGet2D(b[i][j], 1, 0).val[0];
			n3 = cvGet2D(b[i][j], 2, 0).val[0];
			if (n3 == 0) {
				a1[i][j] = 0;
				a2[i][j] = 0;
			}
			else {
				a1[i][j] = (-1) * n1 / n3;
				a2[i][j] =  (-1)*n2 / n3;
			}
		}
	}

cout << a1[80][80] << endl;
//system("pause");
cout << a2[80][80] << endl;
//system("pause");

	float **z;
	z = new float*[img1.rows];
	for (int i = 0; i < img1.rows; i++) {
		z[i] = new float[img1.cols];
	}

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			z[i][j] = 0;

			for (int k = 0; k < i; k++) {
				z[i][j] = z[i][j] + a2[k][0];
			}
			for (int l = 0; l < j; l++) {
				z[i][j] = z[i][j] + a1[i][l];
			}

		if (img1.at<uchar>(i, j) == 0) {
				z[i][j] = 0;
		}
		if (z[i][j] < -100) {
			z[i][j] = -100;
		}
			
/*
			for (int k = 0; k < i; k++) {
				z[i][j] = z[i][j] + a2[k][0];
			}
			for (int l = 0; l < j; l++) {
				z[i][j] = z[i][j] + a1[i][l];
			}
			z[i][j] = z[i][j] / 2;
	*/
		}
	}
cout << z[80][80] << endl;
//system("pause");


	ofstream outfile;
	outfile.open("hw1_test.ply");
	outfile << "ply" << endl;
	outfile << "format ascii 1.0" << endl;
	outfile << "comment alpha = 1.0" << endl;
	outfile << "element vertex 14400" << endl;
	outfile << "property float x" << endl;
	outfile << "property float y" << endl;
	outfile << "property float z" << endl;
	outfile << "property uchar red" << endl;
	outfile << "property uchar green" << endl;
	outfile << "property uchar blue z" << endl;
	outfile << "end_header" << endl;

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			outfile << i << " " << j << " " << z[i][j] << " " << "255 255 255" << endl;
		}
	}


//	cout <<endl<<endl << cvGet2D(b[80][80], 1, 0).val[0] <<endl<<endl;
//	system("pause");


/*
	float test = img1.at<uchar>(80, 80);

	cout <<"intensity :"<< test << endl;

	
	for (int j = 0; j < 50; j++) {
		for (int k = 0; k < 50; k++) {
			img1.at<uchar>(k, j) = 255;
		}
	}
	*/

	





	cv::imshow("Hi", img1);
	cv::waitKey(); // Wait for the user to press a key.
	return 0;
}

void PrintMatrix(CvMat *Matrix, int Rows, int Cols)
{
	for (int i = 0; i<Rows; i++)
	{
		for (int j = 0; j<Cols; j++)
		{
			printf("%.4f ", cvGet2D(Matrix, i, j).val[0]);
		}
		printf("\n");
	}
}
