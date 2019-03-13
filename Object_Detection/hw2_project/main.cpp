#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <time.h>

using namespace std;
using namespace cv;

class correspondence {
public:
	KeyPoint obj;
	KeyPoint tar;
	float distance;
};

bool sort_condi(correspondence a, correspondence b) {
	return a.distance < b.distance;
}

void knn(int, int, Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, vector<vector<correspondence>> &);
void RANSAC(vector<vector<correspondence>>, vector<vector<correspondence>>,int &,Mat &);

int main()
{
	Mat img_1 = imread("object13.bmp");
	Mat img_2 = imread("target_1.bmp");
	Mat img_3 = imread("target_1.bmp");

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	////-- Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)    
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);

	Mat feat1, feat2 ,feat3,feat4;
	drawKeypoints(img_1, keypoints_1, feat1);
	drawKeypoints(img_2, keypoints_2, feat2);
	imshow("Object_KeyPoint.jpg", feat1);
	imshow("Target_KeyPoint.jpg", feat2);
	imwrite("Object_KeyPoint.jpg", feat1);
	imwrite("Target_KeyPoint.jpg", feat2);
	int key1 = keypoints_1.size();
	int key2 = keypoints_2.size();
	printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);
	printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)", descriptors_1.size().height, descriptors_1.size().width, descriptors_2.size().height, descriptors_2.size().width);

	//---------------    find k-n-n   ---------------------------------------------
	vector<vector<correspondence>> knn_matrix;
	knn(key1, key2, descriptors_1, descriptors_2, keypoints_1 ,keypoints_2, knn_matrix);
	//--  get the most accurate keypoint 
	vector<vector<correspondence>> accurate_pt;
	for (int i = 0; i < key1; i++) {
		if ( abs(knn_matrix[i][0].distance - knn_matrix[i][1].distance ) > 500) {
			accurate_pt.push_back(knn_matrix[i]);
		}
	}

	//----------------------------   RANSAC   ---------------------------------
	int loop;
	loop = 0;
	int inliers=0;
	Mat homography(3, 3, CV_32FC1);
	while ( loop<50  && inliers < keypoints_1.size()/2 ) {
		RANSAC(accurate_pt, knn_matrix, inliers, homography);
		loop++;
		cout << endl << loop;
//cout << endl << inliers;
	}
	cout << endl << "final inliers: " << endl << inliers;

	//----------------------------   Warping   --------------------------------
	for (float r = 0; r < img_1.rows; r++) {
		for (float c = 0; c < img_1.cols; c++) {
			int R = img_1.at<Vec3b>(r, c)[0];
			int G = img_1.at<Vec3b>(r, c)[1];
			int B = img_1.at<Vec3b>(r, c)[2];
			if (R != 255 || G != 255 || B != 255) {
				Mat original_point = (Mat_<float>(3, 1) << c, r, 1);
				Mat trans_point(3, 1, CV_32FC1);
				trans_point = homography * original_point;
				float warp_point_x = trans_point.at<float>(0, 0) / trans_point.at<float>(2, 0);
				float warp_point_y = trans_point.at<float>(1, 0) / trans_point.at<float>(2, 0);
				if (warp_point_x >= 0 &&warp_point_x<img_3.cols && warp_point_y >= 0 && warp_point_y<img_3.rows) {
					img_3.at<Vec3b>(warp_point_y, warp_point_x)[0] = R;
					img_3.at<Vec3b>(warp_point_y, warp_point_x)[1] = G;
					img_3.at<Vec3b>(warp_point_y, warp_point_x)[2] = B;
				}
			}
		}
	}
	
	imshow("result_image.jpg", img_3);
	imwrite("result_image13.jpg", img_3);

	cout << endl << "finish";
	waitKey(0);
	return 0;

}

void knn(int key1,int key2,Mat descriptors_1,Mat descriptors_2,vector<KeyPoint>keypoints_1,vector<KeyPoint> keypoints_2, vector<vector<correspondence>> &knn_matrix)
{
//	Mat temp_row(1,128,CV_64FC1);
	float descriptor_sum;
	vector<correspondence> target_pt_dis;
	correspondence temp_corr;

	for (int i = 0; i < key1; i++) {
		for (int j = 0; j < key2; j++) {
//			temp_row.row(0) = descriptors_1.row(i) - descriptors_2.row(j);
			descriptor_sum = 0;
			for (int k = 0; k < 128; k++) {
				descriptor_sum = descriptor_sum + abs(descriptors_1.at<float>(i, k) - descriptors_2.at<float>(j, k) );
			}
//			descriptor_sum = sqrt(descriptor_sum);
			temp_corr.obj = keypoints_1[i];
			temp_corr.tar = keypoints_2[j];
			temp_corr.distance = descriptor_sum;
			target_pt_dis.push_back(temp_corr);
		}
		sort ( target_pt_dis.begin() , target_pt_dis.end() ,sort_condi ) ;
		knn_matrix.push_back(target_pt_dis);
		target_pt_dis.clear( );
	}

}

void RANSAC(vector<vector<correspondence>> accurate_pt, vector<vector<correspondence>> knn_matrix, int &best_inliers,Mat &best_homo)
{
	unsigned seed;
	seed = (unsigned)time(NULL);
	srand(seed);

	int p[4];
	p[0] = rand() % accurate_pt.size(); // random pick  4  point  from  accurate point
	p[1] = rand() % accurate_pt.size();
	p[2] = rand() % accurate_pt.size();
	p[3] = rand() % accurate_pt.size();

	//-- build the 8*9 matrix for 4-n-n and calculate homography matrix
	Mat calcu_homo(8, 9, CV_32FC1);
	
	for (int a = 0; a < 4; a++) {
		for (int b = 0; b < 4; b++) {
			for (int c = 0; c < 4; c++) {
				for (int d = 0; d < 4; d++) {
					//--build matrix U (8*9)
					calcu_homo.at<float>(0, 0) = (float)accurate_pt[p[0]][0].obj.pt.x;
					calcu_homo.at<float>(0, 1) = (float)accurate_pt[p[0]][0].obj.pt.y;
					calcu_homo.at<float>(0, 2) = 1;
					calcu_homo.at<float>(0, 3) = 0;
					calcu_homo.at<float>(0, 4) = 0;
					calcu_homo.at<float>(0, 5) = 0;
					calcu_homo.at<float>(0, 6) = (float)(-1 * accurate_pt[p[0]][0].obj.pt.x * accurate_pt[p[0]][a].tar.pt.x);
					calcu_homo.at<float>(0, 7) = (float)(-1 * accurate_pt[p[0]][0].obj.pt.y * accurate_pt[p[0]][a].tar.pt.x);
					calcu_homo.at<float>(0, 8) = (float)(-1 * accurate_pt[p[0]][a].tar.pt.x);
					calcu_homo.at<float>(1, 0) = 0;
					calcu_homo.at<float>(1, 1) = 0;
					calcu_homo.at<float>(1, 2) = 0;
					calcu_homo.at<float>(1, 3) = (float)accurate_pt[p[0]][0].obj.pt.x;
					calcu_homo.at<float>(1, 4) = (float)accurate_pt[p[0]][0].obj.pt.y;
					calcu_homo.at<float>(1, 5) = 1;
					calcu_homo.at<float>(1, 6) = (float)(-1 * accurate_pt[p[0]][0].obj.pt.x * accurate_pt[p[0]][a].tar.pt.y);
					calcu_homo.at<float>(1, 7) = (float)(-1 * accurate_pt[p[0]][0].obj.pt.y * accurate_pt[p[0]][a].tar.pt.y);
					calcu_homo.at<float>(1, 8) = (float)(-1 * accurate_pt[p[0]][a].tar.pt.y);

					calcu_homo.at<float>(2, 0) = (float)accurate_pt[p[1]][0].obj.pt.x;
					calcu_homo.at<float>(2, 1) = (float)accurate_pt[p[1]][0].obj.pt.y;
					calcu_homo.at<float>(2, 2) = 1;
					calcu_homo.at<float>(2, 3) = 0;
					calcu_homo.at<float>(2, 4) = 0;
					calcu_homo.at<float>(2, 5) = 0;
					calcu_homo.at<float>(2, 6) = (float)(-1 * accurate_pt[p[1]][0].obj.pt.x * accurate_pt[p[1]][b].tar.pt.x);
					calcu_homo.at<float>(2, 7) = (float)(-1 * accurate_pt[p[1]][0].obj.pt.y * accurate_pt[p[1]][b].tar.pt.x);
					calcu_homo.at<float>(2, 8) = (float)(-1 * accurate_pt[p[1]][b].tar.pt.x);
					calcu_homo.at<float>(3, 0) = 0;
					calcu_homo.at<float>(3, 1) = 0;
					calcu_homo.at<float>(3, 2) = 0;
					calcu_homo.at<float>(3, 3) = (float)accurate_pt[p[1]][0].obj.pt.x;
					calcu_homo.at<float>(3, 4) = (float)accurate_pt[p[1]][0].obj.pt.y;
					calcu_homo.at<float>(3, 5) = 1;
					calcu_homo.at<float>(3, 6) = (float)(-1 * accurate_pt[p[1]][0].obj.pt.x * accurate_pt[p[1]][b].tar.pt.y);
					calcu_homo.at<float>(3, 7) = (float)(-1 * accurate_pt[p[1]][0].obj.pt.y * accurate_pt[p[1]][b].tar.pt.y);
					calcu_homo.at<float>(3, 8) = (float)(-1 * accurate_pt[p[1]][b].tar.pt.y);

					calcu_homo.at<float>(4, 0) = (float)accurate_pt[p[2]][0].obj.pt.x;
					calcu_homo.at<float>(4, 1) = (float)accurate_pt[p[2]][0].obj.pt.y;
					calcu_homo.at<float>(4, 2) = 1;
					calcu_homo.at<float>(4, 3) = 0;
					calcu_homo.at<float>(4, 4) = 0;
					calcu_homo.at<float>(4, 5) = 0;
					calcu_homo.at<float>(4, 6) = (float)(-1 * accurate_pt[p[2]][0].obj.pt.x * accurate_pt[p[2]][c].tar.pt.x);
					calcu_homo.at<float>(4, 7) = (float)(-1 * accurate_pt[p[2]][0].obj.pt.y * accurate_pt[p[2]][c].tar.pt.x);
					calcu_homo.at<float>(4, 8) = (float)(-1 * accurate_pt[p[2]][c].tar.pt.x);
					calcu_homo.at<float>(5, 0) = 0;
					calcu_homo.at<float>(5, 1) = 0;
					calcu_homo.at<float>(5, 2) = 0;
					calcu_homo.at<float>(5, 3) = (float)accurate_pt[p[2]][0].obj.pt.x;
					calcu_homo.at<float>(5, 4) = (float)accurate_pt[p[2]][0].obj.pt.y;
					calcu_homo.at<float>(5, 5) = 1;
					calcu_homo.at<float>(5, 6) = (float)(-1 * accurate_pt[p[2]][0].obj.pt.x * accurate_pt[p[2]][c].tar.pt.y);
					calcu_homo.at<float>(5, 7) = (float)(-1 * accurate_pt[p[2]][0].obj.pt.y * accurate_pt[p[2]][c].tar.pt.y);
					calcu_homo.at<float>(5, 8) = (float)(-1 * accurate_pt[p[2]][c].tar.pt.y);

					calcu_homo.at<float>(6, 0) = (float)accurate_pt[p[3]][0].obj.pt.x;
					calcu_homo.at<float>(6, 1) = (float)accurate_pt[p[3]][0].obj.pt.y;
					calcu_homo.at<float>(6, 2) = 1;
					calcu_homo.at<float>(6, 3) = 0;
					calcu_homo.at<float>(6, 4) = 0;
					calcu_homo.at<float>(6, 5) = 0;
					calcu_homo.at<float>(6, 6) = (float)(-1 * accurate_pt[p[3]][0].obj.pt.x * accurate_pt[p[3]][d].tar.pt.x);
					calcu_homo.at<float>(6, 7) = (float)(-1 * accurate_pt[p[3]][0].obj.pt.y * accurate_pt[p[3]][d].tar.pt.x);
					calcu_homo.at<float>(6, 8) = (float)(-1 * accurate_pt[p[3]][d].tar.pt.x);
					calcu_homo.at<float>(7, 0) = 0;
					calcu_homo.at<float>(7, 1) = 0;
					calcu_homo.at<float>(7, 2) = 0;
					calcu_homo.at<float>(7, 3) = (float)accurate_pt[p[3]][0].obj.pt.x;
					calcu_homo.at<float>(7, 4) = (float)accurate_pt[p[3]][0].obj.pt.y;
					calcu_homo.at<float>(7, 5) = 1;
					calcu_homo.at<float>(7, 6) = (float)(-1 * accurate_pt[p[3]][0].obj.pt.x * accurate_pt[p[3]][d].tar.pt.y);
					calcu_homo.at<float>(7, 7) = (float)(-1 * accurate_pt[p[3]][0].obj.pt.y * accurate_pt[p[3]][d].tar.pt.y);
					calcu_homo.at<float>(7, 8) = (float)(-1 * accurate_pt[p[3]][d].tar.pt.y);

					Mat mul_tranpose = calcu_homo.t()* calcu_homo; //--Ut * U
					vector<float> eigenvalues;
					Mat eigenvectors(9, 9, CV_32FC1);
					eigen(mul_tranpose, eigenvalues, eigenvectors); //--find eigenvalues and eigenvectors in Ut * U 
					float eig_small[9];
					for (int i = 0; i < 9; i++) {
						eig_small[i]= eigenvectors.at<float>(8,i);
					}
					Mat homo_matrix = Mat(3, 3, CV_32FC1,eig_small).clone();//--get the 3*3 homography matrix from smallest eigenvalue's eigenvector
					int inliers_number;
					inliers_number = 0;
					//-- for all object keypoint
					for (int i = 0; i < knn_matrix.size(); i++) {  
						Mat cordinate_obj = (Mat_<float>(3, 1) << knn_matrix[i][0].obj.pt.x , knn_matrix[i][0].obj.pt.y , 1);
						Mat cordinate_homo_trans(3, 1, CV_32FC1);
						cordinate_homo_trans = homo_matrix * cordinate_obj;
						float trans_point_x = cordinate_homo_trans.at<float>(0, 0) / cordinate_homo_trans.at<float>(2, 0);
						float trans_point_y = cordinate_homo_trans.at<float>(1, 0) / cordinate_homo_trans.at<float>(2, 0);
						float dis;
						//--  dis  =  sqrt(  (X1-X2)^2+(Y1-Y2)^2 )
						dis =sqrt( (knn_matrix[i][0].tar.pt.x - trans_point_x) *(knn_matrix[i][0].tar.pt.x - trans_point_x) + (knn_matrix[i][0].tar.pt.y - trans_point_y) * (knn_matrix[i][0].tar.pt.y - trans_point_y) ) ;
						//--  count inliers
						if (dis < 5) {
							inliers_number = inliers_number + 1;
						}
					}
					//-- store the best homography matrix
					if (inliers_number > best_inliers) {
						best_inliers = inliers_number;
						best_homo = homo_matrix;
					}
//cout << endl << "temp  inliers: " << inliers_number;
//	cout <<endl<< "best inliers:  " << best_inliers;
			 
					
/*
cout << endl << endl << "eigenvalues:";
for (int i = 0; i < 9; i++) {
	cout << endl << eigenvalues[i];
}
cout << endl << endl << "eigenvector:" << endl;
for (int i = 0; i < 9; i++) {
	for (int j = 0; j < 9; j++) {
		cout << eigenvectors.at<float>(i, j) << " ";
	}
	cout << endl;
}
cout << "3*3" << endl;
for (int i = 0; i < 3; i++) {
	cout <<homo_matrix.row(i)<< " ";
}
waitKey(0);
*/


				}
			}
		}
	}
	





}