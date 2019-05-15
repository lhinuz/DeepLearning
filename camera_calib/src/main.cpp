#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main()
{
    int numBoardToTest = 0;
    int numCornersHor;
    int numCornersVer;

    printf("Enter number of corners horizontally: ");
    scanf("%d", &numCornersHor);
    printf("Enter number of corners vertically: ");
    scanf("%d", &numCornersVer);
    printf("Enter number of boards: ");
    scanf("%d", &numBoardToTest);

    int numSquares = numCornersHor * numCornersVer;
    Size boardSize = Size(numCornersHor, numCornersVer);

    // capture frome camera
    VideoCapture capture = VideoCapture(0);

    vector<vector<Point3f> > objectPos;
    vector<vector<Point2f> > imagePos;

    vector<Point2f> corners;
    int successes=0;

    Mat image;
    Mat grayImage;

    vector<Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));

    while( successes < numBoardToTest )
    {
        capture >> image;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        bool found = findChessboardCorners(image, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            drawChessboardCorners(grayImage, boardSize, corners, found);
        }

        imshow("origin", image);
        imshow("gray", grayImage);

        int key = waitKey(1);
        if(27 == key)
        {
            return 0;
        }

        if(' ' == key and found)
        {
            imagePos.push_back(corners);
            objectPos.push_back(obj);
            successes++;

            if(successes>=numBoardToTest)
                break;
        }
    }

    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    calibrateCamera(objectPos, imagePos, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

    Mat imageUndistorted;
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, intrinsic, distCoeffs);

        imshow("origin", image);
        imshow("corrected", imageUndistorted);
        if (27 == waitKey(1))
            break;
    }

    capture.release();

    return 0;
}
