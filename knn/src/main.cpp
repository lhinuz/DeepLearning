#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

const char *train_images_file_name = "../mnist/train-images-idx3-ubyte";
const char *train_labels_file_name = "../mnist/train-labels-idx1-ubyte";
const char *test_images_file_name = "../mnist/t10k-images-idx3-ubyte";
const char *test_labels_file_name = "../mnist/t10k-labels-idx1-ubyte";

int readFlippedInteger(FILE *fp);

int main()
{
    // open training files
    FILE *train_images = fopen(train_images_file_name, "rb");
    FILE *train_labels = fopen(train_labels_file_name, "rb");
    if (train_images == nullptr or train_labels == nullptr)
    {
        printf("%s\n", "Failed open train files.");
        return EXIT_SUCCESS;
    }

    // read head
    int magicNumber = readFlippedInteger(train_images);
    // printf("%#010x\n", magicNumber);
    int numImages = readFlippedInteger(train_images);
    int numRows = readFlippedInteger(train_images);
    int numCols = readFlippedInteger(train_images);
    fseek(train_labels, 8, SEEK_SET);

    // read data
    int size = numRows * numCols;
    cv::Mat trainingImageVector;
    trainingImageVector.create(numImages, size, CV_32FC1);
    cv::Mat trainingLabelVector;
    trainingLabelVector.create(numImages, 1, CV_32FC1);

    unsigned char *temp = new unsigned char[size];
    unsigned char tempClass;
    for (int i=0; i<numImages; i++)
    {
        fread(temp, size, 1, train_images);
        fread(&tempClass, 1, 1, train_labels);

        trainingLabelVector.at<float>(i, 0) = tempClass;
        for (int j=0; j<size; j++)
        {
            trainingImageVector.at<float>(i, j) = temp[j];
        }
    }

    fclose(train_images);
    fclose(train_labels);

    // train knn
    printf("%s\n", "start trainning...");
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(10);
    knn->setIsClassifier(true);
    knn->train(trainingImageVector, 0, trainingLabelVector);
    if (!knn->isTrained())
    {
        return 0;
    }

    // read test data
    FILE *test_image_fp = fopen(test_images_file_name, "rb");
    FILE *test_label_fp = fopen(test_labels_file_name, "rb");
    if (test_image_fp == nullptr or test_label_fp == nullptr)
    {
        return 0;
    }

    // read header
    magicNumber = readFlippedInteger(test_image_fp);
    numImages = readFlippedInteger(test_image_fp);
    numRows = readFlippedInteger(test_image_fp);
    numCols = readFlippedInteger(test_image_fp);
    fseek(test_label_fp, 0x08, SEEK_SET);

    printf("%s\n", "start predict...");
    size = numRows * numCols;
    int count = 0;
    cv::Mat sample(1, size, CV_32FC1);
    cv::Mat predict_result(1, 1, CV_32FC1);
    for (int i=0; i<numImages; i++)
    {
        fread(temp, size, 1, test_image_fp);
        fread(&tempClass, 1, 1, test_label_fp);
        for (int j=0; j<size; j++)
        {
            sample.at<float>(0, j) = (float)temp[j];
        }

        float res = knn->findNearest(sample, 10, predict_result);
        if (predict_result.at<float>(0, 0) == (float)tempClass)
        {
            printf("found %f, %d\n", predict_result.at<float>(0, 0), tempClass);
            count++;
        }
        else
        {
            printf("not found %f, %d\n", predict_result.at<float>(0, 0), tempClass);
        }
    }

    printf("predict success at %f\n", count * 1.0f/numImages);

    delete [] temp;
    fclose(test_image_fp);
    fclose(test_label_fp);

    return EXIT_SUCCESS;
}

int readFlippedInteger(FILE *fp)
{
    int ret = 0;
    fread(&ret, sizeof(int), 1, fp);
    // printf("%#010x\n", ret);
    return ((ret & 0x000000FF) << 24)
        | ((ret & 0xFF000000) >> 24)
        | ((ret & 0x0000FF00) << 8)
        | ((ret & 0x00FF0000) >> 8);
}
