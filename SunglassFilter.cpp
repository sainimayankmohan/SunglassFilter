#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "dataPath.hpp"

using namespace std;
using namespace cv;

/** Global variables */
String faceCascadePath;
CascadeClassifier faceCascade;

int main( int argc, const char** argv )
{
	int faceNeighborsMax = 10;
	int neighborStep = 1;
	faceCascadePath = DATA_PATH + "models/haarcascade_frontalface_default.xml";

	if( !faceCascade.load( faceCascadePath))
	{
	  cout<<"Error loading face cascade"<<endl;
	  return -1;
	}

	VideoCapture cap(0);

	if(cap.isOpened())
	{
		cout<<"Cap is open"<<endl;
	}
	else
	{
		cout<<"Error opening the video capture"<<endl;
		return -1;
	}

	Mat glass = imread(DATA_PATH + "images/sunglass.png",-1);

	while(1){

		Mat frame;
		// Capture frame-by-frame
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
		  break;

		// Display the resulting frame
		imshow( "Frame", frame );

		Mat frameGray;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);

		std::vector<Rect> faces;
		Mat frameClone = frame.clone();
		int neighbours = 8;
		faceCascade.detectMultiScale( frameGray, faces, 1.2, neighbours);

		for(int i = 0; i < faces.size();i++)
		{
			Mat sunglass = glass.clone();

			int x = faces[i].x;
			int y = faces[i].y;
			int w = faces[i].width;
			int h = faces[i].height;

			rectangle(frame, Point(x, y), Point(x + w, y + h), Scalar(255,0,0), 2, 4);

		  //Now get the eye region patch for sunglass replacement
		  //The detected face region will have the eye region near the middle of the face region

			int patchX = x + faces[i].width/8;
			int patchY = y + faces[i].height/4 + faces[i].height/12;
			int patchWidth = faces[i].width - 2*faces[i].width/8;
			int patchHeight = faces[i].height/4;

			// Resize the image to fit over the eye region
			resize(sunglass,sunglass, Size(patchWidth,patchHeight));

			// Separate the Color and alpha channels
			Mat glassRGBAChannels[4];
			Mat glassRGBChannels[3];
			split(sunglass, glassRGBAChannels);

			Mat glassMask = glassRGBAChannels[3];

			for (int i = 0; i < 3; i++){
			    // Copy R,G,B channel from RGBA to RGB
			    glassRGBChannels[i] = glassRGBAChannels[i];
			}

			Mat eyePatch = frame(Range(patchY,patchY+patchHeight),Range(patchX,patchX + patchWidth));

			// Make the values [0,1] since we are using arithmetic operations
			glassMask = glassMask/255;

			for (int i = 0; i < 3; i++)
			{
				// Use the mask to create the masked sunglass region
			    multiply(glassRGBChannels[i], glassMask, glassRGBChannels[i]);
			}

			merge(glassRGBChannels,3,sunglass);

			for(int i = 0;i < sunglass.rows;i++)
			{
				for(int j=0;j<sunglass.cols;j++)
				{
					if(sunglass.at<Vec3b>(i,j) != Vec3b(0,0,0))
					{
						eyePatch.at<Vec3b>(i,j) = 0.6*eyePatch.at<Vec3b>(i,j);
						sunglass.at<Vec3b>(i,j) = 0.4*sunglass.at<Vec3b>(i,j);
					}
				}
			}

			add(eyePatch, sunglass, eyePatch);
		}

		imshow("Output",frame);

		// Press ESC on keyboard to exit
		char c=(char)waitKey(25);
		if(c==27)
		  break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();
	return 0;
}
