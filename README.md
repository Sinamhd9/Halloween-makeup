# Halloween-makeup
A repository for creating innovative digital makeups for Halloween

<p align="center">
  <img src="demo.gif" alt="Halloween makeup demo" width="400" height="300" />
</p>

## Overview
A joker Halloween makeup that is built on the user camera input.

It uses face recognition to detect the face landmarks, OpenCV for reading camera input and frame manipulation,
and PIL to draw shapes.

If the user smile, the eyes will turn into a bold random color.

### How it works

The program gets the face landmark coordinates using face recognition's landmark detection.

All we need is to manipulate the features in a scale invariant way that the features stick in the right place when the person changes her position with respect to the camera.
In order to this, I used the euclidean distance between to end points of the chin as a scale for all other features.

At last, the features (in this case joker's face features) are drawn for all faces detected in the frame. 

### How to run

Install the [required libraries](/requirements.txt) and run the main.py file. 

If you are having problem with installing face recognition or dlib, check this out [face recognition installation](https://github.com/ageitgey/face_recognition)

