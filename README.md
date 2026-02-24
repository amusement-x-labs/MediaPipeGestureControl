# MediaPipeGestureControll
MediaPipe Gesture Controller (Python)

This folder contains the "brain" of the project. It uses MediaPipe to track hand landmarks and calculates the distance between the thumb and middle finger to trigger the drawing state.
Features
* CV Engine: Real-time 21-point hand tracking.
* Gesture Logic: Detects "Pinch" (Thumb tip to Middle finger tip).
* Communication: Acts as a Named Pipe Server, streaming coordinates to Unity.

## Related repos:
* Unity project: https://github.com/amusement-x-labs/MediaPipeGestureControlDrawer

## Fullfit demonstration
https://youtu.be/C2TmPbPkd5g