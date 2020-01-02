# FRC Team 612 Vision
> Reflective tape vision algrorithm for FRC

Python-side of vision algrorithm to detect reflective tape and send data to ROBORIO through pynetworktables.
1) Identifies contours and finds bounding boxes of tape.
2) Evaluates the angles created by the midpoints of the short side of the tape.
3) Determines tape pairs based on opposite negative/positive angles
4) Find the outer vertex points of each tape in pair.
5) Determines intersection point of vectors formed by top vertex and bottom vertex of opposite tapes.
6) Returns offset of intersection point from center, and length of each vector.

<!--  ![](header.png)  - Input image -->

