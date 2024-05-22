
L=10;
r = L/4;

dh = L; // Minimum size is the length of the beam
Mesh.Algorithm = 5;

Point(1) = {0, 0, 0, dh};
Point(2) = {L, 0, 0, dh};
Point(3) = {L, L, 0, dh};
Point(4) = {0, L, 0, dh};

Point(5) = {L/2, L/2, 0, dh};

Point(6) = {L/2, L/2-r, 0, dh};
Point(7) = {L-r, L/2, 0, dh};
Point(8) = {L/2, L/2+r, 0, dh};
Point(9) = {L/2-r, L/2, 0, dh};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {6,5,7};
Circle(6) = {7,5,8};
Circle(7) = {8,5,9};
Circle(8) = {9,5,6};

Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {5,6,7,8};

Plane Surface(1) = {1, 2};

Physical Surface(100) = {1};

Physical Line (111) = {1} ;
Physical Line (112) = {2} ;
Physical Line (113) = {3} ;
Physical Line (114) = {4} ;

Physical Curve (115) = {5,6,7,8} ;

Physical Point (200) = {1,2,3,4};





Mesh.MshFileVersion = 2.2;

