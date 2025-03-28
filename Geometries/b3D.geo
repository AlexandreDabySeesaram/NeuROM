dh=2;
R = 2; //radius of the hole
L=40;
La = 8;
H=8;

SetFactory("OpenCASCADE");
Point(1) = {0,0,0,dh};
Point(2) = {La,0,0,dh};
Point(3) = {La,L,0,dh};
Point(4) = {0,L,0,dh};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(5) = {1,2,3,4};

Plane Surface(6) = {5};


surfaceVector[] = Extrude {0, 0, H} {
    Surface{6};
};

Physical Surface(110) = surfaceVector[0];
Physical Volume(100) = surfaceVector[1];
Physical Surface(114) = surfaceVector[2];
Physical Surface(111) = surfaceVector[3];
Physical Surface(115) = surfaceVector[4];
Physical Surface(112) = surfaceVector[5];
Physical Surface(113) = {6};


// export
Mesh.MshFileVersion = 2.2;