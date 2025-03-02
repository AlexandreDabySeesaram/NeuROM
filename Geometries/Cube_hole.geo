dh=200;
R = 2; //radius of the hole
L=10;
La = 10;
H=10;

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




Sphere(2) = {5, 5, 5, 6, -Pi/2, Pi/2, 2*Pi};


//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
//+
Physical Surface(114) = {5};
//+
Physical Surface(115) = {2};
//+
Physical Surface(111) = {1};
//+
Physical Surface(112) = {7};
//+
Physical Surface(113) = {4};
//+
Physical Surface(110) = {3};
//+
Physical Surface(210) = {6};
//+
Physical Volume(100) = {1};

// export
Mesh.MshFileVersion = 2.2;