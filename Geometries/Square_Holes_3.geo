
L=10;
L1 = L/10;

L2 = L/6;

L3 = L/20;


x_1 = L*1/10;
x_2 = L*2/3;
x_3 = L*1/2;

y_1 = L*1/10;
y_2 = L*3/4;
y_3 = L*1/2;

dh = L; // Minimum size is the length of the beam
//Mesh.Algorithm = 8;

Point(1) = {0, 0, 0, dh};
Point(2) = {L, 0, 0, dh};
Point(3) = {L, L, 0, dh};
Point(4) = {0, L, 0, dh};

Point(5) = {x_1, y_1, 0, dh};
Point(6) = {x_1+L1, y_1, 0, dh};
Point(7) = {x_1+L1, y_1+L1, 0, dh};
Point(8) = {x_1, y_1+L1, 0, dh};

Point(50) = {x_2, y_2, 0, dh};
Point(60) = {x_2+L2, y_2, 0, dh};
Point(70) = {x_2+L2, y_2+L2, 0, dh};
Point(80) = {x_2, y_2+L2, 0, dh};

Point(51) = {x_3, y_3, 0, dh};
Point(61) = {x_3+L3, y_3, 0, dh};
Point(71) = {x_3+L3, y_3+L3, 0, dh};
Point(81) = {x_3, y_3+L3, 0, dh};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(10) = {5, 6};
Line(20) = {6, 7};
Line(30) = {7, 8};
Line(40) = {8, 5};

Line(11) = {50, 60};
Line(21) = {60, 70};
Line(31) = {70, 80};
Line(41) = {80, 50};

Line(12) = {51, 61};
Line(22) = {61, 71};
Line(32) = {71, 81};
Line(42) = {81, 51};



Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {10,20,30,40};
Curve Loop(20) = {11,21,31,41};
Curve Loop(21) = {12,22,32,42};


Plane Surface(1) = {1, 2, 20, 21};

Physical Surface(100) = {1};

Physical Line (111) = {1} ;
Physical Line (112) = {2} ;
Physical Line (113) = {3} ;
Physical Line (114) = {4} ;

Physical Curve (115) = {10,20,30,40,11,21,31,41,12,22,32,42} ;

Physical Point (200) = {1,2};





Mesh.MshFileVersion = 2.2;

