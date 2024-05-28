
L=10;
r1 = L/10;

r2 = L/6;

r3 = L/20;


x_1_1 = L*1/4;
x_1_2 = L*3/4;
x_2 = L*1/2;
y_1 = L*1/4;
y_2 = L*3/4;

dh = L; // Minimum size is the length of the beam
//Mesh.Algorithm = 8;

Point(1) = {0, 0, 0, dh};
Point(2) = {L, 0, 0, dh};
Point(3) = {L, L, 0, dh};
Point(4) = {0, L, 0, dh};

Point(5) = {x_2, y_2, 0, dh};

Point(6) = {x_2, y_2-r2, 0, dh};
Point(7) = {x_2+r2, y_2, 0, dh};
Point(8) = {x_2, y_2+r2, 0, dh};
Point(9) = {x_2-r2, y_2, 0, dh};

Point(50) = {x_1_1, y_1, 0, dh};

Point(60) = {x_1_1, y_1-r1, 0, dh};
Point(70) = {x_1_1+r1, y_1, 0, dh};
Point(80) = {x_1_1, y_1+r1, 0, dh};
Point(90) = {x_1_1-r1, y_1, 0, dh};

Point(51) = {x_1_2, y_1, 0, dh};

Point(61) = {x_1_2, y_1-r3, 0, dh};
Point(71) = {x_1_2+r3, y_1, 0, dh};
Point(81) = {x_1_2, y_1+r3, 0, dh};
Point(91) = {x_1_2-r3, y_1, 0, dh};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {6,5,7};
Circle(6) = {7,5,8};
Circle(7) = {8,5,9};
Circle(8) = {9,5,6};

Circle(50) = {60,50,70};
Circle(60) = {70,50,80};
Circle(70) = {80,50,90};
Circle(80) = {90,50,60};

Circle(51) = {61,51,71};
Circle(61) = {71,51,81};
Circle(71) = {81,51,91};
Circle(81) = {91,51,61};

Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {5,6,7,8};
Curve Loop(20) = {50,60,70,80};
Curve Loop(21) = {51,61,71,81};


Plane Surface(1) = {1, 2, 20, 21};

Physical Surface(100) = {1};

Physical Line (111) = {1} ;
Physical Line (112) = {2} ;
Physical Line (113) = {3} ;
Physical Line (114) = {4} ;

Physical Curve (115) = {5,6,7,8,50,60,70,80,51,61,71,81} ;

Physical Point (200) = {1,2};





Mesh.MshFileVersion = 2.2;

