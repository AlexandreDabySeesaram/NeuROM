L=10;

//dh = L; // Minimum size is the length of the beam
Mesh.Algorithm = 6;

// Define the square geometry
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, 2*L, 0};
Point(4) = {0, 2*L, 0};

// Define lines connecting the points to form the square
Line(6) = {1, 2};
Line(7) = {2, 3};
Line(8) = {3, 4};
Line(9) = {4, 1};

// Define the surface by creating a loop from the lines
Line Loop(10) = {6,7,8,9};

// Define the surface
Plane Surface(11) = {10};
// Recombine Surface{1};

Physical Surface(100) = {11};

Physical Line (111) = {6} ;
Physical Line (112) = {7} ;
Physical Line (113) = {8} ;
Physical Line (114) = {9} ;

Physical Point (200) = {1,2,3,4};



Mesh.MshFileVersion = 2.2;

