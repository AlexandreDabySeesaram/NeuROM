L=10;
dh = L; // Minimum size is the length of the beam
Mesh.Algorithm = 8;

// Define the square geometry
Point(1) = {0, 0, 0, dh};
Point(2) = {L, 0, 0, dh};
Point(3) = {L, L, 0, dh};
Point(4) = {0, L, 0, dh};

// Define lines connecting the points to form the square
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define the surface by creating a loop from the lines
Line Loop(1) = {1, 2, 3, 4};

// Define the surface
Plane Surface(1) = {1};
// Recombine Surface{1};

Physical Surface(100) = {1};


Mesh.MshFileVersion = 2.2;

