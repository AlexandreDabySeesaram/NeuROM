L=10;
dh = L; // Minimum size is the length of the beam
Point(1) = {0,0,0,dh};
Point(2) = {L,0,0,dh}; 
Line(3) = {1,2};
Physical Line(100) = {3};
Physical Point(3) = {1};
Physical Point(4) = {2};

// export
Mesh.MshFileVersion = 2.2;
