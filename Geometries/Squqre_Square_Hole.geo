lc = 5;
L = 10;
Point(1) = {-2*L,-L,0,lc};
Point(2) = {-2*L,1*L,0,lc};
Point(3) = {0,-1*L,0,lc};
Point(4) = {0,1*L,0,lc};
Point(5) = {-3*L,-2*L,0,lc};
Point(6) = {-3*L,2*L,0,lc};
Point(7) = {1*L,-2*L,0,lc};
Point(8) = {1*L,2*L,0,lc};
Line(1) = {6,8};
Line(2) = {8,7};
Line(3) = {7,5};
Line(4) = {5,6};
Line(5) = {2,1};
Line(6) = {1,3};
Line(7) = {3,4};
Line(8) = {4,2};
Line Loop(1) = {5,6,7,8}; // interior loop
Line Loop(2) = {1,2,3,4}; // exterior loop
//Plane Surface(1) = {1}; // interior surface
Plane Surface(2) = {2,1}; // exterior surface (with a whole)
//Plane Surface(2) = {2}; // exterior surface (with a whole)
Physical Surface(100) = {2};

Physical Line (111) = {3} ;
Physical Line (112) = {4} ;
Physical Line (113) = {1} ;
Physical Line (114) = {2} ;
Physical Line (115) = {5,6,7,8} ;



Mesh.MshFileVersion = 2.2;