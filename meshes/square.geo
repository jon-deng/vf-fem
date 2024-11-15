// Create a unit square mesh

// Create 4 vertices clockwise starting from the lower left corner
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};

// Create the 4 edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Create the edge loop
Curve Loop(1) = {1, 2, 3, 4};

// Create the volume (surface for 2D)
Plane Surface(1) = {1};

// Mark Physical entities

// Mark the top left and top right points as "inferior" and "superior"
// Physical Point("inferior") = {4};
// Physical Point("superior") = {3};

// Mark the bottom, right, top, and left surfaces
Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("top") = {3};
Physical Curve("left") = {4};

Physical Curve("dirichlet") = {1};
Physical Curve("neumann") = {2, 3, 4};

// Mark the plane surface
Physical Surface("volume") = {1};
