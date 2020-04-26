Geometry.OCCTargetUnit = "CM";
Merge "geometries/M5-3layers-cm.STEP";

lc_fine = 0.01;
lc_medi = 0.1;
lc_coar = 0.1;

Physical Surface("cover") = {2};
Physical Surface("slp") = {3};
Physical Surface("body") = {1};
Physical Curve("fixed") = {14, 20, 1, 19, 8};
Physical Curve("pressure") = {13, 12, 11, 10, 9};

Characteristic Length {11} = lc_fine;
Characteristic Length {12} = lc_fine;
Characteristic Length {13} = lc_fine;
Characteristic Length {10} = lc_fine;

Characteristic Length {14} = lc_coar;
Characteristic Length {9} = lc_coar;

Field[1] = Distance;
Field[1].NodesList = {11, 12, 13, 10};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_fine;
Field[2].LcMax = lc_coar;
Field[2].DistMin = 0.0;
Field[2].DistMax = 0.2;

Background Field = 2;

OptimizeMesh "Gmsh";

Mesh.Smoothing = 50;

