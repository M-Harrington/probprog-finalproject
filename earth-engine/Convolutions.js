// Typical farm size is maybe 4-8 pixels across, low pass filter can be pretty local then
// Note: radii for kernels must be meters

//probably want to give ED subset of relevant bands like G R NIR SWIR
var edge1 = ee.Algorithms.CannyEdgeDetector(image, ); //less smoothing (more features)
var edge2 = ee.Algorithms.CannyEdgeDetector(image, ); //more smoothing (large features)

//Local averages 3x3 or so, maybe 4x4 gaussian

//Convolution average on edge detection alg
