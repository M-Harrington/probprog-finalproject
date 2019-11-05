// Typical farm size is maybe 4-8 pixels across, low pass filter can be pretty local then
// Note: radii for kernels must be meters

///////////////////////Canny//////////////////////////
//probably want to give ED subset of relevant bands like G R NIR SWIR
var edge1 = ee.Algorithms.CannyEdgeDetector(dataset_med, 1000,1.5); //more smoothing (large features)
var edge2 = ee.Algorithms.CannyEdgeDetector(dataset_med, 500,2); //less smoothing (more features)
var edge3 = ee.Algorithms.CannyEdgeDetector(evi, 0.35,2); 

// (Optional) Visualization
var visParams_can = {bands: ['B5', 'B4', 'B3']};  
Map.addLayer(edge1,visParams_can);
Map.addLayer(edge2,visParams_can);
Map.addLayer(edge3); 

//////////////////////Low-Passes//////////////////////
//Local averages 3x3 or so, maybe 4x4 gaussian

// smoothed EVI edges
var boxcar = ee.Kernel.square({
  radius: 180, units: 'meters', normalize: true 
});
var smooth_edge_evi = edge3.convolve(boxcar);

// smoothed all edges (less features canny, all bands)
var boxcar = ee.Kernel.square({
  radius: 210, units: 'meters', normalize: true 
});
var smooth_edge1 = edge1.convolve(boxcar);

// smoothed all edges (more features canny, all bands)
var boxcar = ee.Kernel.square({
  radius: 120, units: 'meters', normalize: true 
});
var smooth_edge2 = edge2.convolve(boxcar);

//Gaussian 
var gaus = ee.Kernel.gaussian({
  radius: 180, sigma:90, units: 'meters', normalize: true 
});
var gauss_smooth = dataset_med.convolve(gaus);

// (Optional) Visualization
Map.addLayer(smooth_edge_evi);
Map.addLayer(smooth_edge1,visParams_can);
Map.addLayer(smooth_edge2,visParams_can);
Map.addLayer(gauss_smooth,visParams_can);

