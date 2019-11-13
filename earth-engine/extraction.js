//Dates to create:
//pomk= August-Nov; pomr= Nov-Jan; prm= Jan-May; mo= May-August;
//pomk= 08-15,11-15; pomr= 11-15,01-15; prm= 01-15,05-15; mo= 05-15,08-15;
//17 periods

//Create all dates to extract
var pomk= ["08-15","11-14"]; var pomr= ["11-15","01-14"]; var prm= ["01-15","05-14"]; var mo= ["05-15","08-14"];
var season_list = [pomr, prm, mo, pomk];

var i; var extract_dates = new Array(4*17); 
for (i = 0; i < 4*17; i++) {
  var season = i%4;
  var year = Math.floor(i / 4) + 1996;
  
  if (season==0){
    extract_dates[i] = [(year-1).toString()+ "-" + season_list[season][0] ,year.toString() + "-" + season_list[season][1]];
  }
  else {
      extract_dates[i] = [year.toString()+ "-" + season_list[season][0] ,year.toString() + "-" + season_list[season][1]];
  }
}

for (i = 0; i < 4*17; i++) {
  ///////////Classify current year's image//////////////
  var dataset = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
                    .filterDate(extract_dates[i][0], extract_dates[i][1])
                    .map(cloudMaskL457);

  var dataset_med = dataset.median();
  //var dataset_med = dataset_med.reproject('EPSG:4326', null, 1); //prevent GEE scaling

  // Compute the EVI using an expression.
  var evi = dataset_med.expression(
      '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': dataset_med.select('B4'),
        'RED': dataset_med.select('B3'),
        'BLUE': dataset_med.select('B1')
  });

  /////////////////////////
  // Convolutions (extract)
  // Canny Edge Detection
  var edge1 = ee.Algorithms.CannyEdgeDetector(dataset_med, 1000,1.5); //more smoothing (large features)
  var edge2 = ee.Algorithms.CannyEdgeDetector(dataset_med, 500,2); //less smoothing (more features)
  var edge3 = ee.Algorithms.CannyEdgeDetector(evi, 0.35,2); 

  //// Low pass filters
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


  /////////////////////////
  // Classifier
  // Choosing best pixels from Landsat 5 imagery over filter date

  // Classification and CV scheme
  var all_layers= dataset_med.addBands(
    smooth_edge_evi.select(['constant'],['se_evi'])).addBands(
    smooth_edge1.select(['B5','B4','B3'],['se1_b5','se1_b4','se1_b3'])).addBands(
    smooth_edge2.select(['B5','B4','B3'],['se2_b5','se2_b4','se2_b3'])).addBands(
    gauss_smooth.select(['B5','B4','B3'],['gauss_b5','gauss_b4','gauss_b3'])).addBands(
    evi.select(['constant'],['evi']));

  // create image collection for the classifier (last two are satellite characteristics)
  var bands = ['B5', 'B4', 'B3','B2','evi','se_evi', 'se2_b3','se2_b4', 'se2_b5',
               'se1_b3','se1_b4', 'se1_b5', 'gauss_b3','gauss_b4','gauss_b5','pixel_qa','radsat_qa'];

  //Perform classification
  // Classify the input imagery.
  var classified = all_layers.select(bands).classify(classifier);
  var evi_mask = evi.updateMask(classified);
  
  
  //export current year with correct name
  // Export the image, specifying scale and region.
  Export.image.toDrive({
    image: classified.visualize({min: 0, max: 1, palette: palette}),
    description: ('classification'+extract_dates[i][0]),
    scale: 150,
    'crs':'EPSG:4326',
    region: roi
  });

  // Export the image, specifying scale and region.
  Export.image.toDrive({
    image: evi_mask.visualize({min: -1, max: 1}),
    description: 'evi_mask'+extract_dates[i][0],
    scale: 150,
    'crs':'EPSG:4326',
    region: roi
  });

}



             

//Note: 30m takes 16 minutes to run, 500m takes 1 minute to run
// 90m and 150m both took only 5 minutes to run (1 vs 2 actual runtime)

//Currently have 17*4*2 image files


