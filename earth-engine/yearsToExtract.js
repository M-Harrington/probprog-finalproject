// Years + months compiled to extract data over TODO: make proper dates
var collection = ee.ImageCollection("LANDSAT/LE07/C01/T1")
  .filterDate('2003-01-01', '2003-05-01')
  .filterBounds(geometry);
 
 Map.addLayer(goodComposite,
             {bands: ['B3', 'B2', 'B1'], gain: 3.5},
             'good composite');
             


// Export the image, specifying scale and region.
Export.image.toDrive({
  image: classified,
  description: 'imageToDriveTest',
  scale: 30,
  region: geometry
});
