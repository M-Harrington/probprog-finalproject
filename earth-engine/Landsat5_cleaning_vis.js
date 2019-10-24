// Choosing best pixels from Landsat 5 imagery over filter date

var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};

var dataset = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
                  .filterDate('2011-01-01', '2011-12-31')
                  .map(cloudMaskL457);
                  
                  var visParams = {
  bands: ['B5', 'B4', 'B3'],
  min: 0,
  max: 4000,
  gamma: 1.4,
};  // Worth going into settings under layers to change stretch to 90% for nice visualization

Map.setCenter(71.48545013599528, 26.203854380719925, 8);
Map.addLayer(dataset.median(), visParams);
