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
  min: 1200,
  max: 4000,
  gamma: .6,
};  // Above params close to matching stretch to 90% (can play with these)

Map.setCenter(71.48545013599528, 26.203854380719925, 9);
Map.addLayer(dataset.median(), visParams);



// Region of Interest
var geometry = 
    /* color: #98ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[71.18575854846165, 26.77769880073871],
          [71.18575854846165, 25.65411556477628],
          [71.87240405627415, 25.65411556477628],
          [71.87240405627415, 26.77769880073871]]], null, false);
          
Map.addLayer(geometry);
