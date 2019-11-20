//Define region of interest
// Create a geometry representing an export region (near Jaisalmer).
var geometry = /* color: #00ffff */ee.Geometry.Polygon(
        [[[70.75894231129689, 25.54670603185635],
          [71.17642278004689, 24.85085434858454],
          [71.82461613942189, 25.139617951574408],
          [72.51675481129689, 25.278777563996034],
          [73.37368840504689, 25.487217850980564],
          [73.75820988942189, 26.04128869562203],
          [74.16470403004689, 26.53379405484255],
          [74.64810246754689, 26.91648844655037],
          [75.17544621754689, 27.31741344387549],
          [75.81265324879689, 28.318214077961912],
          [75.40615910817189, 28.58867210968267],
          [75.28530949879689, 29.012273919059222],
          [75.02163762379689, 29.041093051127362],
          [74.71402043629689, 29.319262408067498],
          [74.60415715504689, 29.127502148519604],
          [74.37344426442189, 29.127502148519604],
          [73.64834660817189, 29.165882911008435],
          [72.36294621754689, 28.694733850850596],
          [72.14321965504689, 28.221454556198225],
          [71.91250676442189, 27.90152644473874],
          [71.42910832692189, 27.823824965717016],
          [71.05557317067189, 27.746067833136603],
          [70.70401067067189, 27.648793312241942],
          [70.49527043629689, 27.920943106548464],
          [70.31948918629689, 27.852969547792586],
          [69.89102238942189, 27.40522860895214],
          [69.61636418629689, 26.96545856024489],
          [69.83609074879689, 26.710582139385473],
          [70.22061223317189, 26.671319537640652],
          [70.16568059254689, 26.42562252048915],
          [70.27554387379689, 25.814036981332112],
          [70.70401067067189, 25.744786582647283]]]); 

//To view exactly where
Map.centerObject(geometry);  
Map.addLayer(geometry, {color: 'FF0000'}, 'geodesic polygon');

// Load Landsat 5 data, filter by date and bounds.
var collection = ee.ImageCollection("LANDSAT/LE07/C01/T1")
  .filterDate('2003-01-01', '2003-05-01')
  .filterBounds(geometry);

// Also filter the collection by the IMAGE_QUALITY property.
var filtered = collection
  .filterMetadata('IMAGE_QUALITY', 'equals', 9);

// Create two composites to check the effect of filtering by IMAGE_QUALITY.
var goodComposite = ee.Algorithms.Landsat.simpleComposite(filtered, 75, 3);

// Display the composites.
Map.addLayer(goodComposite,
             {bands: ['B3', 'B2', 'B1'], gain: 3.5},
             'good composite');


// Compute the Normalized Difference Vegetation Index (NDVI).
var nir = goodComposite.select('B5');
var red = goodComposite.select('B4');
var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

// Display the result.
var ndviParams = {min: -1, max: 1, palette: ['green', 'white', 'blue']}; //colors maybe need playing with
Map.addLayer(ndvi, ndviParams, 'NDVI image');

// Add PC band
var Panchromatic = goodComposite.select(['B8']);
Map.addLayer(Panchromatic,
             {gain: 1.5},
             'Panchromatic');

//////////////////////////Edge Detection///////////////////////

// Detect edges in the panchromatic composite.
var canny = ee.Algorithms.CannyEdgeDetector(Panchromatic, 12, 2);

// Mask the image with itself to get rid of areas with no edges.
canny = canny.updateMask(canny);
Map.addLayer(canny, {min: 0, max: 1, palette: 'FF0000'}, 'CE Pan');



////////////////////
// Detect edges in the NDVI composite.
var canny1 = ee.Algorithms.CannyEdgeDetector(ndvi, .1, 5.5);

// Mask the image with itself to get rid of areas with no edges.
canny1 = canny1.updateMask(canny1);
Map.addLayer(canny1, {min: 0, max: 1, palette: '3498DB'}, 'CE NDVI');

// Create buffers around NDVi and Pan bands
var bufferSize = 80 // in meters
var edgeBuffer_NDVI = canny1.focal_max(bufferSize, 'square', 'meters'); 
Map.addLayer(edgeBuffer_NDVI.updateMask(edgeBuffer_NDVI),{palette: 'FF0000'},'BE NDVI');

var bufferSize = 65 // in meters
var edgeBuffer_Pan = canny.focal_max(bufferSize, 'square', 'meters'); 
Map.addLayer(edgeBuffer_Pan.updateMask(edgeBuffer_Pan),{palette: 'FF0000'},'BE Pan');



//Overview
//Map.setCenter(72.2595654822021, 27.14173618881201, 7);
//crowded place
//Map.setCenter(72.2595654822021, 27.14173618881201, 11);
//middle of nowhere
//Map.setCenter(72.06070709101755,27.140570645200075,14)
