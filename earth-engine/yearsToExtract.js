// 2003 <---- already have (_trial)
// 2002 <---- One funny tile
// 2001 <---- Small amount of clouds
// 2000 <---- Looks fine 
// 2004+ <---- Poor quality for most images


// Years + months compiled to extract data over

var collection = ee.ImageCollection("LANDSAT/LE07/C01/T1")
  .filterDate('2003-01-01', '2003-05-01')
  .filterBounds(geometry);
 
 Map.addLayer(goodComposite,
             {bands: ['B3', 'B2', 'B1'], gain: 3.5},
             'good composite');
             
             
