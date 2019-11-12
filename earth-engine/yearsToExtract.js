//Dates to create:
//pomk= August-Nov; pomr= Nov-Jan; prm= Jan-May; mo= May-August;
//pomk= 08-15,11-15; pomr= 11-15,01-15; prm= 01-15,05-15; mo= 05-15,08-15;
//14 periods

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
  region: roi
});
