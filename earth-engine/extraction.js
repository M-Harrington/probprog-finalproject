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



             


// Export the image, specifying scale and region.
Export.image.toDrive({
  image: classified.visualize({min: 0, max: 1, palette: palette}),
  description: 'imageToDriveTest',
  scale: 30,
  'crs':'EPSG:4326',
  region: roi
});
