//For counting the number of correctly classified pixels within districts
//To be added: district shape file importing
/////////////

//Add in pre-loaded table (remove those in namelist)
var namelist = ['Bharatpur' ,'Karauli', 'Dhalpur', 'Sawai Madhopur','Bundi', 'Tonk', 'Kota', 'Jhalawar', 'Chittaurgarh', 'Udaipur','Dungarpur', 'Baran','Pratapgarh' , 'Banswara', 'Rajsamand', 'Sirohi','Dhaulpur', 'Dausa', 'Alwar'];
var rajDist = ee.FeatureCollection("users/mrh2182/DISTRICT_11").filter(ee.Filter.eq('STATE_UT','Rajasthan')).filter(
  ee.Filter.inList('DISTRICT',namelist).not());

var classified_mask = classified.updateMask(classified);

var pixPerDist = classified_mask.reduceRegions({
  collection: rajDist,
  reducer: ee.Reducer.sum(),
  scale: 30,
});

Export.table.toDrive({
  collection: pixPerDist,
  description: 'pixPerDist_trial',
  fileFormat: 'CSV'
});




//Note the above borrowed from https://gis.stackexchange.com/questions/253164/count-the-number-of-pixel-identified-as-water-from-a-collection-of-landsat-image
//This might be helpful too: https://gis.stackexchange.com/questions/312568/counting-number-of-pixels-with-negative-ndvi-value-within-polygon-using-google-e
//This might be the best: https://developers.google.com/earth-engine/reducers_reduce_region
