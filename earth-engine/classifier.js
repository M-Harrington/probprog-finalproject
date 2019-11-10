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



var farm_points= all_layers.select(bands).sampleRegions({
  collection:Farmland,
  properties:['farm'],
  scale:30
});

var nf_points= all_layers.select(bands).sampleRegions({
  collection:NotFarmland,
  properties:['farm'],
  scale:30
});


//Decrease size of training/testing points for computational times (not clear if still needed)  
var all_points = nf_points.merge(farm_points);



//subsample data into training and testing (second arg seed)
var temp_points = all_points.randomColumn('x', 234543);
var testing = temp_points.filter(ee.Filter.lte('x',0.1));
var training= temp_points.filter(ee.Filter.gt('x',0.1));


// Make a Random Forest classifier and train it.
var classifier = ee.Classifier.randomForest().train({
  features: training,
  classProperty: 'farm',    ////This actually might be necessary, which means would have to change some of the imports
  inputProperties: bands
});

// Classify the input imagery.
var classified = all_layers.select(bands).classify(classifier);
var classified_test =testing.classify(classifier); 

// Define a palette for the Land Use classification.
var palette = [
  '0000FF', // water (1)  // blue
  '008000' //  forest (2) // green
];

// Confusion matrix on training data
print('RF error matrix: ', classifier.confusionMatrix());
print('RF accuracy: ', classifier.confusionMatrix().accuracy());

// Get a confusion matrix on test data
var testAccuracy = classified_test.errorMatrix('farm', 'classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy());

//Map.addLayer(classified, {min: 0, max: 1, palette: palette}, 'Land Use Classification');
