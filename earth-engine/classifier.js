// Classification and CV scheme
var all_layers= goodComposite.addBands(
  edgeBuffer_NDVI.select(['NDVI'],['eb_ndvi'])).addBands(
    edgeBuffer_Pan.select(['B8'],['eb_pan'])).addBands(
      ndvi);


//Band list : 'B5', 'B4', 'B3','B2','evi','se_evi', 'se2_b3','se2_b4', 'se2_b5',
//            'se3_b3','se3_b4', 'se3_b5', 'gauss_b3','gauss_b4','gauss_b5'

var bands = ['B5', 'B4', 'B3','B2','evi','se_evi', 'se2_b3','se2_b4', 'se2_b5',
             'se3_b3','se3_b4', 'se3_b5', 'gauss_b3','gauss_b4','gauss_b5'];

// create image collection for the classifier
var bands =['B1','B2','B3', 'B4','B5','B6_VCID_2','B7','eb_ndvi', 'eb_pan', 'NDVI','B8'];
var farm_points= all_layers.select(bands).sampleRegions({
  collection:farm_polys,
  properties:['farm'],
  scale:30
});

var nf_points= all_layers.select(bands).sampleRegions({
  collection:notfarm_polys,
  properties:['farm'],
  scale:30
});

var all_points = nf_points.randomColumn('x',12412).filter(ee.Filter.lte('x',0.02)).merge(farm_points);



//subsample data into training and testing (second arg seed)
var temp_points = all_points.randomColumn('x', 234543);
var testing = temp_points.filter(ee.Filter.lte('x',0.1));
var training= temp_points.filter(ee.Filter.gt('x',0.1));


// Make a Random Forest classifier and train it.
var classifier = ee.Classifier.randomForest().train({
  features: training,
  classProperty: 'farm',
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

//Map.addLayer(classified, {min: 0, max: 1, palette: palette}, 'Land Use Classification');

// Confusion matrix on training data
print('RF error matrix: ', classifier.confusionMatrix());
print('RF accuracy: ', classifier.confusionMatrix().accuracy());

// Get a confusion matrix on test data
var testAccuracy = classified_test.errorMatrix('farm', 'classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy());
