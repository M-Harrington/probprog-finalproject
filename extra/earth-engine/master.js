/////Script to run entire proceedure 
 

Map.setCenter(71.48545013599528, 26.203854380719925, 12); 
 
/////////////////////////
// Define ROI

var roi = 
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

/////////////////////////
// Define training and testing sets
// Currently only sampled from '2011-01-01', '2011-2-26'

var Farmland = /* color: #ffc82d */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[71.35648650317216, 26.4929586452491],
                  [71.35515612750078, 26.49178717181977],
                  [71.35680836825395, 26.490538867322908],
                  [71.3580958285811, 26.49180637639847]]]),
            {
              "farm": 1,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.30905475120858, 26.48425602223461],
                  [71.3106426189454, 26.48550439494534],
                  [71.30939807396248, 26.48644546695275],
                  [71.30804624061898, 26.4852931328239]]]),
            {
              "farm": 1,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.28147873710691, 26.491994110079077],
                  [71.28293785881101, 26.493165581399573],
                  [71.28167185615598, 26.494317836625854],
                  [71.28010544609128, 26.49314637704794]]]),
            {
              "farm": 1,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.27481974417105, 26.491082270521833],
                  [71.27698696905509, 26.493482832714488],
                  [71.27477682882682, 26.4948655337838],
                  [71.27258814627066, 26.492753067111256]]]),
            {

              "farm": 1,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.65285879079534, 26.597867686495793],
                  [71.64976888601018, 26.595450120883594],
                  [71.65303045217229, 26.59307087946561],
                  [71.65633493367864, 26.595258248409337]]]),
            {

              "farm": 1,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.69029138005226, 26.56136945447393],
                  [71.69194362080543, 26.56321196209423],
                  [71.69042012608497, 26.564248359612957],
                  [71.68893954670875, 26.56273214525404]]]),
            {
                    
              "farm": 1,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.71275586347076, 26.493312081050046],
                  [71.71018094281646, 26.49023934789631],
                  [71.71185464124176, 26.48893341143628],
                  [71.71430081586334, 26.491583678757014]]]),
            {
                    
              "farm": 1,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.31091469968317, 26.26757210774967],
                  [71.3121807023382, 26.268264813711074],
                  [71.30964869702814, 26.27059304512095],
                  [71.30879039014337, 26.26997731904183]]]),
            {
                    
              "farm": 1,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.35766004323227, 26.213672010882117],
                  [71.35643695592148, 26.212728717137367],
                  [71.35819648503525, 26.211689160072673],
                  [71.35918353795273, 26.212459203234467]]]),
            {
                    
              "farm": 1,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.30072164289868, 26.084687887920964],
                  [71.29939126722729, 26.0839555456999],
                  [71.30059289686596, 26.082143942622903],
                  [71.30183744184887, 26.083184654200814]]]),
            {
                    
              "farm": 1,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.52515838305897, 25.995671411267292],
                  [71.52369926135486, 25.99489993919339],
                  [71.52481506030506, 25.99339555408031],
                  [71.52623126666492, 25.994051314061686]]]),
            {
                    
              "farm": 1,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.51301334063953, 25.996982902167808],
                  [71.51438663165516, 25.99775436056492],
                  [71.51142547290272, 26.00130300394389],
                  [71.51030967395252, 26.000454425061797]]]),
            {
                    
              "farm": 1,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.5007749480477, 25.998704004581604],
                  [71.49948748772056, 25.997739688588435],
                  [71.5004316252938, 25.99669821842554],
                  [71.4987579268685, 25.99554101852809],
                  [71.49978789513023, 25.99469239802646],
                  [71.50274905388267, 25.997083949191946]]]),
            {
                    
              "farm": 1,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.4579965502918, 26.125344814503634],
                  [71.45421999999883, 26.123341195351227],
                  [71.4565374285877, 26.12172286248966],
                  [71.45971316406133, 26.123803572051642]]]),
            {
                    
              "farm": 1,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.43533724853398, 26.11995037697014],
                  [71.43190402099492, 26.117869598792446],
                  [71.43353480407598, 26.115788783565286],
                  [71.43679637023808, 26.118177864564068]]]),
            {
                    
              "farm": 1,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.44289034911992, 26.10738807877321],
                  [71.44452113220098, 26.10846710216972],
                  [71.44289034911992, 26.109931475136154],
                  [71.44014376708867, 26.108390029400233],
                  [71.44189932734218, 26.106711595846487]]]),
            {
                    
              "farm": 1,
              "system:index": "15"
            })]),
    NotFarmland = /* color: #00ffff */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
                [[[71.35059680443123, 26.493263207576135],
                  [71.35205592613534, 26.494396257680965],
                  [71.35102595787362, 26.49501078865957],
                  [71.34980287056283, 26.493877744612245]]]),
            {
                    
              "farm": 0,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.31077136497811, 26.482661926610636],
                  [71.31263818245247, 26.484179198855927],
                  [71.31102885704354, 26.48569645108256],
                  [71.30909766655282, 26.484198404705396]]]),
            {
                    
              "farm": 0,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.28137144874631, 26.49118751632877],
                  [71.27997670005857, 26.489900795536204],
                  [71.28190789054929, 26.48847962449145],
                  [71.28338846992551, 26.489900795536204]]]),
            {
                    
              "farm": 0,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.3168108524851, 26.50647067553291],
                  [71.31595254560034, 26.503878359182682],
                  [71.31824851651709, 26.503244672956402],
                  [71.3188278736643, 26.505779396891988]]]),
            {
                    
              "farm": 0,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.6531388869953, 26.66208861892964],
                  [71.65127206952093, 26.660746287842358],
                  [71.65388990551946, 26.658253345360233],
                  [71.6560571304035, 26.660055939991878]]]),
            {
                    
              "farm": 0,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.69484489656111, 26.661585731893727],
                  [71.69589632249495, 26.660415981961485],
                  [71.69724815583845, 26.66137479426754],
                  [71.69604652619978, 26.66262123822126]]]),
            {
                    
              "farm": 0,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.35321505503725, 26.283317514835424],
                  [71.34935267405581, 26.2809318196804],
                  [71.35209925608706, 26.27777647010029],
                  [71.35587580638003, 26.28023918931801]]]),
            {
                    
              "farm": 0,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.35575031041367, 26.211073121874776],
                  [71.35768150090439, 26.209282742353047],
                  [71.35946248769028, 26.210322320912294],
                  [71.35673736333115, 26.212054931209824]]]),
            {
                    
              "farm": 0,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.34597146248609, 26.214808823294725],
                  [71.34453379845411, 26.213576776666],
                  [71.34672248101026, 26.21186344014064],
                  [71.34818160271436, 26.213230261201396]]]),
            {
                    
              "farm": 0,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.28844785444653, 26.08676925658128],
                  [71.2870745634309, 26.085998383615845],
                  [71.29080819837964, 26.080486493969598],
                  [71.29243898146069, 26.081874126999583]]]),
            {
                    
              "farm": 0,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.30608606092846, 26.084302445217933],
                  [71.30363988630688, 26.082683572003578],
                  [71.30565690748608, 26.080486493969598],
                  [71.30788850538647, 26.082066852508277]]]),
            {
                    
              "farm": 0,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.60715862465372, 25.938750864008984],
                  [71.60346790504923, 25.936744055386782],
                  [71.6072444553422, 25.93226720526867]]]),
            {
                    
              "farm": 0,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.46510842796886, 25.856499961462895],
                  [71.45995858666026, 25.850629719750764],
                  [71.46356347557628, 25.848775898664385],
                  [71.46682504173839, 25.853642117034077]]]),
            {
                    
              "farm": 0,
              "system:index": "12"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.48043686946016, 25.816777911613357],
                  [71.47683198054415, 25.81260558008981],
                  [71.48232514460665, 25.809205793889028],
                  [71.48524338801485, 25.814305436600414]]]),
            {       
              "farm": 0,
              "system:index": "13"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.39032974726547, 25.717383293585666],
                  [71.38861313349594, 25.719857788321363],
                  [71.38681068903793, 25.718929858827742],
                  [71.38784065729965, 25.716184691693297]]]),
            {
              "farm": 0,
              "system:index": "14"
            }),
        ee.Feature(
            ee.Geometry.Polygon(
                [[[71.58172803539014, 26.130210088377712],
                  [71.57953935283399, 26.128977147395307],
                  [71.58026891368604, 26.12747448297686],
                  [71.58250051158643, 26.128861558510952]]]),
            {
              "farm": 0,
              "system:index": "15"
            })]);

/////////////////////////
// Bring in basic datsets
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

var filter_date = ['2011-01-01', '2011-12-31']

var dataset = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
                  .filterDate(filter_date[0], filter_date[1])
                  .map(cloudMaskL457);
                  
                  var visParams = {
  bands: ['B5', 'B4', 'B3'],
  min: 1200,
  max: 4000,
  gamma: .6,
};  // Above params close to matching stretch to 90% (can play with these)


var dataset_med = dataset.median();
//var dataset_med = dataset_med.reproject('EPSG:4326', null, 1); //prevent GEE scaling



// Compute the EVI using an expression.
var evi = dataset_med.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': dataset_med.select('B4'),
      'RED': dataset_med.select('B3'),
      'BLUE': dataset_med.select('B1')
});



/////////////////////////
// Convolutions
// Canny Edge Detection
var edge1 = ee.Algorithms.CannyEdgeDetector(dataset_med, 1000,1.5); //more smoothing (large features)
var edge2 = ee.Algorithms.CannyEdgeDetector(dataset_med, 500,2); //less smoothing (more features)
var edge3 = ee.Algorithms.CannyEdgeDetector(evi, 0.35,2); 

//// Low pass filters
// smoothed EVI edges
var boxcar = ee.Kernel.square({
  radius: 180, units: 'meters', normalize: true 
});
var smooth_edge_evi = edge3.convolve(boxcar);

// smoothed all edges (less features canny, all bands)
var boxcar = ee.Kernel.square({
  radius: 210, units: 'meters', normalize: true 
});
var smooth_edge1 = edge1.convolve(boxcar);

// smoothed all edges (more features canny, all bands)
var boxcar = ee.Kernel.square({
  radius: 120, units: 'meters', normalize: true 
});
var smooth_edge2 = edge2.convolve(boxcar);

//Gaussian 
var gaus = ee.Kernel.gaussian({
  radius: 180, sigma:90, units: 'meters', normalize: true 
});
var gauss_smooth = dataset_med.convolve(gaus);


/////////////////////////
// Classifier
// Choosing best pixels from Landsat 5 imagery over filter date

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
//var nf_small = nf_points.randomColumn('x',12412).filter(ee.Filter.lte('x',0.02));
//var farm_small = farm_points.randomColumn('x',12412).filter(ee.Filter.lte('x',0.02))

//var all_points = nf_small.merge(farm_small);


// Merge the points together
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



/////////////////////////
// (optional) Visualizations 
// base dataset
Map.addLayer(roi);
Map.addLayer(dataset_med, visParams);
Map.addLayer(evi,{min: -1, max: 1});

// Edge detection
var visParams_can = {bands: ['B5', 'B4', 'B3']};  
Map.addLayer(edge1,visParams_can);
Map.addLayer(edge2,visParams_can);
Map.addLayer(edge3); 

// Convolution Layers
Map.addLayer(smooth_edge_evi);
Map.addLayer(smooth_edge1,visParams_can);
Map.addLayer(smooth_edge2,visParams_can);
Map.addLayer(gauss_smooth,visParams_can);

// Classified layer
Map.addLayer(classified, {min: 0, max: 1, palette: palette}, 'Land Use Classification');
