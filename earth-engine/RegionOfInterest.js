// Describes a potential area to focus on

var table = ee.FeatureCollection("users/mrh2182/well_point"),
    geometry = 
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
          
// Map and visualize
Map.addLayer(table);
Map.addLayer(geometry);
