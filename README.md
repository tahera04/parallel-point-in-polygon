# parallel-point-in-polygon
Parallel Point-in-Polygon Classification for Large-Scale Geospatial Data 

## Data Structures

Point:
- x, y coordinates

BoundingBox:
- min_x, max_x
- min_y, max_y

Polygon:
- id
- outer boundary (list of points)
- holes (list of polygons)
- bounding box

## File Format

### polygons.txt

Each polygon is defined as:

polygon_id outer_vertex_count hole_count  
x y  
x y  
... (outer vertices)

(For each hole:)
hole_vertex_count  
x y  
x y  
... (hole vertices)

---

### Example:

1 4 1  
0 0  
10 0  
10 10  
0 10  

4  
3 3  
7 3  
7 7  
3 7  

---

Explanation:
- Polygon ID = 1
- Outer boundary has 4 points
- There is 1 hole
- The hole also has 4 points