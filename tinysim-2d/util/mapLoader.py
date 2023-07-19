import numpy as np
import cv2
import yaml
import os

def contours2points(contours, mapMetadata, mapImg):
    """Transforms contours around walls in map to points in world coordinates
    
    Parameters
    ----------
    contours: list
            List of contours, each depicted by pixels (as returned by opencv.findContours)
    mapMetadata: dict
            Metadata found in map.yaml
    mapImg: array_like
            Thresholded map image
    
    Returns
    -------
    retContours: list
            List containing (n,2) np.float32 arrays, which depict points on contours
    """
    # Track map origin and create placeholder for transformed contours
    origin2d = np.array(mapMetadata["origin"][:2], dtype=np.float32)
    cellSize2d = np.full((2), mapMetadata["resolution"], dtype=np.float32)
    retContours = []
    
    # Pad map to not require checks when applying the kernel
    paddedShape = np.array(mapImg.shape) + np.array([2,2])
    paddedMap = np.full(paddedShape, 255)
    paddedMap[1:-1, 1:-1] = mapImg
    
    for c, contour in enumerate(contours):
        # Placeholder for result
        correctedContour = []
        # Correct each coordinate in contour towards edge between contour and wall pixel
        for i, point in enumerate(contour):
            # Store coordinates of point in contour
            y, x = np.array(point).ravel()
            
            # Make sure check only occurs up to second to last entry
            if i+1 < len(contour): ynext, xnext = np.array(contour[i+1]).ravel()
            else: ynext, xnext = y, x
            
            ## Correct position of external corners (i.e. two neighbouring contour points)
            if np.abs(y - ynext) == 1 and np.abs(x - xnext) == 1:
                # Check if point is diagonally next to the next point. If they are, average out their positions
                y = (y + ynext)/2
                x = (x + xnext)/2
                correctedCoords = np.array([x, y], dtype=np.float32)
                
            ## Correct position of single contour points (i.e. no external corners) 
            else:
                # Create kernel to determine edge of geometry
                kernel = np.array([
                    [255,               paddedMap[x, y+1],   255                ],
                    [paddedMap[x+1, y], 255,                 paddedMap[x+1, y+2]],
                    [255,               paddedMap[x+2, y+1], 255                ]
                ])
                # Lookup coordinates of detected wall and move the contour over by half a pixel
                coords = np.asarray(np.where(kernel == 0)).T
                coords = np.subtract(coords, np.ones_like(coords))/2
                correctedCoords = np.sum(np.vstack((coords, np.array([x, y]))), axis=0)
            
            # Rotate to match ROS convention
            # Transform point to world coordinates and store resulting point
            correctedContour.append(
                correctedCoords*mapMetadata["resolution"] + origin2d + (cellSize2d/2)
            )
        
        # Store resulting contour points       
        retContours.append(np.array(correctedContour))
        
    return retContours

def process2DPoints(contourPoints):
    """Transforms a list 2D points to a mesh by mirroring the points along the z-axis
    
    Parameters
    ----------
    contourPoints: array_like
            (n,2) np.float32 array containing points of the contour in world coordinates
    
    Returns
    -------
    mapVertices: array_like
            (n,3) np.float32 array containing vertex points in world coordinates for the raycasting scene
                
    mabTriangles: array_like
            (n,3) np.uint32 array containing triangle indices that create the polygons from mapVertices
    """
    # Create two lists of 3d coordinates in 2 z-planes
    lowerZ = np.full((contourPoints.shape[0], 1), -0.5)
    upperZ = np.full((contourPoints.shape[0], 1), 0.5)
    # Combine the lists with alternating rows to serve as the mesh-vertices
    # https://stackoverflow.com/a/64621147
    vertices = np.ravel(
        [
            np.hstack((contourPoints, lowerZ)),
            np.hstack((contourPoints, upperZ))
        ], order="F"
    ).reshape(3,-1).T
    # Manually create the triangles, which is convenient due to the vertex list ordering
    triangles = np.array([
        np.array([0, 1, 2]) + np.full((3), i)
        for i, _ in enumerate(vertices)
    ], dtype=np.int32)
    # Manually fix last 2 entries to make the mesh loop around
    triangles[-1][1] = 0
    triangles[-1][2] = 1
    triangles[-2][2] = 0
      
    # Return vertices and triangles
    return vertices, triangles

def reconstruct2DMap(inFile):
    """Creates vertices and triangles of a 3D mesh representing a 2D ROS map.
    
    Parameters
    ----------
    inFile : string
            Path to map.yaml file the simulated world is based on
                
    Returns
    -------
    mapVertices: array_like
            (n,3) np.float32 array containing vertex points in world coordinates for the raycasting scene
                
    mabTriangles: array_like
            (n,3) np.uint32 array containing triangle indices that create the polygons from mapVertices
    """
    # Load map image (we use ROS' native yaml map format)
    with open(inFile) as f:
        yamlContent = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # Set correct path for map image and make sure it has been properly loaded
    imgPath = os.path.join(
        os.path.dirname(inFile),
        yamlContent["image"]
    )
    
    mapImg = cv2.imread(imgPath)
    assert mapImg is not None, "map.pgm could not be openened. Please check your robot configuration and/or map.yaml"
    
    # Rotate image to match ROS conventions
    mapImg = cv2.rotate(mapImg, cv2.ROTATE_90_CLOCKWISE)
    
    # Thresshold map for contour detection
    mapImg = cv2.cvtColor(mapImg, cv2.COLOR_BGR2GRAY)
    _, mapImg = cv2.threshold(mapImg, 127, 255, cv2.THRESH_BINARY)

    # TODO: thresshold based on map metadata
    #mapImg = (mapImg/255).astype(np.float32)
    #_, mapImg = cv2.threshold(mapImg, yamlContent["occupied_thresh"], 1.0, cv2.THRESH_BINARY)
    #_, mapImg = cv2.threshold(mapImg, yamlContent["free_thresh"], 1.0, cv2.THRESH_BINARY)
    #mapImg = (mapImg*255).astype(np.uint32)

    # Get simplified contours (approx_simple -> reduces the number of resulting vertices)
    contours, hierarchy = cv2.findContours(mapImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## Create mesh
    # Transform points to world frame
    transformedContours = contours2points(contours, yamlContent, mapImg)

    # Placeholder for resulting mesh components (we don't colour the mesh)
    mapVertices = np.array([], dtype=np.float32)
    mapTriangles = np.array([], dtype=np.int32)
    idx = 0

    # Iterate over each contour and create the correct vertex, triangle combinations
    for entry in transformedContours[1:]:
        vertices, triangles = process2DPoints(entry)
        mapVertices = np.vstack((mapVertices, vertices)) if mapVertices.size else vertices

        # Increment each value in triangles to the correct vertex
        triangles = triangles + np.full(triangles.shape, idx)
        idx += len(triangles)
        mapTriangles = np.vstack((mapTriangles, triangles)) if mapTriangles.size else triangles
    
    # Make sure vertices and triangles are of correct dtype when returning
    return mapVertices.astype(np.float32), mapTriangles.astype(np.int32)
