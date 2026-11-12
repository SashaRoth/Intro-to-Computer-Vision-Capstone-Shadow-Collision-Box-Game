# Module for all of our test functions, which returns quantitative metrics for the pipeline.py processing steps

def mean_edge_distance(contour1, contour2):
    """Compute the mean edge distance between two contours."""
    # Compute the distance from each point in contour1 to the closest point in contour2
    distances = []
    for point1 in contour1:
        min_distance = float('inf')
        for point2 in contour2:
            distance = ((point1[0][0] - point2[0][0]) ** 2 + (point1[0][1] - point2[0][1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    
    # Return the mean of the distances
    return sum(distances) / len(distances) if distances else 0.0