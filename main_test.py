import numpy as np

def main():
    points = np.array([[1,1],[3,2]])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))




main()