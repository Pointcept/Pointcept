from pointcept.models.utils.structure import Point

class BirdsEye:
    def __init__(self):
        # self.point = point
        pass

    def transform(self, point: Point):
        """
        Transforms the point cloud to a birds-eye view perspective.
        This involves projecting the 3D coordinates onto the XY plane.
        """
        assert "coord" in point.keys(), "Point must have 'coord' attribute"
        
        # Extract the XY coordinates and ignore the Z coordinate
        birds_eye_coord = point.coord[:, :2]
        
        # Create a new Point object with the transformed coordinates
        birds_eye_point = Point(point)
        birds_eye_point["coord"] = birds_eye_coord
        
        return birds_eye_point