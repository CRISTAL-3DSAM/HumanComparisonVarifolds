import numpy as np


class Surface:
    # Minimal surface class used to load CVSSP3D tri files as surfaces.
    # Contains as object:
    # vertices : all the vertices
    # centers: the centers of each face
    # faces: faces along with the id of the faces
    # surfel: surface element of each face (area*normal)
    # List of methods:
    # read : from filename type, call readFILENAME and set all surface attributes
    # updateVertices: update the whole surface after a modification of the vertices

    def __init__(self, filename):
        """
        Open cvssp file and initialize Surface object.
        :param filename: filename of the surface
        """
        self.read(filename)
        self.computeCentersAreas()

    def read(self, filename):
        """
        Open cvssp file
        :param filename: filename of the surface
        """
        vertices = []
        faces = []
        with open(filename, "r") as f:
            pithc = f.readlines()
            line_1 = pithc[0]
            n_pts, n_tri = line_1.split(' ')
            for i, line in enumerate(pithc[1:]):
                pouetage = line.split(' ')
                if i <= int(n_pts):
                    vertices.append([float(poupout) for poupout in pouetage[:3]])
                    # vertices.append([float(poupout) for poupout in pouetage[4:7]])
                else:
                    faces.append([int(poupout.split("\n")[0]) for poupout in pouetage[1:]])
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        try:
            xDef1 = self.vertices[self.faces[:, 0], :]
        except:
            print(filename)

    def computeCentersAreas(self):
        # face centers and area weighted normal
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel = np.cross(xDef2-xDef1, xDef3-xDef1)

    def updateVertices(self, x0):
        # modify vertices without topological change
        self.vertices = np.copy(x0)
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)
        self.volume, self.vols = self.surfVolume()