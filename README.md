# Compatible mesh creator
This repository provides an algorthim for prividing compatble meshes (the meshes that share the same nodes and same mesh connectivity). This algorithm was implemented for creating compatible meshes for interpolating different shapes using Poisson shape interpolation:
https://github.com/Mahdidaeidaei/Poisson-shape-interpolation

This algorithm does not work for every type of geometries. Only on those without disconnectivity and holes inside. 
In each iteration a plane ortogonal to the x-y plane intersects with the mesh and then, the intersection line is homogeneously devided into 200 points. These planes are equally distanced from each other between the *x* and *y* axis.  

Due to symmetry, only a quarter of each geometry is taken into account:


<img width="740" height="712" alt="image" src="https://github.com/user-attachments/assets/83a07ada-6193-4808-81e4-5306a7f36d49" />

A mechanical part meshed with compatible mesh


<img width="651" height="864" alt="image" src="https://github.com/user-attachments/assets/1a77e15b-d06b-45b6-87c9-557bc0d3e892" />

A cube meshed with the same nodes and mesh connectivity


# Dependencies
Main dependencies that are not included in the repo and should be installed first (with PIP or conda):

Trimesh

# Instructions

The input STL files must be placed in the **database** folder, and the results will be saved in the **npyfiles** folder.  

In your command prompt (Anaconda command prompt in case using Anaconda) activate the environment including the Pytorch, then:

```
python main.py --e 1
```
The argument passed is the maximum threshold for the edge length. The algorithm features an internal tool to refine the STL mesh, thereby reducing anomalies in the result. This threshold should never be greater than the minimum thickness in the geometry.
