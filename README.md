# Implementation of a HiDeNN for a 1D Bar
The objective of this code is to provide an implementation of a HiDeNN. The input of the layer is the coordinate $\vec{x}$ where the output is evaluated. In this case the output of the network is the displacement $\vec{u}\left(\vec{x}\right)$

The first hidden layers play the role of linear shape functions by applying constraints on the weights and biais of sub-networks composing the first layers. The last hidden layer, called interpolation layer in the reminder of the document, utilises the output of the shape functions to interpolate the output. Training the weights of that last hidden layer is the same as solving a FEM problem on a fixed mesh. The weights of the interpolation layer directly correspond to the nodal values associated to each shape function. Therefore prescibing dirichlet boundary conditions is straight forward by freezing the weights associated to the prescribed values of fixed DoFs.

## Architecture of the NN

### Constituing blocks

   * Linear left and right classes that are the two elementary blocks to create a linear 1D shape function
   * Shape class that are based on the two afforemetionned building blocks and, given a index, build the shape function associated to that index i
   * MeshNN class that, given geometric parameters, "assemble" the shape functions accordingly
    
        * model = MeshNN(np,L)
        * np & L being the number of DoFs and the length of the bar respectively
   * The Dirichlet boudary conditions (BCs) are set by calling
   
        * model.setBCs($x_0,x_L$) 
        * with $x_0$ and $x_L$ being the prescribed displacement at $x=0$ and $x=L$ respectivly


## Definition of the Geometry and BCs

The user interface is easy to use. The geometric parameters need to be specified and the method MeshBeam.SetBCs is then called to set the BCs up.

## Training the NN 

The Volumic forces are accounted for in the loss function through the right hand side (RHS) function and the loss function is the potential energy.