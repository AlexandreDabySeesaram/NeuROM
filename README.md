# Implementation of a HiDeNN for a 1D Bar
The objective of this code is to provide an implementation of a HiDeNN. The input of the layer is the coordinate $\underline{x}$ where the output is evaluated. In this case the output of the network is the displacement $\underline{u}\left(\underline{x}\right)$

The first hidden layers play the role of linear shape functions by applying constraints on the weights and biais of sub-networks composing the first layers. The last hidden layer, called interpolation layer in the reminder of the document, utilises the output of the shape functions to interpolate the output. Training the weights of that last hidden layer is the same as solving a FEM problem on a fixed mesh. The weights of the interpolation layer directly correspond to the nodal values associated to each shape function. Therefore prescibing dirichlet boundary conditions is straight forward by freezing the weights associated to the prescribed values of fixed DoFs.

## Architecture of the NN

### Constituing blocks

   * Linear left and right classes that are the two elementary blocks to create a linear 1D shape function
   * Shape class that are based on the two afforemetionned building blocks and, given a index, build the shape function associated to that index i
   * MeshNN class that, given geometric parameters, "assemble" the shape functions accordingly
    
        * `model = MeshNN(np,L)`
        * np & L being the number of DoFs and the length of the bar respectively
   * The Dirichlet boudary conditions (BCs) are set by calling
   
        * `model.setBCs(u_0,u_L)` 
        * with $u_0$ and $u_L$ being the prescribed displacement at $x=0$ and $x=L$ respectivly


## Definition of the Geometry and BCs

The user interface is easy to use. The geometric parameters need to be specified and the method MeshBeam.SetBCs is then called to set the BCs up.

## Training the NN 

The Volumic forces are accounted for in the loss function through the right hand side (RHS) function and the loss function is the potential energy.

The trainable parameters can be changed on the fly. 

* `model.Freeze_Mesh()` Freezes the mesh so that only the nodale values are trained
* `model.UnFreeze_Mesh()` Unfreezes the mesh so that the coordinates values can be trained

* `model.Freeze_FEM()` Freezes the nodale values so that only the coordinates are trained
* `model.UnFreeze_FEM()` Unfreezes the nodale so that FEM problem can be solved

## TODO
 Here is a list of short term ojectives

 * Compare the frozen-mesh solutions to similarly discretised FE solutions
 * Test different training strategies (when to freeze or un freeze part of the NN)
     * Use the hybrid training (with real FEM in between)
* Test to "Train"the hyper parameter $\alpha$ in the regularisation
* Implement reference elements