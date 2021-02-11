from dolfin import *
import numpy as np
import torch

# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

class ObstacleLeftBottom(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (0.0, 0.5)) and between(x[1], (0.0, 0.5)))

class ObstacleLeftTop(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (0.0, 0.5)) and between(x[1], (0.5, 1.0)))

class ObstacleRightBottom(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (0.5, 1.0)) and between(x[1], (0.0, 0.5)))

class ObstacleRightTop(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (0.5, 1.0)) and between(x[1], (0.5, 1.0)))


class Poisson():
    def __init__(self, args):
        self.numObs = int(np.sqrt(args.xDim))

        # Initialize sub-domain instances
        self.left = Left()
        self.top = Top()
        self.right = Right()
        self.bottom = Bottom()
        self.regionFirst = ObstacleLeftBottom()
        self.regionSecond = ObstacleRightBottom()
        self.regionThird = ObstacleLeftTop()
        self.regionFourth = ObstacleRightTop()

        # Define mesh
        self.partition = (self.numObs + 1) ** 2
        self.mesh = UnitSquareMesh(self.partition, self.partition)

        # Initialize mesh function for interior domains
        self.domains = MeshFunction('size_t', self.mesh, self.mesh.topology().dim())
        self.regionFirst.mark(self.domains, 1)
        self.regionSecond.mark(self.domains, 2)
        self.regionThird.mark(self.domains, 3)
        self.regionFourth.mark(self.domains, 4)

        # Initialize mesh function for boundary domains
        self.boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        self.boundaries.set_all(0)
        self.left.mark(self.boundaries, 1)
        self.top.mark(self.boundaries, 2)
        self.right.mark(self.boundaries, 3)
        self.bottom.mark(self.boundaries, 4)

        # Define input data
        self.permeabilityFirst = Constant(1.0)
        self.permeabilitySecond = Constant(1.0)
        self.permeabilityThird = Constant(1.0)
        self.permeabilityFourth = Constant(1.0)
        self.neumannConditionLeft = Expression("- 10*exp(- pow(x[1] - 0.5, 2))", degree=2)
        self.neumannConditionRight = Constant("1.0")
        self.rightHandSide = Constant(1.0)

        # Define observation points
        self.V = FunctionSpace(self.mesh, "CG", 2)
        self.dof_coordinates = self.V.tabulate_dof_coordinates()
        self.dof_coordinates.resize((self.V.dim(), self.mesh.geometry().dim()))
        self.dof_x = self.dof_coordinates[:, 0]
        self.dof_y = self.dof_coordinates[:, 1]
        self.numPartition = 2 * self.partition + 1
        self.dictionary = {}
        for i in range(len(self.dof_x)):
            self.dictionary[str(int((self.numPartition - 1) * self.dof_x[i])),
                            str(int((self.numPartition - 1) * self.dof_y[i]))] = i
        self.observationIndices = []
        for j in range(self.numObs):
            for k in range(self.numObs):
                self.observationIndices.append(
                    self.dictionary[str(2 * (self.numObs + 1) * (j + 1)), str(2 * (self.numObs + 1) * (k + 1))])

    def poissonSingleSimulation(self, theta):
        # Assign parameters
        self.permeabilityFirst = Constant(theta[0])
        self.permeabilitySecond = Constant(theta[1])
        self.permeabilityThird = Constant(theta[2])
        self.permeabilityFourth = Constant(theta[3])

        # Define function space and basis functions
        self.V = FunctionSpace(self.mesh, "CG", 2)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Define Dirichlet boundary conditions at top and bottom boundaries
        self.bc = [DirichletBC(self.V, 5.0, self.boundaries, 2),
                   DirichletBC(self.V, 0.0, self.boundaries, 4)]

        # Define new measures associated with the interior domains and
        # exterior boundaries
        #self.dx = Measure("dx")[self.domains]
        #self.ds = Measure("ds")[self.boundaries]
        self.dx = Measure("dx", subdomain_data=self.domains)
        self.ds = Measure("ds", subdomain_data=self.boundaries)

        # Define variational form
        self.F = (inner(self.permeabilityFirst * grad(self.u), grad(self.v)) * self.dx(1)
                  + inner(self.permeabilitySecond * grad(self.u), grad(self.v)) * self.dx(2)
                  + inner(self.permeabilityThird * grad(self.u), grad(self.v)) * self.dx(3)
                  + inner(self.permeabilityFourth * grad(self.u), grad(self.v)) * self.dx(4)
                  - self.neumannConditionLeft * self.v * self.ds(1)
                  - self.neumannConditionRight * self.v * self.ds(3)
                  - self.rightHandSide * self.v * self.dx(1)
                  - self.rightHandSide * self.v * self.dx(2)
                  - self.rightHandSide * self.v * self.dx(3)
                  - self.rightHandSide * self.v * self.dx(4))

        # Separate left and right hand sides of equation
        self.a, self.L = lhs(self.F), rhs(self.F)

        # Solve problem
        self.u = Function(self.V)
        solve(self.a == self.L, self.u, self.bc)
        #self.plot(theta)
        return np.array(self.u.vector().get_local()).reshape(-1)[self.observationIndices]

    def run(self, thetas, observation):
        result = torch.Tensor([])
        for k in range(thetas.shape[0]):
            result = torch.cat((result, torch.Tensor(self.poissonSingleSimulation(thetas[k].cpu().detach().numpy())).reshape(-1)))

        return result.reshape(thetas.shape[0], -1)