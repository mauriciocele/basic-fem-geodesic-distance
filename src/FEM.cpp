/**
 * BASIC 2D FEM on a 3D MESH
 *
 * This follows the method described in: 
 * a) Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 * b) Hans Peter Langtangen, "Introduction to finite element methods", 2014 
 *    http://hplgit.github.io/INF5620/doc/pub/H14/fem/html/main_fem.html
 * 
 * 
 * Compute approximated geodesic distance on the mesh by the Heat Method using Finite Element Method (FEM)
 * 
 * First solves the Heat Equation PDE with dirichlet boundary conditions:
 * 
 * Laplacian U(x,y, t) = dU(x,y, t)/dt   in the region D
 *           U(x,y, t) = G(x,y)             on the region boundary #D for all times
 *           U(x,y, 0) = U_0(x,y)           initial condition at time t = 0
 * 
 * Then computes the negative of the gradient of U(x,y) on every triangle element and normalizes it:
 * 
 * X(x,y) = - grad U(x,y) / |grad U(x,y)|
 * 
 * Then solves a Poisson Equation PDE with dirichlet boundary conditions:
 * 
 * Laplacian Phi(x,y, t) = div X(x,y)         in the region D
 *           Phi(x,y, t) = G(x,y)             on the region boundary #D for all times
 * 
 * Finally shifts the distance Phi so the lowest value is zero
 */
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif

#if defined (__APPLE__) || defined (OSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include "GA/c3ga.h"
#include "GA/c3ga_util.h"
#include "GA/gl_util.h"

#include "primitivedraw.h"
#include "gahelper.h"
#include "Laplacian.h"

#include <memory>

#include <vector>
#include <queue>
#include <map>
#include <fstream>
#include <functional>
#include "numerics.h"
#include "HalfEdge/Mesh.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

// #include <ppl.h>

const char *WINDOW_TITLE = "FEM BASIC 2D";

// GLUT state information
int g_viewportWidth = 800;
int g_viewportHeight = 600;

void display();
void reshape(GLint width, GLint height);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void KeyboardUpFunc(unsigned char key, int x, int y);
void SpecialFunc(int key, int x, int y);
void SpecialUpFunc(int key, int x, int y);
void Idle();
void DestroyWindow();
Eigen::Vector3d valueToColor( double d );

//using namespace boost;
using namespace c3ga;
using namespace std;
using namespace numerics;

class Camera
{
public:
	float		pos[3];
	float		fw[3];
	float		up[3];
	float		translateVel;
	float		rotateVel;

	Camera()
	{
		float		_pos[] = { 0, 0, 2};
		float		_fw[] = { 0, 0, -1 };
		float		_up[] = { 0, 1, 0 };

		translateVel = 0.005;
		rotateVel = 0.005;
		memcpy(pos, _pos, sizeof(float)*3);
		memcpy(fw, _fw, sizeof(float)*3);
		memcpy(up, _up, sizeof(float)*3);
	}

	void glLookAt()
	{
		gluLookAt( pos[0], pos[1], pos[2], fw[0],  fw[1],  fw[2], up[0],  up[1],  up[2] );
	}
};

class VertexBuffer
{
public:
	std::vector<Eigen::Vector3d> positions; //mesh vertex positions
	std::vector<Eigen::Vector3d> normals; //for rendering (lighting)
	std::vector<Eigen::Vector3d> colors; //for rendering (visual representation of values)
	int size;

	VertexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		positions.resize(size);
		normals.resize(size);
		colors.resize(size);
	}
	int get_size() { return size; }

};

class IndexBuffer {
public:
	std::vector<int> faces;
	int size;

	IndexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		faces.resize(size);
	}
	int get_size() { return size; }

};

Camera g_camera;
Mesh mesh;
vectorE3GA g_prevMousePos;
bool g_rotateModel = false;
bool g_rotateModelOutOfPlane = false;
rotor g_modelRotor = _rotor(1.0);
float g_dragDistance = -1.0f;
int g_dragObject;
bool g_showWires = true;
bool g_showVectorField = false;
bool g_showIsolines = true;


VertexBuffer vertexBuffer;
IndexBuffer indexBuffer;
std::shared_ptr<SparseMatrix> A;
Eigen::VectorXd right_hand_side;
Eigen::VectorXd solutionU;
Eigen::VectorXd divergenceD;
Eigen::VectorXd distances;
Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
std::set<int> allconstraints;
std::vector<Eigen::Vector3d> faceGradients;
std::vector<Eigen::Vector3d> faceGradientsDDG;
std::vector<Eigen::Vector3d> faceCentroids;
std::vector<Eigen::Vector3d> isolinesPositions;
/**
 * Add up all quantities associated with the element.
 * 
 * It computes the gradients on a reference 2D triagle then "push forward" it
 * to the triangle using the "differential" of the mapping. All of that follows
 * from the chain-rule as described in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleStiffnessElement(Vertex* v[3]) {

	Eigen::Matrix2d B, Binv;
	Eigen::Vector2d gradN[3];
	Eigen::Matrix3d elementMatrix;

	B(0,0) = v[1]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y();
	B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,1) = v[2]->p.y() - v[0]->p.y();

	Binv = B.inverse().transpose();

	double faceArea = 0.5 * abs(B.determinant());

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector2d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1));
	gradN[1] = Eigen::Vector2d(Binv(0,0), Binv(1,0));
	gradN[2] = Eigen::Vector2d(Binv(0,1), Binv(1,1));
	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
			elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]);
			if (i != j) {
				elementMatrix(j, i) = elementMatrix(i, j);
			}
		}
	}
	return elementMatrix;
}

/**
 * Extend computation of per-element stiffness matrix of triangle elements embedded in 3D space.
 * Original method only works for triangle elements on 2D space, see:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleStiffnessElementEmbedded(Vertex* v[3]) {

	Eigen::MatrixXd B(3, 2);
	Eigen::MatrixXd Binv(3, 2);
	Eigen::Vector3d gradN[3];
	Eigen::Matrix3d elementMatrix;

	B(0,0) = v[1]->p.x() - v[0]->p.x(); B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y(); B(1,1) = v[2]->p.y() - v[0]->p.y();
	B(2,0) = v[1]->p.z() - v[0]->p.z(); B(2,1) = v[2]->p.z() - v[0]->p.z();
    
	Binv = ((B.transpose() * B).inverse() * B.transpose()).transpose();

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector3d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1), -Binv(2,0) - Binv(2,1));
	gradN[1] = Eigen::Vector3d( Binv(0,0), Binv(1,0), Binv(2,0));
	gradN[2] = Eigen::Vector3d( Binv(0,1), Binv(1,1), Binv(2,1));
	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
			elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]);
			if (i != j) {
				elementMatrix(j, i) = elementMatrix(i, j);
			}
		}
	}
	return elementMatrix;
}

/**
 * Extend computation of per-element mass matrix of triangle elements embedded in 3D space.
 * Original method only works for triangle elements on 2D space, see:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3d AssembleMassElement(Vertex* v[3], bool useMidpoints = true) {
	std::function<double(const Eigen::Vector2d &)> N[3];
	Eigen::Vector2d M_0, M_1, M_2;
	Eigen::Matrix3d massMatrix;

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();
 
	N[0] = [](const Eigen::Vector2d &U) { return 1 - U.x() - U.y(); };
	N[1] = [](const Eigen::Vector2d &U) { return U.x(); };
	N[2] = [](const Eigen::Vector2d &U) { return U.y(); };

	if (useMidpoints) {
		// we use triangle mid-points instead of vertices for exact quadrature rule
		M_0 = 0.5 * (Eigen::Vector2d(0,0) + Eigen::Vector2d(1,0)); 
		M_1 = 0.5 * (Eigen::Vector2d(1,0) + Eigen::Vector2d(0,1));
		M_2 = 0.5 * (Eigen::Vector2d(0,1) + Eigen::Vector2d(0,0));
	} else {
		M_0 = Eigen::Vector2d(0,0); 
		M_1 = Eigen::Vector2d(1,0);
		M_2 = Eigen::Vector2d(0,1);
	}

	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K / 3 [ N_i(M1) N_j(M1) + N_i(M2) N_j(M2) + N_i(M3) N_j(M3)]
			massMatrix(i, j) = (faceArea / 3.0) * (N[i](M_0) * N[j](M_0) + N[i](M_1) * N[j](M_1) + N[i](M_2) * N[j](M_2));
			if (i != j) {
				massMatrix(j, i) = massMatrix(i, j);
			}
		}
	}
	return massMatrix;
}

/**
 * Assemble the stiffness matrix. It does not take into account boundary conditions.
 * Boundary conditions will be applied when linear system is pre-factored (LU decomposition)
 * Original method can be found in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
std::shared_ptr<SparseMatrix> AssembleMatrix(Mesh *mesh, double delta_t) {
	std::shared_ptr<SparseMatrix> A(new SparseMatrix(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3d stiffnessMatrix, massMatrix;
	double wij;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		stiffnessMatrix = AssembleStiffnessElementEmbedded(v);
		//massMatrix = AssembleMassElement(v);
		for( int i = 0 ; i < 3 ; ++i ) {
			for (int j = 0 ; j < 3 ; ++j ) {
				if (i < j) continue; // since stifness matrix is symmetric
				//wij = massMatrix(i, j) + delta_t * stiffnessMatrix(i, j);
				wij = delta_t * stiffnessMatrix(i, j);
				(*A)(v[i]->ID, v[j]->ID) += wij;
				if (i != j) {
					(*A)(v[j]->ID, v[i]->ID) += wij;
				}
			}
		}
	}
	return A;
}

std::shared_ptr<SparseMatrix> AssembleDiagonalMassMatrix(Mesh *mesh) {
	std::shared_ptr<SparseMatrix> A(new SparseMatrix(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3d stiffnessMatrix, massMatrix;
	double wij;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		massMatrix = AssembleMassElement(v, false); // mass matrix is diagonal when useMidpoints = false
		wij = massMatrix(0, 0);
		(*A)(v[0]->ID, v[0]->ID) += wij;
		wij = massMatrix(1, 1);
		(*A)(v[1]->ID, v[1]->ID) += wij;
		wij = massMatrix(2, 2);
		(*A)(v[2]->ID, v[2]->ID) += wij;
	}
	return A;
}

void IplusMinvTimesA(std::shared_ptr<SparseMatrix> M, std::shared_ptr<SparseMatrix> A)
{
	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		SparseMatrix::RowIterator aIter = A->iterator(i);
		double oneOverVertexOneRingArea = 1.0 / (*M)(i, i);
		for (; !aIter.end(); ++aIter)
		{
			auto j = aIter.columnIndex();
			(*A)(i, j) *= oneOverVertexOneRingArea;
			if (i == j) {
				(*A)(i, j) += 1.0; // this completes the (I + M^-1 L)
			}
		}
	}
}

/**
 * 
 * GRADIENT
 * 
 * Discretised function U
 * 
 * U(x,y) = sum_i U_i N_i(x,y)
 * 
 * Grandient in the reference triangle is:
 * 
 * grad U(x,y) = grad sum_i U_i N_i(x,y)
 * grad U(x,y) = sum_i U_i grad N_i(x,y)
 * 
 * grad_U N_1(U) = (-1, -1)
 * grad_U N_2(U) = (1, 0)
 * grad_U N_3(U) = (0, 1)
 * 
 * grad U(x,y) = U_1 (-1, -1) + U_2 (1, 0) + U_3 (0, 1)
 * grad U(x,y) = (U_2 - U_1, U_3 - U_1)
 * 
 * Gradient in the "physical" triangle is:
 * 
 * grad_X U(X) = sum_i U_i grad N^k_i(x,y)
 * 
 * grad_X N^k_i(X) = grad_X N_i( F^-1(X) )
 *                 = grad_U N_i( F^-1(X) ) * grad_X F^-1(X)    By chain rule
 *                 = B^-*T grad_U N_i( F^-1(X) )
 * 
 * grad_X F^-1(X) = grad_X B^-* (X - X_1) 
 *                = grad_X B^-* X - grad B^-* X_1     (but grad B^-* X_1 = 0 as there is no X)
 *                = grad_X B^-* X
 *                = B^-*T      (transposed, where B^-*T is a 3x2 matrix)
 * 
 * grad_X N^k_1(X) = B^-*T (-1, -1)
 * grad_X N^k_2(X) = B^-*T (1, 0)
 * grad_X N^k_3(X) = B^-*T (0, 1)
 * 
 * grad_X U(X) = U_1 B^-*T (-1, -1) + U_2 B^-*T (1, 0) + U_3 B^-*T (0, 1)
 * grad_X U(X) = B^-*T (U_2 - U_1, U_3 - U_1)
 */
void ComputeTriangleGradients(Mesh *mesh, const Eigen::VectorXd& solutionU, std::vector<Eigen::Vector3d>& gradients) {
	Vertex* v[3];
	Eigen::MatrixXd B(3, 2);
	Eigen::MatrixXd Binv(3, 2);
	Eigen::Vector2d gradU;
	Eigen::Vector3d gradX;
	double faceArea;

	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;

		B(0,0) = v[1]->p.x() - v[0]->p.x(); B(0,1) = v[2]->p.x() - v[0]->p.x();
		B(1,0) = v[1]->p.y() - v[0]->p.y(); B(1,1) = v[2]->p.y() - v[0]->p.y();
		B(2,0) = v[1]->p.z() - v[0]->p.z(); B(2,1) = v[2]->p.z() - v[0]->p.z();

		Binv = ((B.transpose() * B).inverse() * B.transpose()).transpose();
		gradU.x() = solutionU[v[1]->ID] - solutionU[v[0]->ID];
		gradU.y() = solutionU[v[2]->ID] - solutionU[v[0]->ID];
		gradX = Binv * gradU;
		gradients[face.ID] = gradX;
	}
}

void ComputeTriangleGradientsDDG(Mesh *mesh, const Eigen::VectorXd& solutionU, std::vector<Eigen::Vector3d>& gradients) {
	Vertex* v[3];
	Eigen::Vector3d gradX;
	Eigen::Vector3d N;
	double faceArea;

	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;

		N = (v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p);
		faceArea = N.norm();
		N *= 1.0 / faceArea;
		faceArea *= 0.5;
		// 1 - 0
		// 2 - 1
		// 0 - 2
		gradX.setZero();
		gradX += solutionU[v[2]->ID] * N.cross(v[1]->p - v[0]->p);
		gradX += solutionU[v[0]->ID] * N.cross(v[2]->p - v[1]->p);
		gradX += solutionU[v[1]->ID] * N.cross(v[0]->p - v[2]->p);

		gradients[face.ID] = (1.0 / (2.0*faceArea)) * gradX;
	}
}

/**
 * DIVERGENCE
 * 
 * The divergence operator acts on a vector-valued function A(X). Where X is a point on the surface.
 * Therefore A(X) is a vector field. For 2D case it should be A(x,y)
 * 
 * In FEM the vector-valued function A(X) can be defined on triangles, nodes or edges. 
 * According to some books it is best to define divergence as constant on nodes and 
 * vector-valued function A(X) be constant on triangles
 * 
 * 
 * WEAK FORMULATION OF DIVERGENCE
 * 
 * In order to discretize the divergence on each vertex we multiply it with a “test function” V(X) of compact support so:
 * 
 * div A_i(X) = Integral_Di V(X) div A(X) dA
 * 
 * Recall the vector calculus identity:
 * 
 * div (V(X) A(X)) = <grad V(X), A(X)> +  V(X) div A(X)            by “product rule”
 * 
 * So
 * 
 * V(X) div A(X)  = div (V(X) A(X)) - <grad V(X), A(X)>
 * 
 * So
 * Integral_D V(X) div A(X) dA  = Integral_D div (V(X) A(X)) dA - Integral_D <grad V(X), A(X)> dA
 *                              = itengral_#D V(X) <A(X), n> dS - Integral_D <grad V(X), A(X)> dA    by “divergence theorem”
 * 
 * Divergence Theorem:
 * 
 * Integral_D div (V(X) A(X)) dA = itengral_#D  <V(X) A(X), n> dS
 *                               = itengral_#D  V(X) <A(X), n> dS
 * 
 * If we define V(X) to be zero at the boundary #D then we have:
 * 
 * Integral_D V(X) div A(X) dA  = - Integral_D <grad V(X), A(X)> dA
 * 
 */
Eigen::Vector3d AssembleIntegratedDivergenceElement(Vertex* v[3], const Eigen::Vector3d& elementGradient) {
	Eigen::MatrixXd B(3, 2);
	Eigen::MatrixXd Binv(3, 2);
	Eigen::Vector3d gradN[3];
	Eigen::Vector3d divergenceAtElementNodes;
	Eigen::Vector3d faceGradient;

	B(0,0) = v[1]->p.x() - v[0]->p.x(); B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y(); B(1,1) = v[2]->p.y() - v[0]->p.y();
	B(2,0) = v[1]->p.z() - v[0]->p.z(); B(2,1) = v[2]->p.z() - v[0]->p.z();
    
	Binv = ((B.transpose() * B).inverse() * B.transpose()).transpose();

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector3d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1), -Binv(2,0) - Binv(2,1));
	gradN[1] = Eigen::Vector3d( Binv(0,0), Binv(1,0), Binv(2,0));
	gradN[2] = Eigen::Vector3d( Binv(0,1), Binv(1,1), Binv(2,1));

	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		divergenceAtElementNodes(i) = -faceArea * gradN[i].dot(elementGradient);
	}
	return divergenceAtElementNodes;

}

Eigen::Vector3d AssembleIntegratedDivergenceElementDDG(Vertex* v[3], const Eigen::Vector3d& elementGradient) {
	Eigen::Vector3d divergenceAtElementNodes;
	std::function<double(const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&)> cotangent = 
		[](const Eigen::Vector3d& vi, const Eigen::Vector3d& vk, const Eigen::Vector3d& vj, const Eigen::Vector3d& gradientX) {
			/**
			 * 
			 *   k
			 *   º
			 *   |   \   e3
			 * e1|        \
			 *   |              \
			 *   º---------------º
			 *   i      e2        j
			 */
			Eigen::Vector3d e1 = vi - vk; // ki
			Eigen::Vector3d e2 = vj - vi; // ij
			Eigen::Vector3d e3 = vk - vj; // jk
			
			double cotTheta2 = e1.dot(-e3) / e1.cross(-e3).norm();
			double cotTheta1 = -e2.dot(e3) / -e2.cross(e3).norm();
			return  0.5 * (cotTheta1 * e1.dot(gradientX) + cotTheta2 * e2.dot(gradientX));
		};

	for(int i = 0 ; i < 3 ; ++i) {
		int j = (i + 1) % 3;
		int k = (j + 1) % 3;
		divergenceAtElementNodes(i) = cotangent(v[i]->p, v[j]->p, v[k]->p, elementGradient);
	}

	return divergenceAtElementNodes;

}

/**
 * Assemble the stiffness matrix. It does not take into account boundary conditions.
 * Boundary conditions will be applied when linear system is pre-factored (LU decomposition)
 * Original method can be found in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
void AssembleDivergenceVector(Mesh *mesh, const std::vector<Eigen::Vector3d>& gradients, Eigen::VectorXd& divergenceD) {
	Eigen::Vector3d divergeceElement, divergeceElementDDG;
	Vertex* v[3];
	divergenceD.setZero();
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		divergeceElement = AssembleIntegratedDivergenceElement(v, gradients[face.ID]);
		divergeceElementDDG = AssembleIntegratedDivergenceElementDDG(v, gradients[face.ID]);
		for( int i = 0 ; i < 3 ; ++i ) {
			divergenceD(v[i]->ID) += divergeceElement(i);
		}
	}
}

bool is_constrained(std::set<int>& constraints, int vertex)
{
	return constraints.find(vertex) != constraints.end();
}

/**
 * Apply boundaty conditions to stiffness matrix and pre-factor it 
 * using sparse LU decomposition.
 */
void PreFactor(std::shared_ptr<SparseMatrix> A, std::set<int>& constraints, Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>& solver)
{

	Eigen::SparseMatrix<double> Lc = Eigen::SparseMatrix<double>(A->numRows(), A->numColumns());

	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		if (!is_constrained(constraints, i))
		{
			SparseMatrix::RowIterator aIter = A->iterator(i);
			for (; !aIter.end(); ++aIter)
			{
				auto j = aIter.columnIndex();
				Lc.insert(i, j) = (*A)(i, j);
			}
		}
		else
		{
			Lc.insert(i, i) = 1.0;
		}
	}

	Lc.makeCompressed();
	solver.compute(Lc);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Error: " << "Prefactor failed." << std::endl;
		exit(1);
	}
}

void Isolines(Mesh *mesh, double maxDistance, const Eigen::VectorXd& distance, std::vector<Eigen::Vector3d>& isolinesPositions) {
	Vertex* v[3];
	double distBetweenLines = maxDistance / 10.0;
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		std::vector<Eigen::Vector3d> segment;
		for(int i = 0 ; i < 3 ; ++i) {
			int j = (i + 1) % 3;
			
			double region1 = floor(distance(v[i]->ID) / distBetweenLines);
			double region2 = floor(distance(v[j]->ID) / distBetweenLines);

			if (region1 != region2) {
				double lambda = region1 < region2 ?
					(region2 * distBetweenLines - distance(v[i]->ID)) / (distance(v[j]->ID) - distance(v[i]->ID)) :
					(region1 * distBetweenLines - distance(v[i]->ID)) / (distance(v[j]->ID) - distance(v[i]->ID));
				Eigen::Vector3d p = v[i]->p + (v[j]->p - v[i]->p) * lambda;

				segment.push_back(p);
			}
		}
		if (segment.size() == 2) {
			for (int i = 0; i < 2; i++) {
				isolinesPositions.push_back(segment[i]);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	/**
	 * Load the FEM mesh
	 */
	mesh.readFEM("lake_nodes.txt", "lake_elements.txt");
	//mesh.readOBJ("cactus1.obj");
	mesh.CenterAndNormalize();
	mesh.computeNormals();

	// GLUT Window Initialization:
	glutInit (&argc, argv);
	glutInitWindowSize(g_viewportWidth, g_viewportHeight);
	glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow(WINDOW_TITLE);

	// Register callbacks:
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutKeyboardUpFunc(KeyboardUpFunc);
	glutSpecialFunc(SpecialFunc);
	glutSpecialUpFunc(SpecialUpFunc);
	glutIdleFunc(Idle);
	atexit(DestroyWindow);

	InitializeDrawing();

	vertexBuffer.resize(mesh.numVertices());
	indexBuffer.resize(mesh.numFaces() * 3);

	/**
	 * Initialize the vertex-buffer for OpenGL rendering purposes
	 */
	for( Vertex& vertex : mesh.getVertices())
	{
		vertexBuffer.positions[vertex.ID] = vertex.p;
		vertexBuffer.normals[vertex.ID] = vertex.n;
		vertexBuffer.colors[vertex.ID] = valueToColor(0);
	}
	
	double edgeCount = 0;
	double edgeLength = 0;
	/**
	 * Initialize the index-buffer for OpenGL rendering purposes
	 */
	for (Face& face : mesh.getFaces()) {
		int i = face.ID;
		int	v1 = face.edge->vertex->ID;
		int	v2 = face.edge->next->vertex->ID;
		int	v3 = face.edge->next->next->vertex->ID;
		indexBuffer.faces[i * 3 + 0] = v1;
		indexBuffer.faces[i * 3 + 1] = v2;
		indexBuffer.faces[i * 3 + 2] = v3;
		
		edgeLength += (face.edge->vertex->p - face.edge->next->vertex->p).norm();
		edgeLength += (face.edge->next->next->vertex->p - face.edge->vertex->p).norm();
		edgeLength += (face.edge->next->vertex->p - face.edge->next->next->vertex->p).norm();
		edgeCount += 3;
	}

	edgeLength /= edgeCount;

	/**
	 * Assemble the stiffness sparse matrix
	 */
	A = AssembleMatrix(&mesh, edgeLength*edgeLength);
	std::shared_ptr<SparseMatrix> M;
	M = AssembleDiagonalMassMatrix(&mesh);
	IplusMinvTimesA(M, A);

	/**
	 * Setup the right-hand-side of the linear system - including boundary conditions
	 */
	right_hand_side = Eigen::VectorXd(mesh.numVertices());
	right_hand_side.setZero(); // solve laplace's equation where RHS is zero
	
	for( Vertex& vertex : mesh.getVertices())
	{
		if (vertex.p.norm() < 2.5e-2) {
			right_hand_side(vertex.ID) = 1.0;
			allconstraints.insert(vertex.ID);
		}
		/*
		if (vertex.p.z() > 0.83) {
			right_hand_side(vertex.ID) = 1.0;
			allconstraints.insert(vertex.ID);
		}
		*/
	}

	/**
	 * Apply boundary conditions and perform sparse LU decomposition
	 */
	PreFactor(A, allconstraints, solver);

	/**
	 * Solve Heat equation
	 */
	solutionU = solver.solve(right_hand_side);

	/**
	 * Compute gradients
	 */
	faceGradients.resize(mesh.numFaces());
	ComputeTriangleGradients(&mesh, solutionU, faceGradients);

	faceGradientsDDG.resize(mesh.numFaces());
	ComputeTriangleGradientsDDG(&mesh, solutionU, faceGradientsDDG);

	// Normalize gradients
	for (Face& face : mesh.getFaces()) {
		// Compute the negative of the gradient for distance function
		faceGradients[face.ID] = -faceGradients[face.ID].normalized();
		faceGradientsDDG[face.ID] = -faceGradientsDDG[face.ID].normalized();
	}

	divergenceD = Eigen::VectorXd(mesh.numVertices());
	AssembleDivergenceVector(&mesh, faceGradients, divergenceD);

	/**
	 * Assemble the stiffness sparse matrix
	 */
	A = AssembleMatrix(&mesh, 1.0);

	/**
	 * Apply boundary conditions and perform sparse LU decomposition
	 */
	PreFactor(A, allconstraints, solver);

	/**
	 * Solve Poissoin's equation
	 */
	distances = solver.solve(divergenceD);

	/** 
	 * Shift such that it's min value is zero
	 */
	double min_dist = *std::min_element(distances.begin(), distances.end());

	for(int i = 0; i < distances.size() ; ++i) {
		distances(i) = distances(i) - min_dist;
	}

	/**
	 * Map solution values into colors
	 */
	double min_bc = *std::min_element(distances.begin(), distances.end());
	double max_bc = *std::max_element(distances.begin(), distances.end());

	for( Vertex& vertex : mesh.getVertices())
	{
		vertexBuffer.colors[vertex.ID] = valueToColor((distances[vertex.ID] - min_bc)/(max_bc - min_bc));
	}

	/**
	 * Compute face's centroids
	 */
	faceCentroids.resize(mesh.numFaces());
	for (Face& face : mesh.getFaces()) {
		auto v0 = face.edge->vertex;
		auto v1 = face.edge->next->vertex;
		auto v2 = face.edge->next->next->vertex;

		faceCentroids[face.ID] = (1.0/3.0) * (v0->p + v1->p + v2->p);
	}

	Isolines(&mesh, max_bc, distances, isolinesPositions);

	glutMainLoop();

	return 0;
}

void display()
{
	/*
	 *	matrices
	 */
	glViewport( 0, 0, g_viewportWidth, g_viewportHeight );
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	pickLoadMatrix();
	GLpick::g_frustumFar = 1000.0;
	GLpick::g_frustumNear = .1;
	gluPerspective( 60.0, (double)g_viewportWidth/(double)g_viewportHeight, GLpick::g_frustumNear, GLpick::g_frustumFar );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);	//gouraud shading
	glClearDepth(1.0f);
	glClearColor( .75f, .75f, .75f, .0f );
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

	/*
	 *	estados
	 */
	glEnable(GL_CULL_FACE);		//face culling
	glCullFace( GL_BACK );
	glFrontFace( GL_CCW );
	glEnable(GL_DEPTH_TEST);	//z-buffer
	glDepthFunc(GL_LEQUAL);

	/*
	 *	iluminacion
	 */
	float		ambient[] = { .3f, .3f, .3f, 1.f };
	float		diffuse[] = { .3f, .3f, .3f, 1.f };
	float		position[] = { .0f, 0.f, 15.f, 1.f };
	float		specular[] = { 1.f, 1.f, 1.f };

	glLightfv( GL_LIGHT0, GL_AMBIENT, ambient );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0);
	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0125);
	glEnable(  GL_LIGHT0   );
	glEnable(  GL_LIGHTING );
	//glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, specular );
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.f );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glLoadIdentity();

	g_camera.glLookAt();

	glLightfv( GL_LIGHT0, /*GL_SPOT_DIRECTION*/GL_POSITION, position );

	glPushMatrix();

	rotorGLMult(g_modelRotor);

	if (GLpick::g_pickActive) glLoadName((GLuint)-1);

	double alpha = 1.0;

	//glEnable (GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//alpha = 0.5;

	//Mesh-Faces Rendering
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL /*GL_LINE GL_FILL GL_POINT*/);
	glEnable (GL_POLYGON_OFFSET_FILL);
	glPolygonOffset (1., 1.);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable( GL_COLOR_MATERIAL );
	if (GLpick::g_pickActive) glLoadName((GLuint)10);

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
	glNormalPointer(GL_DOUBLE, 0, &vertexBuffer.normals[0]);
	glColorPointer(3, GL_DOUBLE, 0, &vertexBuffer.colors[0]);

	// draw the model
	glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
	// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	if (g_showWires)
	{
		if (!GLpick::g_pickActive)
		{
			//Mesh-Edges Rendering (superimposed to faces)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE /*GL_LINE GL_FILL GL_POINT*/);
			glColor4d(.5, .5, .5, alpha);
			glDisable(GL_LIGHTING);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
			// draw the model
			glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
			// deactivate vertex arrays after drawing
			glDisableClientState(GL_VERTEX_ARRAY);
			glEnable(GL_LIGHTING);
		}
	}

	glDisable( GL_COLOR_MATERIAL );
	glDisable(GL_POLYGON_OFFSET_FILL);

	if (g_showVectorField) {
		for(auto &face : mesh.getFaces()) {
			DrawArrow(
				c3gaPoint(faceCentroids[face.ID].x(), faceCentroids[face.ID].y(), faceCentroids[face.ID].z()), 
				_vectorE3GA(5e-1*faceGradients[face.ID].x(), 5e-1*faceGradients[face.ID].y(), 5e-1*faceGradients[face.ID].z())
			);
		}
	}

	if(g_showIsolines && isolinesPositions.size() >= 2) {
		for(int i = 0; i < isolinesPositions.size() ; i += 2) {
			int j = i + 1;
			DrawLine(
				isolinesPositions[i].x(), isolinesPositions[i].y(), isolinesPositions[i].z(),
				isolinesPositions[j].x(), isolinesPositions[j].y(), isolinesPositions[j].z()
			);
		}
	}

	//glDisable (GL_BLEND);

	glPopMatrix();

	glutSwapBuffers();
}

Eigen::Vector3d valueToColor( double d )
{
	static Eigen::Vector3d	c0 = Eigen::Vector3d( 1, 1, 1);
	static Eigen::Vector3d	c1 = Eigen::Vector3d( 1, 1, 0);
	static Eigen::Vector3d	c2 = Eigen::Vector3d( 0, 1, 0);
	static Eigen::Vector3d	c3 = Eigen::Vector3d( 0, 1, 1);
	static Eigen::Vector3d	c4 = Eigen::Vector3d( 0, 0, 1);

	if( d < 0.25 )
	{
		double alpha = (d - 0.0) / (0.25-0.0);
		return (1.0 - alpha) * c0 + alpha * c1;
	}
	else if( d < 0.5 )
	{
		double alpha = (d - 0.25) / (0.5-0.25);
		return (1.0 - alpha) * c1 + alpha * c2;
	}
	else if( d < 0.75 )
	{
		double alpha = (d - 0.5) / (0.75-0.5);
		return (1.0 - alpha) * c2 + alpha * c3;
	}
	else
	{
		double alpha = (d - 0.75) / (1.0-0.75);
		return (1.0 - alpha) * c3 + alpha * c4;
	}
}


void reshape(GLint width, GLint height)
{
	g_viewportWidth = width;
	g_viewportHeight = height;

	// redraw viewport
	glutPostRedisplay();
}

vectorE3GA mousePosToVector(int x, int y) {
	x -= g_viewportWidth / 2;
	y -= g_viewportHeight / 2;
	return _vectorE3GA((float)-x * e1 - (float)y * e2);
}

void MouseButton(int button, int state, int x, int y)
{
	g_rotateModel = false;

	if (button == GLUT_LEFT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);

		if(g_dragObject == -1 || g_dragObject == 10 )
		{
			vectorE3GA mousePos = mousePosToVector(x, y);
			g_rotateModel = true;

			if ((_Float(norm_e(mousePos)) / _Float(norm_e(g_viewportWidth * e1 + g_viewportHeight * e2))) < 0.2)
				g_rotateModelOutOfPlane = true;
			else g_rotateModelOutOfPlane = false;
		}
	}

	if (button == GLUT_RIGHT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);
	}
}

void MouseMotion(int x, int y)
{
	if (g_rotateModel )
	{
		// get mouse position, motion
		vectorE3GA mousePos = mousePosToVector(x, y);
		vectorE3GA motion = mousePos - g_prevMousePos;

		if (g_rotateModel)
		{
			// update rotor
			if (g_rotateModelOutOfPlane)
				g_modelRotor = exp(g_camera.rotateVel * (motion ^ e3) ) * g_modelRotor;
			else 
				g_modelRotor = exp(0.00001f * (motion ^ mousePos) ) * g_modelRotor;
		}

		// remember mouse pos for next motion:
		g_prevMousePos = mousePos;

		// redraw viewport
		glutPostRedisplay();
	}
}

void SpecialFunc(int key, int x, int y)
{
	switch(key) {
		case GLUT_KEY_F1 :
			{
				int mod = glutGetModifiers();
				if(mod == GLUT_ACTIVE_CTRL || mod == GLUT_ACTIVE_SHIFT )
				{
				}
			}
			break;
		case GLUT_KEY_UP:
			{
			}
			break;
		case GLUT_KEY_DOWN:
			{
			}
			break;
	}
}

void SpecialUpFunc(int key, int x, int y)
{
}

void KeyboardUpFunc(unsigned char key, int x, int y)
{
	if(key == 'w' || key == 'W')
	{
		g_showWires = !g_showWires;
		glutPostRedisplay();
	}
	if(key == 'v' || key == 'V')
	{
		g_showVectorField = !g_showVectorField;
		glutPostRedisplay();
	}
	if(key == 'i' || key == 'I')
	{
		g_showIsolines = !g_showIsolines;
		glutPostRedisplay();
	}
}

void Idle()
{
	// redraw viewport
}

void DestroyWindow()
{
	ReleaseDrawing();
}

