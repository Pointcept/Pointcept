#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <vector>
#include <torch/extension.h>

using std::vector;

// Disjoint-set forest element
typedef struct {
  int rank;
  int p;
  int size;
} uni_elt;

// Disjoint-set forest class
class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);
  void join(int x, int y);
  int size(int x) const;
  int num_sets() const;

private:
  uni_elt *elts;
  int num;
};

// Edge structure for graph segmentation
typedef struct {
  float w;
  int a, b;
} edge;

bool operator<(const edge &a, const edge &b);

// Simple 3D vector class
class vec3f {
public:
  float x, y, z;
  vec3f();
  vec3f(float _x, float _y, float _z);
  vec3f operator+(const vec3f &o);
  vec3f operator-(const vec3f &o);
};

// Function prototypes
universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c);
vec3f cross(const vec3f &u, const vec3f &v);
vec3f lerp(const vec3f &a, const vec3f &b, const float v);

vector<int> segment_mesh_kernel(const float *verts_ptr,
                               const size_t vertexCount,
                               const int64_t *faces_ptr,
                               const size_t faceCount,
                               const float kthr,
                               const int segMinVerts);

torch::Tensor segment_mesh(torch::Tensor vertices,
                          torch::Tensor faces,
                          float kthr,
                          int segMinVerts);

vector<int> segment_point_kernel(const float *points_ptr,
                                const float *normals_ptr,
                                const size_t pointCount,
                                const int64_t *edges_ptr,
                                const size_t edgeCount,
                                const float kthr,
                                const int segMinVerts);

torch::Tensor segment_point(torch::Tensor vertices,
                           torch::Tensor normals,
                           torch::Tensor edges,
                           float kthr,
                           int segMinVerts);

#endif // SEGMENTATION_H