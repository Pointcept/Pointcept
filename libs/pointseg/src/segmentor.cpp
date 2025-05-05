#include "segmentor.h"
#include <algorithm>
#include <cmath>

using std::vector;

// Universe class implementation
universe::universe(int elements) {
  elts = new uni_elt[elements];
  num = elements;
  for (int i = 0; i < elements; i++) {
    elts[i].rank = 0;
    elts[i].size = 1;
    elts[i].p = i;
  }
}

universe::~universe() { delete[] elts; }

int universe::find(int x) {
  int y = x;
  while (y != elts[y].p)
    y = elts[y].p;
  elts[x].p = y;
  return y;
}

void universe::join(int x, int y) {
  if (elts[x].rank > elts[y].rank) {
    elts[y].p = x;
    elts[x].size += elts[y].size;
  } else {
    elts[x].p = y;
    elts[y].size += elts[x].size;
    if (elts[x].rank == elts[y].rank)
      elts[y].rank++;
  }
  num--;
}

int universe::size(int x) const { return elts[x].size; }

int universe::num_sets() const { return num; }

// Edge comparison operator
bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

// Segment graph implementation
universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c) {
  std::sort(edges, edges + num_edges);      // sort edges by weight
  universe *u = new universe(num_vertices); // make a disjoint-set forest
  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++) {
    threshold[i] = c;
  }
  // for each edge, in non-decreasing weight order
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];
    // components connected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
        u->join(a, b);
        a = u->find(a);
        threshold[a] = pedge->w + (c / u->size(a));
      }
    }
  }
  delete[] threshold;
  return u;
}

// vec3f class implementation
vec3f::vec3f() {
  x = 0;
  y = 0;
  z = 0;
}

vec3f::vec3f(float _x, float _y, float _z) {
  x = _x;
  y = _y;
  z = _z;
}

vec3f vec3f::operator+(const vec3f &o) {
  return vec3f{x + o.x, y + o.y, z + o.z};
}

vec3f vec3f::operator-(const vec3f &o) {
  return vec3f{x - o.x, y - o.y, z - o.z};
}

vec3f cross(const vec3f &u, const vec3f &v) {
  vec3f c = {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
  float n = sqrtf(c.x * c.x + c.y * c.y + c.z * c.z);
  c.x /= n;
  c.y /= n;
  c.z /= n;
  return c;
}

vec3f lerp(const vec3f &a, const vec3f &b, const float v) {
  const float u = 1.0f - v;
  return vec3f(v * b.x + u * a.x, v * b.y + u * a.y, v * b.z + u * a.z);
}

// Segment mesh kernel implementation
vector<int> segment_mesh_kernel(const float *verts_ptr,
                               const size_t vertexCount,
                               const int64_t *faces_ptr,
                               const size_t faceCount,
                               const float kthr,
                               const int segMinVerts) {
  vector<float> verts(verts_ptr, verts_ptr + vertexCount * 3);
  vector<int64_t> faces(faces_ptr, faces_ptr + faceCount * 3);

  // create points, normals, edges, counts vectors
  vector<vec3f> points(vertexCount);
  vector<vec3f> normals(vertexCount);
  vector<int> counts(verts.size(), 0);
  const size_t edgeCount = faceCount * 3;
  edge *edges = new edge[edgeCount];

  // Compute face normals and smooth into vertex normals
  for (int i = 0; i < faceCount; i++) {
    const int fbase = 3 * i;
    const int64_t i1 = faces[fbase];
    const int64_t i2 = faces[fbase + 1];
    const int64_t i3 = faces[fbase + 2];
    int vbase = 3 * i1;
    vec3f p1(verts[vbase], verts[vbase + 1], verts[vbase + 2]);
    vbase = 3 * i2;
    vec3f p2(verts[vbase], verts[vbase + 1], verts[vbase + 2]);
    vbase = 3 * i3;
    vec3f p3(verts[vbase], verts[vbase + 1], verts[vbase + 2]);
    points[i1] = p1;
    points[i2] = p2;
    points[i3] = p3;
    const int ebase = 3 * i;
    edges[ebase].a = i1;
    edges[ebase].b = i2;
    edges[ebase + 1].a = i1;
    edges[ebase + 1].b = i3;
    edges[ebase + 2].a = i3;
    edges[ebase + 2].b = i2;

    // smoothly blend face normals into vertex normals
    vec3f normal = cross(p2 - p1, p3 - p1);
    normals[i1] = lerp(normals[i1], normal, 1.0f / (counts[i1] + 1.0f));
    normals[i2] = lerp(normals[i2], normal, 1.0f / (counts[i2] + 1.0f));
    normals[i3] = lerp(normals[i3], normal, 1.0f / (counts[i3] + 1.0f));
    counts[i1]++;
    counts[i2]++;
    counts[i3]++;
  }

  for (int i = 0; i < edgeCount; i++) {
    int a = edges[i].a;
    int b = edges[i].b;

    vec3f &n1 = normals[a];
    vec3f &n2 = normals[b];
    vec3f &p1 = points[a];
    vec3f &p2 = points[b];

    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    float dd = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= dd;
    dy /= dd;
    dz /= dd;
    float dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
    float dot2 = n2.x * dx + n2.y * dy + n2.z * dz;
    float ww = 1.0f - dot;
    if (dot2 > 0) {
      ww = ww * ww;
    }
    edges[i].w = ww;
  }

  // Segment!
  universe *u = segment_graph(vertexCount, edgeCount, edges, kthr);

  // Joining small segments
  for (int j = 0; j < edgeCount; j++) {
    int a = u->find(edges[j].a);
    int b = u->find(edges[j].b);
    if ((a != b) && ((u->size(a) < segMinVerts) || (u->size(b) < segMinVerts))) {
      u->join(a, b);
    }
  }

  // Return segment indices as vector
  vector<int> outComps(vertexCount);
  for (int q = 0; q < vertexCount; q++) {
    outComps[q] = u->find(q);
  }
  delete[] edges;
  delete u;
  return outComps;
}

// Segment mesh implementation
torch::Tensor segment_mesh(torch::Tensor vertices, torch::Tensor faces, float kthr, int segMinVerts) {
  float *vertices_ptr = vertices.data_ptr<float>();
  int64_t *faces_ptr = faces.data_ptr<int64_t>();
  const size_t vertexCount = vertices.size(0);
  const size_t faceCount = faces.size(0);

  vector<int> comps = segment_mesh_kernel(vertices_ptr,
                                         vertexCount,
                                         faces_ptr,
                                         faceCount,
                                         kthr,
                                         segMinVerts);

  torch::Tensor result_index = torch::empty({int64_t(vertexCount)}, torch::TensorOptions().dtype(torch::kInt64));
  int64_t *result_ptr = result_index.data_ptr<int64_t>();
  for (int i = 0; i < vertexCount; ++i) {
    result_ptr[i] = int64_t(comps[i]);
  }

  return result_index;
}

// Segment point kernel implementation
vector<int> segment_point_kernel(const float *points_ptr,
                                const float *normals_ptr,
                                const size_t pointCount,
                                const int64_t *edges_ptr,
                                const size_t edgeCount,
                                const float kthr,
                                const int segMinVerts) {
  // create points, normals, edges, counts vectors
  vector<vec3f> points(pointCount);
  vector<vec3f> normals(pointCount);
  edge *edges = new edge[edgeCount];

  for (auto i = 0; i < pointCount; ++i) {
    auto pt_base = i * 3;
    points[i] = vec3f(points_ptr[pt_base], points_ptr[pt_base + 1], points_ptr[pt_base + 2]);
    normals[i] = vec3f(normals_ptr[pt_base], normals_ptr[pt_base + 1], normals_ptr[pt_base + 2]);
  }

  for (auto i = 0; i < edgeCount; ++i) {
    auto ed_base = i * 2;
    edges[i].a = edges_ptr[ed_base];
    edges[i].b = edges_ptr[ed_base + 1];
  }

  for (int i = 0; i < edgeCount; i++) {
    int a = edges[i].a;
    int b = edges[i].b;

    vec3f &n1 = normals[a];
    vec3f &n2 = normals[b];
    vec3f &p1 = points[a];
    vec3f &p2 = points[b];

    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    float dd = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= dd;
    dy /= dd;
    dz /= dd;
    float dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
    float dot2 = n2.x * dx + n2.y * dy + n2.z * dz;
    float ww = 1.0f - dot;
    if (dot2 > 0) {
      ww = ww * ww;
    }
    edges[i].w = ww;
  }

  universe *u = segment_graph(pointCount, edgeCount, edges, kthr);

  for (int j = 0; j < edgeCount; j++) {
    int a = u->find(edges[j].a);
    int b = u->find(edges[j].b);
    if ((a != b) && ((u->size(a) < segMinVerts) || (u->size(b) < segMinVerts))) {
      u->join(a, b);
    }
  }

  vector<int> outComps(pointCount);
  for (int q = 0; q < pointCount; q++) {
    outComps[q] = u->find(q);
  }

  delete[] edges;
  delete u;
  return outComps;
}

// Segment point implementation
torch::Tensor segment_point(torch::Tensor vertices, torch::Tensor normals, torch::Tensor edges, float kthr, int segMinVerts) {
  float *vertices_ptr = vertices.data_ptr<float>();
  float *normals_ptr = normals.data_ptr<float>();
  int64_t *edges_ptr = edges.data_ptr<int64_t>();
  const size_t pointCount = vertices.size(0);
  const size_t edgeCount = edges.size(0);

  vector<int> comps = segment_point_kernel(vertices_ptr,
                                          normals_ptr,
                                          pointCount,
                                          edges_ptr,
                                          edgeCount,
                                          kthr,
                                          segMinVerts);

  torch::Tensor result_index = torch::empty({int64_t(pointCount)}, torch::TensorOptions().dtype(torch::kInt64));
  int64_t *result_ptr = result_index.data_ptr<int64_t>();
  for (int i = 0; i < pointCount; ++i) {
    result_ptr[i] = int64_t(comps[i]);
  }

  return result_index;
}