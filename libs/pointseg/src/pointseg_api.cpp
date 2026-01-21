#include "segmentor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("segment_mesh_fn", segment_mesh, py::arg("vertices"), py::arg("faces"), py::arg("kthr"), py::arg("segMinVerts"));
  m.def("segment_point_fn", segment_point, py::arg("vertices"), py::arg("normals"), py::arg("edges"), py::arg("kthr"), py::arg("segMinVerts"));
}
