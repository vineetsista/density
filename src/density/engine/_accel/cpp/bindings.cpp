// pybind11 glue for the DENSITY scan kernels.
//
// This layer owns everything Python-facing so kernels.cpp can stay a file
// of plain loops over raw pointers: it coerces inputs to C-contiguous
// arrays of the right dtype (c_style | forcecast, so callers may pass any
// layout and pay at most one copy), validates shapes with real error
// messages, allocates the numpy output while still holding the GIL, and
// then releases the GIL for the duration of the scan so a multi-threaded
// caller can overlap searches with other Python work.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// Implemented in kernels.cpp.
void density_sq8_scores(const std::uint8_t* codes, const float* qa,
                        float constant, std::size_t n, std::size_t d,
                        float* out);
void density_pq_adc_scan(const std::uint8_t* codes, const float* lut,
                         std::size_t n, std::size_t m, float* out);
void density_hamming_scan(const std::uint8_t* codes, const std::uint8_t* q,
                          std::size_t n, std::size_t b, std::int32_t* out);

namespace {

using U8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using F32Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> sq8_scores(U8Array codes, F32Array qa, float constant) {
    if (codes.ndim() != 2) {
        throw std::invalid_argument("sq8_scores: codes must be 2-D [n, d]");
    }
    if (qa.ndim() != 1) {
        throw std::invalid_argument("sq8_scores: qa must be 1-D [d]");
    }
    const std::size_t n = static_cast<std::size_t>(codes.shape(0));
    const std::size_t d = static_cast<std::size_t>(codes.shape(1));
    if (static_cast<std::size_t>(qa.shape(0)) != d) {
        throw std::invalid_argument(
            "sq8_scores: qa length must equal codes.shape[1]");
    }
    py::array_t<float> out(static_cast<py::ssize_t>(n));
    const std::uint8_t* codes_ptr = codes.data();
    const float* qa_ptr = qa.data();
    float* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        density_sq8_scores(codes_ptr, qa_ptr, constant, n, d, out_ptr);
    }
    return out;
}

py::array_t<float> pq_adc_scan(U8Array codes, F32Array lut) {
    if (codes.ndim() != 2) {
        throw std::invalid_argument("pq_adc_scan: codes must be 2-D [n, m]");
    }
    if (lut.ndim() != 2) {
        throw std::invalid_argument("pq_adc_scan: lut must be 2-D [m, 256]");
    }
    const std::size_t n = static_cast<std::size_t>(codes.shape(0));
    const std::size_t m = static_cast<std::size_t>(codes.shape(1));
    if (static_cast<std::size_t>(lut.shape(0)) != m) {
        throw std::invalid_argument(
            "pq_adc_scan: lut.shape[0] must equal codes.shape[1]");
    }
    if (lut.shape(1) != 256) {
        throw std::invalid_argument("pq_adc_scan: lut.shape[1] must be 256");
    }
    py::array_t<float> out(static_cast<py::ssize_t>(n));
    const std::uint8_t* codes_ptr = codes.data();
    const float* lut_ptr = lut.data();
    float* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        density_pq_adc_scan(codes_ptr, lut_ptr, n, m, out_ptr);
    }
    return out;
}

py::array_t<std::int32_t> hamming_scan(U8Array codes, U8Array q) {
    if (codes.ndim() != 2) {
        throw std::invalid_argument("hamming_scan: codes must be 2-D [n, b]");
    }
    if (q.ndim() != 1) {
        throw std::invalid_argument("hamming_scan: q must be 1-D [b]");
    }
    const std::size_t n = static_cast<std::size_t>(codes.shape(0));
    const std::size_t b = static_cast<std::size_t>(codes.shape(1));
    if (static_cast<std::size_t>(q.shape(0)) != b) {
        throw std::invalid_argument(
            "hamming_scan: q length must equal codes.shape[1]");
    }
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(n));
    const std::uint8_t* codes_ptr = codes.data();
    const std::uint8_t* q_ptr = q.data();
    std::int32_t* out_ptr = out.mutable_data();
    {
        py::gil_scoped_release release;
        density_hamming_scan(codes_ptr, q_ptr, n, b, out_ptr);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_density_kernels, m) {
    m.doc() =
        "Compiled DENSITY scan kernels. Numerically matches "
        "density.engine._accel.fallback within 1e-5 relative.";
    m.def("sq8_scores", &sq8_scores, py::arg("codes"), py::arg("qa"),
          py::arg("const"),
          "Affine-decoded dot products against SQ8 codes: float32 [n].");
    m.def("pq_adc_scan", &pq_adc_scan, py::arg("codes"), py::arg("lut"),
          "ADC table scan over PQ codes: float32 [n].");
    m.def("hamming_scan", &hamming_scan, py::arg("codes"), py::arg("q"),
          "Popcount hamming distances over packed codes: int32 [n].");
}
