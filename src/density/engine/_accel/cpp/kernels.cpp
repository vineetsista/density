// DENSITY compiled scan kernels.
//
// How the memory is laid out, and why these loops are fast:
//
// Every codes matrix arriving here is C-contiguous (row-major) uint8, one
// row per stored vector, guaranteed by the pybind11 layer (c_style |
// forcecast). Row-major matters: each kernel walks rows front to back, so
// the codes stream through the cache hierarchy sequentially, one 64-byte
// cache line delivering 64 useful code bytes, and the hardware prefetcher
// sees a pure linear scan it can run ahead of. Nothing here allocates,
// nothing aliases (outputs are distinct buffers), and every accumulation
// happens in local scalars, which is exactly the shape a compiler needs to
// auto-vectorize at -O3 without being told anything else.
//
// Per kernel:
//
// - sq8_scores: the inner product is over a contiguous uint8 row and a
//   query that fits in L1 (3 KB at d=768). The inner loop is an 8-lane
//   widening multiply-accumulate the compiler turns into SIMD converts
//   plus FMAs; eight independent accumulators break the floating-point
//   dependency chain that would otherwise serialize the loop.
//
// - pq_adc_scan: the hot loop of the whole library. Scanning n codes costs
//   n * m data-dependent gathers into the lookup table, and that table walk
//   dominates: the table is m * 256 floats (192 KB at m=192), small enough
//   to stay resident in L2 for the entire scan, so each gather is an L1/L2
//   hit while the code bytes themselves stream linearly. Contrast the numpy
//   fallback, which walks column-by-column: with m=192 the column stride
//   spans three cache lines, so every single code byte it touches is a
//   fresh cache miss over a multi-hundred-MB matrix, m times over.
//
// - hamming_scan: packed binary codes are compared 64 bits at a time. XOR
//   of two 8-byte words plus one popcount instruction replaces 64 bit
//   comparisons; __builtin_popcountll compiles to a single POPCNT on any
//   x86-64 with -march=native and to a correct table routine elsewhere.
//   The query is pre-widened to words once, outside the row loop, and the
//   tail bytes of rows whose width is not a multiple of 8 are handled
//   byte-wise after the word loop.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

void density_sq8_scores(const std::uint8_t* codes, const float* qa,
                        float constant, std::size_t n, std::size_t d,
                        float* out) {
    constexpr std::size_t kLanes = 8;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* row = codes + i * d;
        float acc[kLanes] = {};
        std::size_t k = 0;
        for (; k + kLanes <= d; k += kLanes) {
            for (std::size_t t = 0; t < kLanes; ++t) {
                acc[t] += qa[k + t] * static_cast<float>(row[k + t]);
            }
        }
        // Pairwise combine keeps the reduction tree shallow, which both
        // vectorizes cleanly and keeps rounding error small.
        float total = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
                      ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; k < d; ++k) {
            total += qa[k] * static_cast<float>(row[k]);
        }
        out[i] = total + constant;
    }
}

void density_pq_adc_scan(const std::uint8_t* codes, const float* lut,
                         std::size_t n, std::size_t m, float* out) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* row = codes + i * m;
        // Four independent partial sums hide the gather latency: while one
        // lookup is in flight the other three chains keep the pipe busy.
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        std::size_t j = 0;
        for (; j + 4 <= m; j += 4) {
            s0 += lut[(j + 0) * 256 + row[j + 0]];
            s1 += lut[(j + 1) * 256 + row[j + 1]];
            s2 += lut[(j + 2) * 256 + row[j + 2]];
            s3 += lut[(j + 3) * 256 + row[j + 3]];
        }
        float total = (s0 + s1) + (s2 + s3);
        for (; j < m; ++j) {
            total += lut[j * 256 + row[j]];
        }
        out[i] = total;
    }
}

void density_hamming_scan(const std::uint8_t* codes, const std::uint8_t* q,
                          std::size_t n, std::size_t b, std::int32_t* out) {
    const std::size_t words = b / 8;
    // Widen the query to 64-bit words once; memcpy is the strict-aliasing
    // safe way to reinterpret bytes and compiles to plain loads.
    std::vector<std::uint64_t> qw(words);
    if (words != 0) {
        std::memcpy(qw.data(), q, words * 8);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* row = codes + i * b;
        std::int32_t dist = 0;
        for (std::size_t w = 0; w < words; ++w) {
            std::uint64_t cw;
            std::memcpy(&cw, row + w * 8, 8);
            dist += static_cast<std::int32_t>(__builtin_popcountll(cw ^ qw[w]));
        }
        for (std::size_t k = words * 8; k < b; ++k) {
            dist += static_cast<std::int32_t>(__builtin_popcountll(
                static_cast<std::uint64_t>(row[k] ^ q[k])));
        }
        out[i] = dist;
    }
}
