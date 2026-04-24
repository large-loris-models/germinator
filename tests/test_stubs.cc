// test_stubs.cc
//
// Stub implementations of symbols that libgrlf-mlir.a and libsancov_runtime
// reference but are normally provided by the libFuzzer/Centipede runtime.
//
// These stubs are only used in the standalone test_mutations binary.
// They must NOT be linked into the real fuzz target.

#include <cstddef>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// LLVMFuzzerMutate — called by LibFuzzerTool::libfuzzer_mutate().
// In the real fuzz target this is provided by libFuzzer/Centipede.
// For the test binary we just return the input unchanged.
// ---------------------------------------------------------------------------
extern "C" size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  return Size;
}

// ---------------------------------------------------------------------------
// __sanitizer_cov_* — called by sancov-instrumented MLIR libs.
// In the real fuzz target these are provided by libsancov_runtime.pic.a.
// For the test binary we provide empty stubs so coverage is simply ignored.
// ---------------------------------------------------------------------------
extern "C" {

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop) {}
void __sanitizer_cov_trace_pc_guard(uint32_t *guard) {}
void __sanitizer_cov_8bit_counters_init(uint8_t *start, uint8_t *stop) {}
void __sanitizer_cov_pcs_init(const uintptr_t *pcs_beg,
                              const uintptr_t *pcs_end) {}
void __sanitizer_cov_trace_cmp1(uint8_t arg1, uint8_t arg2) {}
void __sanitizer_cov_trace_cmp2(uint16_t arg1, uint16_t arg2) {}
void __sanitizer_cov_trace_cmp4(uint32_t arg1, uint32_t arg2) {}
void __sanitizer_cov_trace_cmp8(uint64_t arg1, uint64_t arg2) {}
void __sanitizer_cov_trace_const_cmp1(uint8_t arg1, uint8_t arg2) {}
void __sanitizer_cov_trace_const_cmp2(uint16_t arg1, uint16_t arg2) {}
void __sanitizer_cov_trace_const_cmp4(uint32_t arg1, uint32_t arg2) {}
void __sanitizer_cov_trace_const_cmp8(uint64_t arg1, uint64_t arg2) {}
void __sanitizer_cov_trace_switch(uint64_t val, uint64_t *cases) {}
void __sanitizer_cov_trace_div4(uint32_t val) {}
void __sanitizer_cov_trace_div8(uint64_t val) {}
void __sanitizer_cov_trace_gep(uintptr_t idx) {}
void __sanitizer_cov_trace_pc() {}

} // extern "C"