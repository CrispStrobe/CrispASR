# Loaded by CTest after Catch's generated discovery files. At this point each
# TEST_LIST contains real cases and ordinary quoted list properties are safe.
set(test-align-ab-bench_LABELS "align")
set(test-conformer-ab-bench_LABELS "conformer")
set(test-cpu-ops-ab-bench_LABELS "cpu_ops;convt")
set(test-firered-ab-bench_LABELS "firered;f5_tts")
set(test-voxcpm2-ab-bench_LABELS "voxcpm2;roformer")
set(test-kokoro-diarize-ab-bench_LABELS "kokoro;diarize")
set(test-cqt-diarize-ab-bench_LABELS "cqt;diarize")
set(test-rvq-istft-ab-bench_LABELS "rvq;istft")

foreach(_target IN ITEMS
    test-align-ab-bench test-conformer-ab-bench test-cpu-ops-ab-bench
    test-firered-ab-bench test-voxcpm2-ab-bench test-kokoro-diarize-ab-bench
    test-cqt-diarize-ab-bench test-rvq-istft-ab-bench)
    if(${_target}_UNIT_TESTS)
        set_tests_properties(${${_target}_UNIT_TESTS} PROPERTIES
            LABELS "unit;performance-parity;${${_target}_LABELS}" ENVIRONMENT "CRISPASR_RUN_PERF=0")
    endif()
    if(${_target}_BENCH_TESTS)
        set_tests_properties(${${_target}_BENCH_TESTS} PROPERTIES
            LABELS "bench;performance-timing;${${_target}_LABELS}" ENVIRONMENT "CRISPASR_RUN_PERF=1")
    endif()
endforeach()

foreach(_target IN ITEMS
    test-core-rvq test-core-istft test-core-cqt test-spectral-diarize
    test-diarize-smooth test-parallel-for test-kaldi-fbank test-arktts-cache
    test-meanvc2-conditioning test-voxcpm2-cpu-conv)
    if(${_target}_TESTS)
        set_tests_properties(${${_target}_TESTS} PROPERTIES LABELS "unit;performance-parity")
    endif()
endforeach()
if(CTC_UNIT_TESTS)
    set_tests_properties(${CTC_UNIT_TESTS} PROPERTIES LABELS "unit;performance-parity;ctc")
endif()
if(CTC_BENCH_TESTS)
    set_tests_properties(${CTC_BENCH_TESTS} PROPERTIES LABELS "bench;ctc")
endif()
if(test-core-istft-blas_TESTS)
    set_tests_properties(${test-core-istft-blas_TESTS} PROPERTIES LABELS "unit;performance-parity;blas")
endif()
if(test-parallel-for-openmp_TESTS)
    set_tests_properties(${test-parallel-for-openmp_TESTS} PROPERTIES LABELS "unit;performance-parity;openmp")
endif()
