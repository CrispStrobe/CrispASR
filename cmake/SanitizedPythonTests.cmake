# Unsanitized Python must load GCC's ASan runtime before ctypes loads an
# instrumented library. Keep this scoped to the demonstrated Linux/GCC tests;
# native tests already link their runtime, and other sanitizers are unchanged.
function(crispasr_preload_asan_for_python_tests)
    if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
            CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CRISPASR_SANITIZE_ADDRESS))
        return()
    endif()

    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libasan.so
        RESULT_VARIABLE runtime_status OUTPUT_VARIABLE runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT runtime_status EQUAL 0 OR NOT IS_ABSOLUTE "${runtime}" OR NOT EXISTS "${runtime}")
        message(FATAL_ERROR "Cannot locate the compiler's ASan runtime for Python tests")
    endif()

    set(preload "${runtime}")
    if(NOT "$ENV{LD_PRELOAD}" STREQUAL "")
        string(APPEND preload ":$ENV{LD_PRELOAD}")
    endif()
    foreach(test_name IN LISTS ARGN)
        if(TEST "${test_name}")
            set_property(TEST "${test_name}" APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${preload}")
        endif()
    endforeach()
endfunction()
