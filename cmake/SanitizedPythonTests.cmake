# Unsanitized Python must load GCC's sanitizer runtime before ctypes loads an
# instrumented library. Keep this scoped to the demonstrated Linux/GCC tests;
# native tests already link their runtime.
#
# ASan: also disable LeakSanitizer for these tests only. The interpreter is not
# instrumented, so its own allocations (PyObject_Malloc, PyType_GenericAlloc)
# are reported as leaks at exit and fail every Python-launched test. Native
# tests keep leak checking.
# TSan: without the preload, dlopen of the instrumented library fails with
# "libtsan.so.0: cannot allocate memory in static TLS block".
function(crispasr_preload_asan_for_python_tests)
    if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        return()
    endif()
    if(CRISPASR_SANITIZE_ADDRESS)
        set(runtime_name libasan.so)
        set(runtime_label ASan)
        set(extra_env "ASAN_OPTIONS=detect_leaks=0")
    elseif(CRISPASR_SANITIZE_THREAD)
        set(runtime_name libtsan.so)
        set(runtime_label TSan)
        set(extra_env "")
    else()
        return()
    endif()

    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=${runtime_name}
        RESULT_VARIABLE runtime_status OUTPUT_VARIABLE runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT runtime_status EQUAL 0 OR NOT IS_ABSOLUTE "${runtime}" OR NOT EXISTS "${runtime}")
        message(FATAL_ERROR "Cannot locate the compiler's ${runtime_label} runtime for Python tests")
    endif()

    set(preload "${runtime}")
    if(NOT "$ENV{LD_PRELOAD}" STREQUAL "")
        string(APPEND preload ":$ENV{LD_PRELOAD}")
    endif()
    foreach(test_name IN LISTS ARGN)
        if(TEST "${test_name}")
            set_property(TEST "${test_name}" APPEND PROPERTY ENVIRONMENT "LD_PRELOAD=${preload}")
            if(extra_env)
                set_property(TEST "${test_name}" APPEND PROPERTY ENVIRONMENT "${extra_env}")
            endif()
        endif()
    endforeach()
endfunction()
