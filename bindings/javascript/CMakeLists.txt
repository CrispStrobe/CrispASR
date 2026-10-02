set(TARGET libwhisper)

add_executable(${TARGET}
    emscripten.cpp
    )

target_link_libraries(${TARGET} PRIVATE
    whisper
    )

unset(EXTRA_FLAGS)

if (CRISPASR_WASM_SINGLE_FILE)
    set(EXTRA_FLAGS "-s SINGLE_FILE=1")
    message(STATUS "Embedding WASM inside crispasr.js")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/libwhisper.js
        ${CMAKE_CURRENT_SOURCE_DIR}/crispasr.js
        )

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/libwhisper.js
        ${CMAKE_CURRENT_SOURCE_DIR}/whisper.js
        )

    # With SINGLE_FILE=1, modern Emscripten inlines the pthread worker
    # into the main .js and does NOT emit a separate .worker.js file.
    # Only copy it when the file actually exists.
    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -Dsrc=${CMAKE_BINARY_DIR}/bin/libwhisper.worker.js
            -Ddst=${CMAKE_CURRENT_SOURCE_DIR}/crispasr.worker.js
            -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_if_exists.cmake
        )

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -Dsrc=${CMAKE_BINARY_DIR}/bin/libwhisper.worker.js
            -Ddst=${CMAKE_CURRENT_SOURCE_DIR}/libwhisper.worker.js
            -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_if_exists.cmake
        )
endif()

# Multithreaded needs a big pre-reserved shared memory (growing shared memory mid-compute stalls in
# a browser worker). Single-threaded uses a normal (growable) memory, so a small initial is fine.
if (CRISPASR_WASM_SINGLE_THREAD)
    set(WASM_THREAD_FLAGS "-s INITIAL_MEMORY=512MB")
elseif (CRISPASR_WASM_PROXY_TO_PTHREAD)
    # Experiment: run the runtime on a proxied pthread so the servicer thread stays free.
    set(WASM_THREAD_FLAGS "-s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=8 -s PROXY_TO_PTHREAD=1 -s INITIAL_MEMORY=2GB")
else()
    set(WASM_THREAD_FLAGS "-s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=0 -s INITIAL_MEMORY=2GB")
endif()

# Opt-in browser memory experiments; retain upstream defaults until decoded
# output and under-identical-load A/B measurements justify changing them.
set(CRISPASR_WASM_INITIAL_MEMORY "" CACHE STRING "Override initial WASM heap, e.g. 128MB")
set(CRISPASR_WASM_THREAD_POOL "" CACHE STRING "Override prestarted pthread pool size")
if (CRISPASR_WASM_INITIAL_MEMORY)
    string(REGEX REPLACE "INITIAL_MEMORY=[^ ]+" "INITIAL_MEMORY=${CRISPASR_WASM_INITIAL_MEMORY}" WASM_THREAD_FLAGS "${WASM_THREAD_FLAGS}")
endif()
if (CRISPASR_WASM_THREAD_POOL AND NOT CRISPASR_WASM_SINGLE_THREAD)
    string(REGEX REPLACE "PTHREAD_POOL_SIZE=[^ ]+" "PTHREAD_POOL_SIZE=${CRISPASR_WASM_THREAD_POOL}" WASM_THREAD_FLAGS "${WASM_THREAD_FLAGS}")
endif()

# C2PA (opt-in) needs native wasm exceptions at LINK time too — but only when
# enabled, since the module is compiled with -fwasm-exceptions only then (a
# mismatch between compile and link EH modes would break the default build).
set(WASM_C2PA_LINK_FLAGS "")
if (CRISPASR_C2PA_ENABLED)
    set(WASM_C2PA_LINK_FLAGS "-fwasm-exceptions -s SUPPORT_LONGJMP=wasm")
endif()

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    ${WASM_C2PA_LINK_FLAGS} \
    -s MODULARIZE=1 \
    -s EXPORT_NAME=\"'whisper_factory'\" \
    -s FORCE_FILESYSTEM=1 \
    ${WASM_THREAD_FLAGS} \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s STACK_SIZE=16MB \
    -s MAXIMUM_MEMORY=4GB \
    ${EXTRA_FLAGS} \
    ")
