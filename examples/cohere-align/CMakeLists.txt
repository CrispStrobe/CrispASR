set(TARGET cohere-align)

add_executable(${TARGET} main.cpp)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE cohere wav2vec2-ggml ctc-align whisper common Threads::Threads)
