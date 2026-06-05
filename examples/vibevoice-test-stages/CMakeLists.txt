set(TARGET vibevoice-test-stages)
add_executable(${TARGET} main.cpp)
include(DefaultTargetOptions)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE
    vibevoice
    crispasr-diff-lib
    Threads::Threads)
