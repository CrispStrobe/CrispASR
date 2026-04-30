set(TARGET crispasr-legacy-quantize)
add_executable(${TARGET} quantize.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common crispasr ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)
