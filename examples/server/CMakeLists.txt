set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET crispasr-server)
add_executable(${TARGET} server.cpp ws_stream.cpp httplib.h ws_stream.h)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${TARGET} PRIVATE common json_cpp whisper crispasr-lib ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    # MSVC auto-links these via #pragma comment in src/crispasr_cache.cpp
    # and httplib.h, but MinGW (msys2 UCRT64 / CLANG64) ignores those
    # pragmas, so the link fails with __imp_WinHttp* / WSASocketW / etc.
    # undefined. Listing them explicitly mirrors the working set used by
    # examples/cli/CMakeLists.txt so every executable that statically
    # links libcrispasr resolves uniformly.
    target_link_libraries(${TARGET} PRIVATE
        winhttp
        ws2_32
        crypt32
        bcrypt
        iphlpapi
    )
endif()

install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)
