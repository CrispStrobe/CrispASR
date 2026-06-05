if (CRISPASR_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # The llama.cpp source set that used to be compiled directly into
    # this executable now lives in the `crispasr-llama-core` STATIC lib
    # defined in src/CMakeLists.txt. Both this example and libcrispasr's
    # chat ABI (src/chat.cpp) link against it — one source set, two
    # consumers. See docs/prompts/chat-abi.md.
    set(TARGET crispasr-talk-llama)
    add_executable(${TARGET} talk-llama.cpp)
    target_include_directories(${TARGET} PRIVATE . ${SDL2_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE
        crispasr-llama-core
        common common-sdl whisper ${SDL2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)

    if(WIN32)
        # It requires Windows 8.1 or later for PrefetchVirtualMemory
        target_compile_definitions(${TARGET} PRIVATE -D_WIN32_WINNT=0x0602)
    endif()

    include(DefaultTargetOptions)
endif ()
