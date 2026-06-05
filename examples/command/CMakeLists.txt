if (CRISPASR_SDL2)
    set(TARGET crispasr-command)
    add_executable(${TARGET} command.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME BUNDLE DESTINATION bin)
endif ()
