# copy_if_exists.cmake — copy ${src} to ${dst} only if ${src} exists.
# Used for the optional .worker.js file that Emscripten may or may not
# emit depending on SINGLE_FILE mode.
if(EXISTS "${src}")
    file(COPY_FILE "${src}" "${dst}")
endif()
