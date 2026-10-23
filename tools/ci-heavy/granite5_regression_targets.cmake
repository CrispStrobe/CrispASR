# Add only the public stage probe to a hosted CPU regression build.
# Loaded through CMAKE_PROJECT_crispasr_INCLUDE, so no unrelated test targets
# or fixture generators need to be configured.
add_executable(granite5-stage-probe
    "${PROJECT_SOURCE_DIR}/tools/ci-heavy/granite5_stage_probe.cpp")
target_include_directories(granite5-stage-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(granite5-stage-probe PRIVATE granite_speech ggml-base)
target_compile_features(granite5-stage-probe PRIVATE cxx_std_17)
set_target_properties(granite5-stage-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
