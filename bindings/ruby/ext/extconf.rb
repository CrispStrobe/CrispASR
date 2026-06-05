require "mkmf"
require_relative "options"
require_relative "dependencies"

cmake = find_executable("cmake") || abort
options = Options.new(cmake).to_s
have_library("gomp") rescue nil
libs = Dependencies.new(cmake, options).to_s

$CFLAGS << " -O3 -march=native"
$INCFLAGS << " -Isources/include -Isources/ggml/include -Isources/examples"
$LOCAL_LIBS << " #{libs}"
$cleanfiles << " build #{libs}"

# libcrispasr.a and libcommon.a both translation-unit-include miniaudio's
# MINIAUDIO_IMPLEMENTATION + stb_vorbis.c (in src/crispasr_audio.cpp and
# examples/common-crispasr.cpp respectively). Apple ld silently picks the
# first definition; GNU ld errors out. The Ruby binding links both as
# static archives, so on Linux we ask the linker to behave like Apple's.
if RUBY_PLATFORM =~ /linux/
  $LDFLAGS << " -Wl,--allow-multiple-definition"
end

create_makefile "whisper" do |conf|
  conf << <<~EOF
    $(TARGET_SO): #{libs}
    #{libs}: cmake-targets
    cmake-targets:
    #{"\t"}#{cmake} -S sources -B build -D BUILD_SHARED_LIBS=OFF -D CMAKE_ARCHIVE_OUTPUT_DIRECTORY=#{__dir__} -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D CRISPASR_BUILD_TESTS=OFF -D CRISPASR_BUILD_EXAMPLES=OFF -D CRISPASR_BUILD_SERVER=OFF #{options}
    #{"\t"}#{cmake} --build build --config Release --target common crispasr-lib
  EOF
end
