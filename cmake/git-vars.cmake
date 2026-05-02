find_package(Git)

# the commit's SHA1
execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=8
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_SHA1
    RESULT_VARIABLE GIT_SHA1_RC
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GIT_SHA1_RC EQUAL 0 OR GIT_SHA1 STREQUAL "")
    set(GIT_SHA1 "unknown")
endif()

# the date of the commit. Use ISO-8601 STRICT — `--date=local` produces
# "Sat May 2 22:41:36 2026 +0200" which has spaces AND localised English
# month names. When that value gets embedded as -DCRISPASR_GIT_DATE="..."
# and the build runs through Windows MSVC's link.exe, the value gets
# split at the first space and "May" is interpreted as a filename:
#
#   LINK : fatal error LNK1181: cannot open input file 'May.obj'
#
# CMake's quoting on the cmake side IS correct, but link.exe re-parses
# /D values when expanding command files. ISO-8601 strict has no spaces
# and no localized substrings, so it survives the round-trip.
execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%ad --date=iso-strict
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_DATE
    RESULT_VARIABLE GIT_DATE_RC
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GIT_DATE_RC EQUAL 0 OR GIT_DATE STREQUAL "")
    set(GIT_DATE "unknown")
endif()
# Defensive: even if some future caller sets a custom format with
# spaces, replace them with underscores so the value remains a single
# token. Same defense pattern the COMMIT_SUBJECT sanitiser uses below.
string(REPLACE " " "_" GIT_DATE "${GIT_DATE}")

# the subject of the commit. Sanitize for safe embedding in a -D define:
# CMake treats ";" as a list separator and a literal ";" inside the value
# of target_compile_definitions splits the definition mid-string, leaving
# unbalanced quotes that the make-shell then tries to parse (the leftover
# "(scope):" of a Conventional-Commit subject ends up looking like a
# subshell, hence `/bin/sh: Syntax error: "(" unexpected`). Replace ";"
# with "," and strip backslashes / double-quotes that would also break
# the C string literal.
execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%s
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
    RESULT_VARIABLE GIT_SUBJECT_RC
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GIT_SUBJECT_RC EQUAL 0 OR GIT_COMMIT_SUBJECT STREQUAL "")
    set(GIT_COMMIT_SUBJECT "unknown")
endif()
string(REPLACE ";"  ","  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE "\\" "/"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE "\"" "'"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
# MSVC /D flag and POSIX sh both choke on parentheses and colons in the
# value (e.g. "fix(test): ..." → LNK1146 on MSVC, unterminated quote on
# sh). Neutralize them.
string(REPLACE "("  "["  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE ")"  "]"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE ":"  "-"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
# `=` makes Windows link.exe split the value when it spills onto the
# linker command line — `GIT_DATE = ISO-8601` becomes positional arg
# `GIT_DATE` → `LNK1181: cannot open input file 'GIT_DATE.obj'`. The
# previous landmine of this kind ("May 02") was cleared by switching
# the date format; this one came from a commit message containing `=`.
# Bracket sanitisation is defensive — even though the broken behaviour
# only manifested under MSVC's dependency-scan path with a CUDA-archs
# quoting bug nearby, future shells may surface other splits.
string(REPLACE "="  "-"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
