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

# the date of the commit
execute_process(COMMAND
    "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_DATE
    RESULT_VARIABLE GIT_DATE_RC
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT GIT_DATE_RC EQUAL 0 OR GIT_DATE STREQUAL "")
    set(GIT_DATE "unknown")
endif()

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
