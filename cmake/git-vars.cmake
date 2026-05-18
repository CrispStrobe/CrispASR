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
#
# We initially mapped `(` → `[` and `)` → `]`, BUT square brackets
# trigger CMake/Ninja's special-quote path on the linux-x86_64 generator:
# the Ninja DEFINES line wraps the whole bracket-containing definition
# in outer `"..."`, which absorbs subsequent `;`-separated definitions
# into the same `-D` value. The compiler then sees
#   -DCRISPASR_GIT_SUBJECT="\"...[regression]...\";CRISPASR_BUILD_DATE=\"...\";..."
# and fails with
#   <command-line>: error: assignment of read-only location ‘"..."’
#   <command-line>: error: expected ‘)’ before ‘;’ token
# (CI run 26059062109 caught this on commit 4127aaa0).
#
# So map everything bracket-ish to `.` instead. Visual cost: a subject
# like "fix(scope): ..." reads "fix.scope.- ..." in --version output
# — slightly noisier but Ninja-safe.
string(REPLACE "("  "."  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE ")"  "."  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
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
# CMake evaluates `$<...>` generator expressions inside
# target_compile_definitions VALUES — even when those values came from
# a string variable. A commit subject like
#   "fix(build): pass chatterbox via $<TARGET_FILE:> so CMake ..."
# would otherwise survive into the compile definition and trip
#   Error evaluating generator expression: $<TARGET_FILE->
#   Expression did not evaluate to a known generator expression
# Map to `.` (same Ninja-safe replacement as the parentheses above —
# square brackets trigger the special-quote bug described there).
string(REPLACE "<"  "."  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
string(REPLACE ">"  "."  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
# `#` is dropped by CMake's preprocessor-definition machinery (warning
# "Preprocessor definitions containing '#' may not be passed on the
# compiler command line"), AND POSIX shells treat `#` as a comment
# inside unquoted tokens. A subject like "fix(#107)" produces a
# truncated -D value at the `#`, the leftover closing quote then
# absorbs subsequent flags. Replace with "no." so "(#107)" reads
# "[no.107]" in `--version` output.
string(REPLACE "#"  "no."  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
# Spaces in the value cause CMake's target_compile_definitions
# generator to wrap the spaceful definition AND all subsequent
# definitions of the same call into one `-D` flag's value:
#
#   -DCRISPASR_GIT_SUBJECT="\"... has spaces ...\";CRISPASR_BUILD_DATE=\"...\";..."
#
# The C preprocessor then sees CRISPASR_GIT_SUBJECT followed by an
# unexpected `;` token + the rest of the definitions as orphaned
# macro bodies, and crispasr_diagnostics.cpp fails to compile with
#   error: assignment of read-only location ‘"May 18 2026 20:45:10"’
#   error: expected ‘)’ before ‘;’ token
# (the BUILD_DATE fallback collapses into the GIT_SUBJECT value).
# Same defence as the GIT_DATE sanitiser above ("Defensive: even if
# some future caller sets a custom format with spaces"). Underscores
# survive every shell we care about and `--version` output reads
# `test[regression]-_add_TTS-]ASR` — a small readability tax for a
# CI-wide build break.
string(REPLACE " "  "_"  GIT_COMMIT_SUBJECT "${GIT_COMMIT_SUBJECT}")
