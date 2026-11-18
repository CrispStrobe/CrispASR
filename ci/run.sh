#!/bin/bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -vf $OUT/*.log
rm -vf $OUT/*.exit
rm -vf $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

ALL_MODELS=( "tiny.en" "tiny" "base.en" "base" "small.en" "small" "medium.en" "medium" "large-v1" "large-v2" "large-v3" "large-v3-turbo" )
BENCH_N_THREADS=4
BENCH_ENCODER_ONLY=0
BENCH_FLASH_ATTN=0

# check for user-specified models first. if not specified, use fast models
if [ ! -z ${GG_BUILD_TEST_MODELS} ]; then
    IFS=',' read -r -a MODELS <<< "${GG_BUILD_TEST_MODELS}"
else
    if [ ! -z ${GG_BUILD_LOW_PERF} ]; then
        MODELS=( "tiny" "base" "small" )
    else
        MODELS=("${ALL_MODELS[@]}")
    fi
fi

CMAKE_EXTRA="-DCRISPASR_FATAL_WARNINGS=ON"

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON"

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
        else
            echo "Warning: Using fallback CUDA architectures"
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
        fi
    else
        echo "Error: nvidia-smi not found, cannot build with CUDA"
        exit 1
    fi
fi

if [ ! -z ${GG_BUILD_ROCM} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_HIP=ON"
    if [ -z ${GG_BUILD_AMDGPU_TARGETS} ]; then
        echo "Missing GG_BUILD_AMDGPU_TARGETS, please set it to your GPU architecture (e.g. gfx90a, gfx1100, etc.)"
        exit 1
    fi

    CMAKE_EXTRA="${CMAKE_EXTRA} -DAMDGPU_TARGETS=${GG_BUILD_AMDGPU_TARGETS}"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:"
        echo "source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
    # Use only main GPU
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    # Enable sysman for correct memory reporting
    export ZES_ENABLE_SYSMAN=1
    # to circumvent precision issues on CPY operations
    export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
fi

if [ ! -z ${GG_BUILD_VULKAN} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_VULKAN=1"

    # if on Mac, disable METAL
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi

fi

if [ ! -z ${GG_BUILD_WEBGPU} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_WEBGPU=1"
fi

if [ ! -z ${GG_BUILD_MUSA} ]; then
    # Use qy1 by default (MTT S80)
    MUSA_ARCH=${MUSA_ARCH:-21}
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=${MUSA_ARCH}"
fi

if [ ! -z ${GG_BUILD_NO_SVE} ]; then
    # arm 9 and newer enables sve by default, adjust these flags depending on the cpu used
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm"
fi

if [ ! -z ${GG_BUILD_NO_AVX512} ]; then
    # GitHub-hosted x86_64 runners are drawn from a heterogeneous CPU pool —
    # some have AVX512 / AVX512_VBMI / AVX512_VNNI, some don't. With
    # GGML_NATIVE=ON the build picks up whatever the build host has, then
    # the compiled binary SIGILLs if a later step (bench, test) runs on a
    # different pool VM without those features. Forcing AVX512 OFF (and
    # disabling native detection so the build doesn't quietly re-enable it)
    # pins a uniform AVX2 + FMA baseline that every x86_64 runner can execute.
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX2=ON -DGGML_FMA=ON"
fi

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_download_model {
    local model_name=$1
    local model_file="$MNT/models/ggml-${model_name}.bin"

    if [ ! -f ${model_file} ]; then
        mkdir -p "$MNT/models" || return $?
        (
            cd "$MNT/models" || exit $?
            bash "$SRC/models/download-ggml-model.sh" "$model_name" .
        )
    fi
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

# Pass PIPESTATUS immediately after the pipeline: local assignments would
# otherwise replace the producer's failure with their own successful status.
function gg_check_command_status {
    local exit_status=$1
    local log_status=$2
    local exit_file=$3
    local command_name=$4

    if [ "$exit_status" -eq 0 ]; then
        exit_status=$log_status
    fi
    if printf '%s\n' "$exit_status" > "$exit_file"; then
        :
    else
        local write_status=$?
        if [ "$exit_status" -eq 0 ]; then
            exit_status=$write_status
        fi
    fi

    if [ $exit_status -ne 0 ]; then
        echo "Error: Command $command_name failed with exit status $exit_status"
        return "$exit_status"
    fi

    return 0
}

function gg_run_logged {
    local log_file=$1
    local exit_file=$2
    local command_name=$3
    shift 3

    { time "$@"; } 2>&1 | tee -a "$log_file"
    gg_check_command_status "${PIPESTATUS[0]}" "${PIPESTATUS[1]}" "$exit_file" "$command_name"
}

# Usage: gg_run <test_name> [additional_args...]
#
# Parameters:
#   test_name       - Name of the test to run (calls gg_run_<test_name>)
#   additional_args - Any additional arguments to pass to the test function (first argument is appended to the log filename)
function gg_run {
    local ci=$1
    local cur

    if [ $# -gt 1 ]; then
        ci="${ci}_${2}"
    fi

    set -o pipefail
    set -x

    gg_run_$1 "$@" | tee $OUT/$ci.log
    if gg_check_command_status "${PIPESTATUS[0]}" "${PIPESTATUS[1]}" "$OUT/$ci.exit" "$ci"; then
        cur=0
    else
        cur=$?
    fi

    set +x
    set +o pipefail

    gg_sum_$1 "$@"

    ret=$((ret | cur))
    return "$cur"
}

function gg_check_build_requirements {
    if ! command -v cmake &> /dev/null; then
        gg_printf 'cmake not found, please install'
        return 127
    fi

    if ! command -v make &> /dev/null; then
        gg_printf 'make not found, please install'
        return 127
    fi
}

## ci

function gg_run_ctest {
    mode=$2

    cd "$SRC" || return $?

    rm -rf "build-ci-${mode}" && mkdir "build-ci-${mode}" && cd "build-ci-${mode}" || return $?

    gg_check_build_requirements || return $?

    gg_run_logged "$OUT/${ci}-cmake.log" "$OUT/${ci}-cmake.exit" configure \
        cmake -DCMAKE_BUILD_TYPE=${mode} -DPython3_EXECUTABLE="$(command -v python3)" ${CMAKE_EXTRA} .. || return $?
    gg_run_logged "$OUT/${ci}-make.log" "$OUT/${ci}-make.exit" build \
        make -j$(nproc) || return $?

    # `-L main` was upstream whisper.cpp's label convention; NOTHING in this
    # tree carries it, so this ctest ran ZERO tests while the job stayed green
    # (found in the 2026-09-02 orphaned-label audit, same failure mode as the
    # dead `cli` label). `unit` is the no-model suite that runs everywhere —
    # and this is the only place it runs in a Debug build.
    gg_run_logged "$OUT/${ci}-ctest.log" "$OUT/${ci}-ctest.exit" ctest \
        ctest --output-on-failure -L unit -E test-opt || return $?
}

function gg_sum_ctest {
    mode=$2

    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in '${mode}' mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

function gg_run_bench {
    cd "$SRC" || return $?

    # set flash attention flag if enabled
    fattn="-nfa"
    if [ "$BENCH_FLASH_ATTN" -eq 1 ]; then
        fattn="-fa"
    fi

    # run memcpy benchmark if not encoder-only mode
    if [ "$BENCH_ENCODER_ONLY" -eq 0 ]; then
        echo "Running memcpy benchmark"
        gg_run_logged "$OUT/${ci}-memcpy.log" "$OUT/${ci}-memcpy.exit" "memcpy benchmark" \
            ./build-ci-release/bin/crispasr-bench -w 1 -t $BENCH_N_THREADS || return $?

        echo "Running ggml_mul_mat benchmark with $BENCH_N_THREADS threads"
        gg_run_logged "$OUT/${ci}-mul_mat.log" "$OUT/${ci}-mul_mat.exit" "ggml_mul_mat benchmark" \
            ./build-ci-release/bin/crispasr-bench -w 2 -t $BENCH_N_THREADS || return $?
    fi

    echo "Running benchmark for all models"

    # generate header for the benchmark table
    {
        printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" "Config" "Model" "Th" "FA" "Enc." "Dec." "Bch5" "PP" "Commit"
        printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" "---" "---" "---" "---" "---" "---" "---" "---" "---"
    } | tee -a $OUT/${ci}-models-table.log
    gg_check_command_status "${PIPESTATUS[0]}" "${PIPESTATUS[1]}" "$OUT/${ci}-header.exit" "benchmark table" || return $?

    # run benchmark for each model
    for model in "${MODELS[@]}"; do
        echo "Benchmarking model: $model"

        # run the benchmark and capture output
        output=$(./build-ci-release/bin/crispasr-bench -m $MNT/models/ggml-$model.bin -t $BENCH_N_THREADS $fattn 2>&1)
        local model_status=$?

        # save the raw output
        echo "$output" > $OUT/${ci}-bench-$model.log

        if [ $model_status -eq 0 ]; then
            # parse the benchmark results
            encode_time=$(echo "$output" | grep "encode time" | awk '{print $11}')
            decode_time=$(echo "$output" | grep "decode time" | awk '{print $11}')
            batchd_time=$(echo "$output" | grep "batchd time" | awk '{print $11}')
            prompt_time=$(echo "$output" | grep "prompt time" | awk '{print $11}')
            system_info=$(echo "$output" | grep "system_info")
            actual_threads=$(echo "$output" | grep "system_info" | awk '{print $4}')

            # determine configuration
            config=""
            if [[ $system_info == *"AVX2 = 1"* ]]; then
                config="$config AVX2"
            fi
            if [[ $system_info == *"NEON = 1"* ]]; then
                config="$config NEON"
            fi
            if [[ $system_info == *"BLAS = 1"* ]]; then
                config="$config BLAS"
            fi
            if [[ $system_info == *"COREML = 1"* ]]; then
                config="$config COREML"
            fi
            if [[ $system_info == *"CUDA = 1"* ]]; then
                config="$config CUDA"
            fi
            if [[ $system_info == *"METAL = 1"* ]]; then
                config="$config METAL"
            fi

            # get commit hash
            commit=$(git rev-parse --short HEAD)

            # add row to benchmark table
            printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" \
                "$config" "$model" "$actual_threads" "$BENCH_FLASH_ATTN" "$encode_time" "$decode_time" "$batchd_time" "$prompt_time" "$commit" \
                | tee -a $OUT/${ci}-models-table.log
        else
            echo "Benchmark failed for model: $model" | tee -a $OUT/${ci}-bench-errors.log
            return "$model_status"
        fi
    done

    return 0
}

function gg_sum_bench {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Whisper Benchmark Results\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"

    # show memcpy and ggml_mul_mat benchmark results if available
    if [ "$BENCH_ENCODER_ONLY" -eq 0 ]; then
        if [ -f "$OUT/${ci}-memcpy.log" ]; then
            gg_printf '#### memcpy Benchmark\n\n'
            gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-memcpy.log)"
        fi

        if [ -f "$OUT/${ci}-mul_mat.log" ]; then
            gg_printf '#### ggml_mul_mat Benchmark\n\n'
            gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-mul_mat.log)"
        fi
    fi

    # show model benchmark results
    gg_printf '#### Model Benchmarks\n\n'
    if [ -f "$OUT/${ci}-models-table.log" ]; then
        gg_printf '%s\n\n' "$(cat $OUT/${ci}-models-table.log)"
    else
        gg_printf 'No model benchmark results available.\n\n'
    fi

    # show any errors that occurred
    if [ -f "$OUT/${ci}-bench-errors.log" ]; then
        gg_printf '#### Benchmark Errors\n\n'
        gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-bench-errors.log)"
    fi
}

ret=0

for model in "${MODELS[@]}"; do
    if gg_download_model "$model"; then
        :
    else
        ret=$?
        break
    fi
done

test $ret -eq 0 && gg_run ctest debug
test $ret -eq 0 && gg_run ctest release

# Run the same performance checks only after both test builds pass. Missing
# binaries, crashes and log failures must remain failures, not timing results.
test $ret -eq 0 && gg_run bench

if [ -f "$OUT/README.md" ]; then
    cat "$OUT/README.md"
fi

exit $ret
