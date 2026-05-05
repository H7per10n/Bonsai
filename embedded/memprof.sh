#!/bin/bash
# memprof.sh - Compares memory usage across different builds (dynamic linking)

# Configuration
SOURCE_FILE="verify.c"
OUTPUT_PREFIX="build"
OPTIMIZATIONS=("-O0" "-O1" "-O2" "-O3" "-Os")
TOOLCHAIN_PREFIX="arm-linux-gnueabihf-"

# Pi Zero 2W specific flags (ARMv6)
ARCH_FLAGS=""

# Clean previous builds
rm -f ${OUTPUT_PREFIX}_*

# Build loop
for i in {0..4}; do
    OPT=${OPTIMIZATIONS[$i]}
    OUTPUT="${OUTPUT_PREFIX}_${i}"

    echo -e "\n\033[1;34m=== Building with $OPT ===\033[0m"

    # Compile with dynamic linking and PI-specific flags
    ${TOOLCHAIN_PREFIX}gcc $ARCH_FLAGS $OPT -o $OUTPUT $SOURCE_FILE -lm
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $OPT"
        exit 1
    fi

    # Verify dynamic linking
    echo -n "Linkage: "
    file $OUTPUT | grep -q "dynamically linked" && echo "Dynamic" || echo "Static (unexpected)"

    # Memory analysis
    echo -e "\n\033[1;35mMemory Report for $OUTPUT:\033[0m"

    # Get section sizes
    declare -A SECTIONS
    while read -r section size _; do
        SECTIONS["$section"]=$size
    done < <(${TOOLCHAIN_PREFIX}size -A $OUTPUT | grep -E '\.text|\.rodata|\.data')

    # Calculate totals
    FLASH=$((${SECTIONS[.text]:-0} + ${SECTIONS[.rodata]:-0}))
    RAM=$((${SECTIONS[.data]:-0}))

    # Display report
    echo "Build $i ($OPT):"
    echo "--------------------------------"
    echo -e "Flash Usage:"
    echo -e "  .text   (Code): $((${SECTIONS[.text]:-0}/1024)) KB"
    echo -e "  .rodata (Const): $((${SECTIONS[.rodata]:-0}/1024)) KB"
    echo -e "  \033[1;32mTotal Flash: $((FLASH/1024)) KB\033[0m"
    echo -e "\nRAM Usage:"
    echo -e "  .data   (Vars): $((${SECTIONS[.data]:-0}/1024)) KB"
    echo -e "  \033[1;33mTotal RAM:  $((RAM/1024)) KB\033[0m"
    echo "--------------------------------"
done

echo -e "\n\033[1;32mDone! Created executables:\033[0m"
ls -lh ${OUTPUT_PREFIX}_* | awk '{print $5, $9}'
