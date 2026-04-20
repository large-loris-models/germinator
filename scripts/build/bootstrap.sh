#!/usr/bin/env bash
# Germinator — bootstrap a fresh system with all OS-level dependencies.
#
# Installs: build essentials, clang/lld, z3, bazelisk, inotify-tools, python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"

echo "=== Germinator: bootstrapping system packages ==="

# ── Detect distro ───────────────────────────────────────────────────────────

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    DISTRO="$ID"
else
    echo "WARNING: cannot detect distro — assuming Debian/Ubuntu"
    DISTRO="ubuntu"
fi

if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
    echo "[1/5] Updating package lists..."
    sudo apt-get update -qq

    echo "[2/5] Installing build essentials..."
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        wget \
        pkg-config \
        zip \
        unzip \
        gcc-13 \
        libstdc++-13-dev

    echo "[3/5] Installing LLVM/clang toolchain..."
    sudo apt-get install -y -qq \
        clang \
        clang-tools \
        lld \
        llvm \
        llvm-dev \
        libclang-dev \
        libz-dev \
        zlib1g-dev \
        libtinfo-dev \
        libxml2-dev \
        libncurses5-dev

    echo "[4/5] Installing fuzzer / oracle / tooling deps..."
    sudo apt-get install -y -qq \
        libz3-dev \
        z3 \
        inotify-tools \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        jq \
        ccache \
        re2c
else
    echo "Unsupported distro: $DISTRO" >&2
    echo "Install manually: cmake, ninja, clang, lld, libz3-dev, z3, inotify-tools, python3" >&2
    exit 1
fi

# ── Bazelisk (Bazel version manager, needed for Centipede) ──────────────────

echo "[5/5] Installing Bazelisk..."
mkdir -p "$HOME/.local/bin"
if ! command -v bazel &>/dev/null; then
    curl -fsSL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
        -o "$HOME/.local/bin/bazel"
    chmod +x "$HOME/.local/bin/bazel"
    echo "  Installed Bazelisk to $HOME/.local/bin/bazel"
else
    echo "  Bazel already installed: $(command -v bazel)"
fi

# ── Verify ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Verifying installations ==="
for tool in cmake ninja git clang clang++ lld llvm-config python3 bazel z3 inotifywait; do
    if command -v "$tool" &>/dev/null; then
        printf "  %-16s %s\n" "$tool" "$(command -v "$tool")"
    else
        printf "  %-16s NOT FOUND\n" "$tool"
    fi
done

echo ""
echo "=== Bootstrap complete ==="
