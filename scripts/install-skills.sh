#!/usr/bin/env bash
# Install the TileLang/CUDA skill suite for agent CLIs (Kimi Code, Claude Code, Codex).
#
# Usage:
#   scripts/install-skills.sh [--scope project|global] [--target kimi|claude|codex|all]
#                             [--copy] [--check-only] [--python /path/to/venv/bin/python]
#
# Defaults: --scope project --target all, skills symlinked (single source of truth
# in this repo; git pull updates every installed CLI at once). --copy snapshots
# the files instead. Project scope installs into the current working directory;
# run it from the task project's root.
#
# The script installs the pinned TileFoundry core and bundled vendor packages,
# with CUTLASS source and cmake/ninja. By default it creates/reuses .venv in the
# suite; --python selects an existing virtual environment (Python 3.12+).
# It never updates TileFoundry to upstream main. CUDA execution needs host nvcc.
#
# The script also runs standalone, fetched without cloning the suite:
#   curl -fsSL https://raw.githubusercontent.com/sablin39/tilelang-cuda-skills/main/scripts/install-skills.sh | bash -s -- --scope global
# It then clones the suite into $SUITE_CACHE (default ~/.local/share/
# tilelang-cuda-skills) and symlinks from there; reruns update the cache.
# Override with $SUITE_URL / $SUITE_REF / $SUITE_CACHE.
#
# Global scope uses ~/.agents/skills as the canonical location (the most widely
# recognized skills convention) and mirrors it with symlinks from ~/.claude/skills
# and ~/.codex/skills. KernelWiki is a standalone skill dependency: per its
# official instructions it is cloned once so the clone root is the skill
# directory, at ~/.agents/skills/KernelWiki, and project-scope installs link it
# in from there.
#
# Network access (proxy variables included) is inherited from the caller's
# environment; set or unset ALL_PROXY/all_proxy before running the script as
# your network requires.
set -euo pipefail

# When piped (curl ... | bash), BASH_SOURCE is unset: fall through to the
# cache path instead of deriving a suite root from the caller's directory.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  SUITE_ROOT=""
fi
SKILLS_SRC="${SUITE_ROOT:+$SUITE_ROOT/skills}"
KERNELWIKI_URL="https://github.com/mit-han-lab/KernelWiki.git"
GLOBAL_SKILLS="$HOME/.agents/skills"
# Standalone (curl | bash) mode: the suite is fetched into a cache and the
# skills symlinked from there, so reruns and git pulls update every CLI.
SUITE_URL="${SUITE_URL:-https://github.com/sablin39/tilelang-cuda-skills.git}"
SUITE_REF="${SUITE_REF:-main}"
SUITE_CACHE="${SUITE_CACHE:-$HOME/.local/share/tilelang-cuda-skills}"

scope=project
target=all
mode=symlink
check_only=0
tf_python=""

usage() {
  cat <<'EOF'
Usage: install-skills.sh [--scope project|global] [--target kimi|claude|codex|all]
                         [--copy] [--check-only] [--python /path/to/venv/bin/python]
Installs skills and pinned TileFoundry into the suite's .venv by default.
--python selects an existing Python 3.12+ virtual environment.
--check-only validates files and that environment offline without changing them.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --scope|--target|--python)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "ERROR: $1 requires a value" >&2; exit 2; } ;;
  esac
  case "$1" in
    --scope)      scope="$2"; shift 2 ;;
    --target)     target="$2"; shift 2 ;;
    --copy)       mode=copy; shift ;;
    --check-only) check_only=1; shift ;;
    --python)    tf_python="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$scope" in project|global) ;; *) echo "--scope must be project or global" >&2; exit 2 ;; esac
case "$target" in kimi|claude|codex|all) ;; *) echo "--target must be kimi, claude, codex, or all" >&2; exit 2 ;; esac

warn() { echo "WARNING: $*" >&2; }

# --- Locate the suite ----------------------------------------------------------
# When the script runs standalone (curl | bash, or a lone download), the skills
# are not beside it; clone or update a cached checkout and install from there.
if [ -z "$SUITE_ROOT" ] || [ ! -d "$SKILLS_SRC" ]; then
  SUITE_ROOT="$SUITE_CACHE"
  SKILLS_SRC="$SUITE_CACHE/skills"
  if [ "$check_only" -eq 1 ]; then
    [ -d "$SKILLS_SRC" ] || { echo "ERROR: no cached suite at $SUITE_CACHE" >&2; exit 1; }
  elif [ -d "$SUITE_CACHE/.git" ]; then
    echo "Updating cached suite at $SUITE_CACHE ..."
    git -C "$SUITE_CACHE" fetch --depth 1 origin "$SUITE_REF" \
      && git -C "$SUITE_CACHE" reset --hard FETCH_HEAD \
      || { echo "ERROR: could not update $SUITE_CACHE; installation stopped" >&2; exit 1; }
  else
    echo "Fetching the skill suite into $SUITE_CACHE ..."
    git clone --depth 1 --branch "$SUITE_REF" "$SUITE_URL" "$SUITE_CACHE"
  fi
fi

TF="$SUITE_ROOT/external/TileFoundry"
tf_env_default=0
if [ -z "$tf_python" ]; then
  tf_python="$SUITE_ROOT/.venv/bin/python"
  tf_env_default=1
fi
# Keep the virtual environment's executable path (do not resolve its symlink).
if [[ "$tf_python" != */* ]]; then
  tf_python="$(command -v "$tf_python" || true)"
fi
if [ -n "$tf_python" ] && [[ "$tf_python" != /* ]]; then
  tf_python="$PWD/$tf_python"
fi
tf_pin="$(git -C "$SUITE_ROOT" rev-parse HEAD:external/TileFoundry 2>/dev/null || true)"
tf_source_ready() {
  local modules
  [ -n "$tf_pin" ] && [ -e "$TF/.git" ] &&
    [ "$(git -C "$TF" rev-parse HEAD 2>/dev/null)" = "$tf_pin" ] &&
    [ -f "$TF/src/tilefoundry/__init__.py" ] &&
    [ -f "$TF/targets/nvidia/pyproject.toml" ] &&
    [ -f "$TF/targets/apple/pyproject.toml" ] &&
    [ -f "$TF/third_party/cutlass/include/cute/tensor.hpp" ] || return 1
  modules="$(git -C "$TF" submodule status --recursive)" || return 1
  ! printf '%s\n' "$modules" | grep -q '^[+-U]'
}

# Check-only is read-only and offline: do not initialize submodules, fetch the
# cache, repair installations, or execute dependency query scripts. Inspect
# package metadata/import locations without importing vendor implementations.
if [ "$check_only" -eq 1 ]; then
  case "$target" in
    all) check_targets=(kimi claude codex) ;;
    kimi|claude|codex) check_targets=("$target") ;;
    *) echo "--target must be kimi, claude, codex, or all" >&2; exit 2 ;;
  esac
  check_status=0
  python3 "$SUITE_ROOT/scripts/skill_layout.py" check "$SKILLS_SRC" || check_status=1
  for cli in "${check_targets[@]}"; do
    if [ "$scope" = global ]; then check_base="$HOME"; else check_base="$PWD"; fi
    case "$cli" in
      kimi) check_dest="$check_base/.agents/skills" ;;
      claude) check_dest="$check_base/.claude/skills" ;;
      codex) check_dest="$check_base/.codex/skills" ;;
    esac
    python3 "$SUITE_ROOT/scripts/skill_layout.py" check "$SKILLS_SRC" \
      --installed "$check_dest" || check_status=1
    if [ ! -f "$check_dest/KernelWiki/SKILL.md" ] || \
       [ ! -f "$check_dest/KernelWiki/scripts/query.py" ] || \
       [ ! -d "$check_dest/KernelWiki/wiki" ]; then
      echo "ERROR: KernelWiki missing/incomplete at $check_dest/KernelWiki" >&2
      check_status=1
    fi
  done
  if ! tf_source_ready; then
    echo "ERROR: TileFoundry or CUTLASS missing/not at suite pin" >&2
    check_status=1
  fi
  if [ ! -x "$tf_python" ]; then
    echo "ERROR: TileFoundry environment missing: $tf_python; run installation first" >&2
    check_status=1
  else
    PYTHONDONTWRITEBYTECODE=1 "$tf_python" "$SUITE_ROOT/scripts/install_tilefoundry.py" \
      --source "$TF" --pin "$tf_pin" --check-only || check_status=1
  fi
  echo "Checked files, links and TileFoundry environment; confirm active agent discovery separately."
  exit "$check_status"
fi

# --- TileFoundry: install the pinned source and packages ----------------------
# The suite records TileFoundry as a submodule and the gitlink is the pin.
# Check out exactly that commit, with its CUTLASS dependency, and report it.
# Never `--remote`: the suite must not track upstream main.
if [ -z "$tf_pin" ]; then
  echo "ERROR: no TileFoundry gitlink at $SUITE_ROOT" >&2; exit 1
fi
if ! tf_source_ready; then
  echo "Checking out TileFoundry at ${tf_pin:0:12} (with third_party/cutlass) ..."
  git -C "$SUITE_ROOT" submodule update --init --recursive -- external/TileFoundry \
    || { echo "ERROR: TileFoundry checkout failed" >&2; exit 1; }
  tf_source_ready || { echo "ERROR: TileFoundry/CUTLASS incomplete or not at pin" >&2; exit 1; }
fi
if [ ! -x "$tf_python" ] && [ "$tf_env_default" -eq 1 ]; then
  bootstrap_python=""
  for candidate in python3 python3.12 python3.13 python3.14; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
      bootstrap_python="$candidate"; break
    fi
  done
  if [ -z "$bootstrap_python" ]; then
    echo "ERROR: install Python 3.12+ or pass --python /path/to/venv/bin/python" >&2; exit 1
  fi
  "$bootstrap_python" -m venv "$SUITE_ROOT/.venv"
fi
[ -x "$tf_python" ] || { echo "ERROR: Python executable not found: $tf_python" >&2; exit 1; }
PYTHONDONTWRITEBYTECODE=1 "$tf_python" "$SUITE_ROOT/scripts/install_tilefoundry.py" \
  --source "$TF" --pin "$tf_pin"

# --- KernelWiki: locate, verify, repair --------------------------------------
# KernelWiki is a standalone skill, cloned once at the canonical
# ~/.agents/skills/KernelWiki and linked into every installed skills directory.
# A checkout is usable when its structure is present and its query script
# answers (official smoke test first, --help as fallback). A checkout with no
# readable commit or one behind the remote is removed and reinstalled.

kw_head() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
kw_rev() { local h; h="$(kw_head "$1")"; echo "${h:-none}"; }

kw_usable() { # $1=checkout path
  local kw="$1"
  [ -f "$kw/SKILL.md" ] && [ -f "$kw/scripts/query.py" ] && [ -d "$kw/wiki" ] || return 1
  python3 "$kw/scripts/query.py" --tag nvfp4 --type kernel --compact >/dev/null 2>&1 && return 0
  python3 "$kw/scripts/query.py" --help >/dev/null 2>&1
}

kw_fresh() { # $1=checkout path; 0 when a commit exists and matches the remote
  local kw="$1" local_head remote_head
  local_head="$(kw_head "$kw")"
  [ -n "$local_head" ] || return 1
  remote_head="$(git ls-remote "$KERNELWIKI_URL" HEAD 2>/dev/null | cut -f1 || true)"
  if [ -z "$remote_head" ]; then
    warn "cannot reach $KERNELWIKI_URL; keeping KernelWiki at $local_head"
    return 0
  fi
  [ "$local_head" = "$remote_head" ]
}

# Official installation: clone so the clone root IS the skill directory.
kw_clone() { # $1=destination
  rm -rf "$1"
  git clone --depth 1 "$KERNELWIKI_URL" "$1"
}

# Remove and reinstall a broken or stale checkout in place.
kw_repair() { # $1=checkout path
  kw_clone "$1"
  kw_usable "$1"
}

KW="$GLOBAL_SKILLS/KernelWiki"
ensure_kernelwiki() {
  if [ -d "$KW" ]; then
    if kw_usable "$KW" && kw_fresh "$KW"; then
      echo "KernelWiki: $KW (rev $(kw_head "$KW")) — usable, up to date"
    elif [ "$check_only" -eq 1 ]; then
      echo "KernelWiki: $KW — present but corrupted or stale (rev $(kw_rev "$KW"))"
    else
      warn "KernelWiki at $KW has no/older commit ($(kw_rev "$KW")); reinstalling"
      kw_repair "$KW" || { echo "ERROR: KernelWiki reinstall failed" >&2; exit 1; }
    fi
  elif [ "$check_only" -eq 1 ]; then
    echo "KernelWiki: $KW — NOT installed"
  else
    echo "Installing KernelWiki per official instructions (clone root = skill dir) ..."
    mkdir -p "$GLOBAL_SKILLS"
    kw_clone "$KW"
    kw_usable "$KW" || { echo "ERROR: fresh KernelWiki clone failed the smoke test" >&2; exit 1; }
  fi
}

ensure_kernelwiki

# --- Discover skills: dirs containing SKILL.md, skipping nested duplicates ---

discover_skills() {
  find "$SKILLS_SRC" -name SKILL.md -print0 | while IFS= read -r -d '' f; do
    local dir
    dir="$(dirname "$f")"
    # skip if an ancestor inside skills/ is itself a skill
    local parent="$dir"
    local nested=0
    while [ "$parent" != "$SKILLS_SRC" ]; do
      parent="$(dirname "$parent")"
      [ -f "$parent/SKILL.md" ] && { nested=1; break; }
    done
    [ "$nested" -eq 0 ] && echo "$dir"
  done
}

# --- Install -----------------------------------------------------------------

# Install one skill, repairing whatever is already at the destination:
# broken symlinks are removed, real directories conflicting with symlink mode
# are replaced (with a warning), stale copies are refreshed.
install_one() { # $1=skill source dir, $2=target skills dir
  local src="$1" dest="$2/$(basename "$1")"
  [ "$src" -ef "$dest" ] 2>/dev/null && return 0
  mkdir -p "$2"
  if [ -L "$dest" ] && [ ! -e "$dest" ]; then
    warn "removing broken symlink $dest"; rm -f "$dest"
  elif [ -e "$dest" ] && [ ! -L "$dest" ] && [ "$mode" = symlink ]; then
    warn "replacing pre-installed directory $dest with a symlink"; rm -rf "$dest"
  fi
  if [ "$mode" = symlink ]; then
    ln -sfn "$src" "$dest"
  else
    rm -rf "$dest"
    cp -R "$src" "$dest"
  fi
  echo "  installed $(basename "$src") -> $dest"
}

# Mirror the canonical ~/.agents/skills entries into another CLI's skills dir.
mirror_skills() { # $1=link dir (e.g. ~/.claude/skills)
  local entry link
  mkdir -p "$1"
  for entry in "$GLOBAL_SKILLS"/*; do
    link="$1/$(basename "$entry")"
    if [ -L "$link" ] && [ ! -e "$link" ]; then
      warn "removing broken symlink $link"; rm -f "$link"
    elif [ -e "$link" ] && [ ! -L "$link" ]; then
      warn "replacing pre-installed directory $link with a symlink to ~/.agents"; rm -rf "$link"
    fi
    ln -sfn "$entry" "$link"
    echo "  linked $link -> $entry"
  done
}

project_skills_dir() { # $1=cli
  case "$1" in
    kimi)   echo "$PWD/.agents/skills" ;;
    claude) echo "$PWD/.claude/skills" ;;
    codex)  echo "$PWD/.codex/skills" ;;
  esac
}

targets=()
case "$target" in
  all)   targets=(kimi claude codex) ;;
  kimi|claude|codex) targets=("$target") ;;
  *) echo "--target must be kimi, claude, codex, or all" >&2; exit 2 ;;
esac

if [ "$scope" = global ]; then
  echo "Installing into $GLOBAL_SKILLS (canonical, $mode):"
  while IFS= read -r skill; do
    install_one "$skill" "$GLOBAL_SKILLS"
  done < <(discover_skills)
  if [ "$mode" = copy ]; then
    python3 "$SUITE_ROOT/scripts/skill_layout.py" relocate "$SKILLS_SRC" --installed "$GLOBAL_SKILLS"
  fi
  for cli in "${targets[@]}"; do
    case "$cli" in
      kimi)   : ;;                      # reads ~/.agents/skills natively
      claude) echo "Mirroring to $HOME/.claude/skills:"; mirror_skills "$HOME/.claude/skills" ;;
      codex)  echo "Mirroring to $HOME/.codex/skills:";  mirror_skills "$HOME/.codex/skills" ;;
    esac
  done
else
  for cli in "${targets[@]}"; do
    dest_dir="$(project_skills_dir "$cli")"
    echo "Installing into $dest_dir ($cli, project scope, $mode):"
    while IFS= read -r skill; do
      install_one "$skill" "$dest_dir"
    done < <(discover_skills)
    if [ "$mode" = copy ]; then
      python3 "$SUITE_ROOT/scripts/skill_layout.py" relocate "$SKILLS_SRC" --installed "$dest_dir"
    fi
    install_one "$KW" "$dest_dir"
  done
fi

cat <<EOF

Done. KernelWiki is a standalone skill at $KW, linked into every installed
skills directory; the suite's skills invoke it by name.

Symlinked installs track this repo; rerun with --copy for a snapshot, and rerun
after renaming or adding skills under skills/.
EOF

cat <<EOF

TileFoundry installed and verified: $TF @ ${tf_pin:0:12}.
  Python: $tf_python
  Activate its environment or use its bin/tilefoundry command.
  CUDA execution also requires a compatible driver and host CUDA toolkit (nvcc).
EOF
