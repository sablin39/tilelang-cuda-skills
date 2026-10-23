#!/usr/bin/env bash
# Install the TileLang/CUDA skill suite for agent CLIs (Kimi Code, Claude Code, Codex).
#
# Usage:
#   scripts/install-skills.sh [--scope project|global] [--target kimi|claude|codex|all]
#                             [--copy] [--check-only]
#
# Defaults: --scope project --target all, skills symlinked (single source of truth
# in this repo; git pull updates every installed CLI at once). --copy snapshots
# the files instead. Project scope installs into the current working directory;
# run it from the task project's root.
#
# The script also ships TileFoundry: it checks out the pinned submodule and its
# CUTLASS dependency at the suite's recorded gitlink and prints the commands
# that run or install it. It never updates TileFoundry to upstream main.
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

while [ $# -gt 0 ]; do
  case "$1" in
    --scope)      scope="$2"; shift 2 ;;
    --target)     target="$2"; shift 2 ;;
    --copy)       mode=copy; shift ;;
    --check-only) check_only=1; shift ;;
    -h|--help)    sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$scope" in project|global) ;; *) echo "--scope must be project or global" >&2; exit 2 ;; esac

warn() { echo "WARNING: $*" >&2; }

# --- Locate the suite ----------------------------------------------------------
# When the script runs standalone (curl | bash, or a lone download), the skills
# are not beside it; clone or update a cached checkout and install from there.
if [ -z "$SUITE_ROOT" ] || [ ! -d "$SKILLS_SRC" ]; then
  SUITE_ROOT="$SUITE_CACHE"
  SKILLS_SRC="$SUITE_CACHE/skills"
  if [ -d "$SUITE_CACHE/.git" ]; then
    echo "Updating cached suite at $SUITE_CACHE ..."
    git -C "$SUITE_CACHE" fetch --depth 1 origin "$SUITE_REF" \
      && git -C "$SUITE_CACHE" reset --hard FETCH_HEAD \
      || warn "could not update $SUITE_CACHE; installing from its current state"
  else
    echo "Fetching the skill suite into $SUITE_CACHE ..."
    git clone --depth 1 --branch "$SUITE_REF" "$SUITE_URL" "$SUITE_CACHE"
  fi
fi

# --- TileFoundry: ship the pinned source -------------------------------------
# The suite records TileFoundry as a submodule and the gitlink is the pin.
# Check out exactly that commit, with its CUTLASS dependency, and report it.
# Never `--remote`: the suite must not track upstream main.
TF="$SUITE_ROOT/external/TileFoundry"
tf_pin="$(git -C "$SUITE_ROOT" rev-parse HEAD:external/TileFoundry 2>/dev/null || true)"
tf_head() {
  [ -e "$TF/.git" ] || return 0
  git -C "$TF" rev-parse HEAD 2>/dev/null || true
}

if [ -z "$tf_pin" ]; then
  warn "no TileFoundry gitlink at $SUITE_ROOT; installing the skills without the pinned source"
else
  tf_now="$(tf_head)"
  if [ "$tf_now" = "$tf_pin" ] && [ -f "$TF/src/tilefoundry/__init__.py" ]; then
    echo "TileFoundry: $TF @ ${tf_pin:0:12} (pinned)"
  elif [ "$check_only" -eq 1 ]; then
    echo "TileFoundry: $TF — not at pin ${tf_pin:0:12} (have ${tf_now:-none})"
  else
    echo "Checking out TileFoundry at ${tf_pin:0:12} (with third_party/cutlass) ..."
    git -C "$SUITE_ROOT" submodule update --init --recursive -- external/TileFoundry \
      || { echo "ERROR: TileFoundry checkout failed" >&2; exit 1; }
    tf_now="$(tf_head)"
    [ "$tf_now" = "$tf_pin" ] \
      || { echo "ERROR: TileFoundry is not at pin ${tf_pin:0:12} after update" >&2; exit 1; }
    echo "TileFoundry: $TF @ ${tf_pin:0:12} (pinned)"
  fi
fi

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

if [ "$check_only" -eq 1 ]; then
  echo "Skills found:"
  discover_skills | sed 's/^/  /'
  exit 0
fi

if [ "$scope" = global ]; then
  echo "Installing into $GLOBAL_SKILLS (canonical, $mode):"
  while IFS= read -r skill; do
    install_one "$skill" "$GLOBAL_SKILLS"
  done < <(discover_skills)
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
    install_one "$KW" "$dest_dir"
  done
fi

cat <<EOF

Done. KernelWiki is a standalone skill at $KW, linked into every installed
skills directory; the suite's skills invoke it by name.

Symlinked installs track this repo; rerun with --copy for a snapshot, and rerun
after renaming or adding skills under skills/.
EOF

if [ -n "${tf_pin:-}" ]; then
  cat <<EOF

TileFoundry: $TF @ ${tf_pin:0:12} (pinned). The designated distribution is this
source checkout, not the PyPI wheel, which ships no CUDA headers.
  Run the CLI without installing:
    PYTHONPATH="$TF/src" python -m tilefoundry.cli --help
  Install editable into the environment that will run it:
    <env>/bin/python -m pip install -e "$TF" cmake ninja
    uv pip install --python <env>/bin/python -e "$TF" cmake ninja
EOF
fi
