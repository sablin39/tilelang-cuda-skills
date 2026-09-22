#!/usr/bin/env bash
# Install the TileLang/CUDA skill suite for agent CLIs (Kimi Code, Claude Code, Codex).
#
# Usage:
#   scripts/install-skills.sh [--scope project|global] [--target kimi|claude|codex|all]
#                             [--copy] [--kernelwiki PATH] [--pypi-mirror URL]
#                             [--no-hmz] [--check-only]
#
# Defaults: --scope project --target all, skills symlinked (single source of truth
# in this repo; git pull updates every installed CLI at once). --copy snapshots
# the files instead. Project scope installs into the current working directory;
# run it from the task project's root.
#
# The script also runs standalone, fetched without cloning the suite:
#   curl -fsSL https://raw.githubusercontent.com/sablin39/tilelang-cuda-skills/main/scripts/install-skills.sh | bash -s -- --scope global
# It then clones the suite into $SUITE_CACHE (default ~/.local/share/
# tilelang-cuda-skills) and symlinks from there; reruns update the cache.
# Override with $SUITE_URL / $SUITE_REF / $SUITE_CACHE.
#
# Global scope uses ~/.agents/skills as the canonical location (the most widely
# recognized skills convention) and mirrors it with symlinks from ~/.claude/skills
# and ~/.codex/skills. KernelWiki is then installed following its official
# instruction: a standalone clone whose root is the skill directory.
#
# Unless --no-hmz is given, the humanize orchestrator (hmz) is installed via uv
# when missing; uv is a hard requirement in that case and the script stops
# immediately without it. Its Python dependencies resolve through --pypi-mirror
# (default: Aliyun; override with the flag or $PYPI_MIRROR). Network access
# (proxy variables included) is inherited from the caller's environment; set or
# unset ALL_PROXY/all_proxy before running the script as your network requires.
set -euo pipefail

SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_SRC="$SUITE_ROOT/skills"
KERNELWIKI_URL="https://github.com/mit-han-lab/KernelWiki.git"
HMZ_PKG="hmz[all] @ git+https://github.com/humanfia/humanize.git"
HMZ_URL="https://github.com/humanfia/humanize.git"
GLOBAL_SKILLS="$HOME/.agents/skills"
# Standalone (curl | bash) mode: the suite is fetched into a cache and the
# skills symlinked from there, so reruns and git pulls update every CLI.
SUITE_URL="${SUITE_URL:-https://github.com/sablin39/tilelang-cuda-skills.git}"
SUITE_REF="${SUITE_REF:-main}"
SUITE_CACHE="${SUITE_CACHE:-$HOME/.local/share/tilelang-cuda-skills}"

scope=project
target=all
mode=symlink
kw_arg=""
with_hmz=1
check_only=0
pypi_mirror="${PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple}"

while [ $# -gt 0 ]; do
  case "$1" in
    --scope)      scope="$2"; shift 2 ;;
    --target)     target="$2"; shift 2 ;;
    --copy)       mode=copy; shift ;;
    --kernelwiki) kw_arg="$2"; shift 2 ;;
    --pypi-mirror) pypi_mirror="$2"; shift 2 ;;
    --no-hmz)     with_hmz=0; shift ;;
    --check-only) check_only=1; shift ;;
    -h|--help)    sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$scope" in project|global) ;; *) echo "--scope must be project or global" >&2; exit 2 ;; esac

warn() { echo "WARNING: $*" >&2; }

# --- Locate the suite ----------------------------------------------------------
# When the script runs standalone (curl | bash, or a lone download), the skills
# are not beside it; clone or update a cached checkout and install from there.
if [ ! -d "$SKILLS_SRC" ]; then
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

# --- Requirements ------------------------------------------------------------

if [ "$with_hmz" -eq 1 ] && ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required to check and install hmz but was not found on PATH. Stop." >&2
  exit 1
fi

for cli in kimi claude codex; do
  if command -v "$cli" >/dev/null 2>&1; then
    echo "backend: $cli — found ($(command -v "$cli"))"
  else
    warn "backend CLI '$cli' not on PATH; hmz cannot drive it until installed and logged in"
  fi
done

# --- KernelWiki: locate, verify, repair --------------------------------------
# A checkout is usable when its structure is present and its query script
# answers (official smoke test first, --help as fallback). A checkout with no
# readable commit, one behind the remote, or — for the suite's pinned
# submodule — one off the recorded pin, is removed and reinstalled.

kw_head() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
kw_rev() { local h; h="$(kw_head "$1")"; echo "${h:-none}"; }

kw_usable() { # $1=checkout path
  local kw="$1"
  [ -f "$kw/SKILL.md" ] && [ -f "$kw/scripts/query.py" ] && [ -d "$kw/wiki" ] || return 1
  python3 "$kw/scripts/query.py" --tag nvfp4 --type kernel --compact >/dev/null 2>&1 && return 0
  python3 "$kw/scripts/query.py" --help >/dev/null 2>&1
}

kw_fresh() { # $1=checkout path; 0 when a commit exists and is where it should be
  local kw="$1" local_head remote_head pin
  local_head="$(kw_head "$kw")"
  [ -n "$local_head" ] || return 1
  # The suite's submodule is pinned deliberately: it is fresh when it matches
  # the superproject's recorded pin, not when it matches the remote.
  if [ "$kw" = "$SUITE_ROOT/external/KernelWiki" ]; then
    pin="$(git -C "$SUITE_ROOT" ls-tree HEAD -- external/KernelWiki 2>/dev/null | awk '{print $3}')"
    if [ -n "$pin" ]; then
      [ "$local_head" = "$pin" ] && return 0
      warn "KernelWiki submodule is at $local_head but the suite pins $pin; reinitializing to the pin"
      return 1
    fi
  fi
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
  local kw="$1"
  if [ "$kw" = "$SUITE_ROOT/external/KernelWiki" ]; then
    rm -rf "$kw"
    git -C "$SUITE_ROOT" submodule update --init -- external/KernelWiki
  else
    kw_clone "$kw"
  fi
  kw_usable "$kw"
}

KW=""
ensure_kernelwiki() {
  if [ "$scope" = global ]; then
    KW="$GLOBAL_SKILLS/KernelWiki"
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
  else
    local candidate
    for candidate in "$kw_arg" "${KERNELWIKI_ROOT:-}" "$SUITE_ROOT/external/KernelWiki" "$GLOBAL_SKILLS/KernelWiki" "$HOME/.local/share/KernelWiki"; do
      [ -n "$candidate" ] && [ -d "$candidate" ] || continue
      if kw_usable "$candidate" && kw_fresh "$candidate"; then
        KW="$candidate"; break
      fi
      if [ "$check_only" -eq 1 ]; then
        echo "KernelWiki: $candidate — corrupted or stale (rev $(kw_rev "$candidate")), would reinstall"
        continue
      fi
      warn "KernelWiki at $candidate has no/older commit ($(kw_rev "$candidate")); reinstalling"
      if kw_repair "$candidate"; then
        KW="$candidate"; break
      fi
    done
    if [ -z "$KW" ] && [ "$check_only" -eq 0 ]; then
      if git -C "$SUITE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$SUITE_ROOT" submodule update --init -- external/KernelWiki
        kw_usable "$SUITE_ROOT/external/KernelWiki" && KW="$SUITE_ROOT/external/KernelWiki"
      fi
      [ -z "$KW" ] && { kw_clone "$HOME/.local/share/KernelWiki"; kw_usable "$HOME/.local/share/KernelWiki" && KW="$HOME/.local/share/KernelWiki"; }
    fi
    [ -z "$KW" ] && [ "$check_only" -eq 0 ] && { echo "ERROR: no usable KernelWiki checkout could be established" >&2; exit 1; }
    [ -n "$KW" ] && echo "KernelWiki: $KW (rev $(kw_head "$KW")) — usable"
  fi
}

# --- humanize (hmz) orchestrator ---------------------------------------------
# Same contract as KernelWiki: broken or older than the remote is reinstalled;
# current is left alone. Every case prints its status. Presence and integrity
# are uv's own call: the tool venv under `uv tool dir`, verified with
# `uv pip check`, which validates the whole dependency closure and needs no
# hmz on PATH. The installed revision is the git commit recorded in uv's
# direct_url.json; if it cannot be read, hmz is treated as broken and
# reinstalled, which also heals non-uv installs into a revision-readable one.

hmz_venv() { echo "$(uv tool dir 2>/dev/null)/hmz"; }

hmz_installed() { [ -d "$(hmz_venv)" ]; }

hmz_usable() { # 0 when uv's integrity check passes on the tool venv
  local venv
  venv="$(hmz_venv)"
  [ -x "$venv/bin/python" ] || return 1
  uv pip check --python "$venv/bin/python" >/dev/null 2>&1
}

hmz_head() {
  local json
  for json in "$(hmz_venv)"/lib/python3.*/site-packages/hmz-*.dist-info/direct_url.json; do
    [ -f "$json" ] || continue
    sed -n 's/.*"commit_id": *"\([0-9a-f]\{7,\}\)".*/\1/p' "$json"
    return 0
  done
}

hmz_rev() { local h; h="$(hmz_head)"; echo "${h:-unknown}"; }

hmz_fresh() { # 0 when a revision is readable and matches the remote
  local local_head remote_head
  local_head="$(hmz_head)"
  [ -n "$local_head" ] || return 1
  remote_head="$(git ls-remote "$HMZ_URL" HEAD 2>/dev/null | cut -f1 || true)"
  if [ -z "$remote_head" ]; then
    warn "cannot reach $HMZ_URL; keeping hmz at $local_head"
    return 0
  fi
  [ "$local_head" = "$remote_head" ]
}

hmz_install() {
  uv tool install --force --index-url "$pypi_mirror" "$HMZ_PKG"
  command -v hmz >/dev/null 2>&1 || warn "hmz installed but not on PATH; run: uv tool update-shell"
  hmz_usable || warn "hmz still fails uv's integrity check after install"
}

ensure_hmz() {
  if ! hmz_installed; then
    if [ "$check_only" -eq 1 ]; then echo "hmz: NOT installed"; return 0; fi
    echo "Installing hmz with uv (index: $pypi_mirror) ..."
    hmz_install
  elif ! hmz_usable; then
    if [ "$check_only" -eq 1 ]; then echo "hmz: present but failed uv's integrity check; would reinstall"; return 0; fi
    warn "hmz failed uv's integrity check; reinstalling"
    hmz_install
  elif hmz_fresh; then
    echo "hmz: $(hmz_venv) (rev $(hmz_rev)) — up to date"
  else
    if [ "$check_only" -eq 1 ]; then echo "hmz: older or unreadable revision ($(hmz_rev)); would reinstall"; return 0; fi
    warn "hmz is an older or unreadable revision ($(hmz_rev)); reinstalling"
    hmz_install
  fi
}

ensure_kernelwiki
[ "$with_hmz" -eq 1 ] && ensure_hmz

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
  [ "$KW" != "$GLOBAL_SKILLS/KernelWiki" ] && install_one "$KW" "$GLOBAL_SKILLS"
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

Done. For agents that query KernelWiki from outside this suite, export:

  export KERNELWIKI_ROOT="$KW"

Symlinked installs track this repo; rerun with --copy for a snapshot, and rerun
after renaming or adding skills under skills/.
EOF
