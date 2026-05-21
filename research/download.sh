#!/bin/bash
download() {
  local id="$1" name="$2" dest="$3"
  local out="$dest/$name"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[skip] $out"
    return
  fi
  curl -sL --max-time 90 "https://arxiv.org/pdf/$id" -o "$out"
  if [ ! -s "$out" ] || ! file "$out" 2>/dev/null | grep -q PDF; then
    echo "[FAIL] $id -> $out"
    rm -f "$out"
  else
    echo "[ ok ] $id -> $out ($(du -h "$out" | cut -f1))"
  fi
}
export -f download
