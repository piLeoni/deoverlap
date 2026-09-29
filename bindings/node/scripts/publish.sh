#!/bin/bash
# Publish the npm/<platform> packages, then deoverlap itself. Skips versions
# already on the registry, so it can be re-run after a failure.
# usage: scripts/publish.sh [otp]
cd "$(dirname "$0")/.." || exit 1
OTP_FLAG=""
if [ -n "$1" ]; then OTP_FLAG="--otp=$1"; fi

# npm view caches 404s for new packages for minutes; ask the registry directly.
published() {
  curl -s -H 'Cache-Control: no-cache' "https://registry.npmjs.org/$1?t=$(date +%s)$RANDOM" \
    | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{process.exit(JSON.parse(s).versions?.[process.argv[1]]?0:1)}catch{process.exit(1)}})' "$2"
}

for dir in npm/* .; do
  name=$(node -p "require('./$dir/package.json').name")
  ver=$(node -p "require('./$dir/package.json').version")
  if published "$name" "$ver"; then
    echo "skip $name@$ver (already published)"
    continue
  fi
  if [ "$dir" != "." ] && ! ls "$dir"/*.node >/dev/null 2>&1; then
    echo "missing binary in $dir: run npm run artifacts first"
    exit 1
  fi
  echo "publishing $name@$ver"
  (cd "$dir" && npm publish --access public --ignore-scripts $OTP_FLAG)
  for _ in 1 2 3 4 5; do published "$name" "$ver" && break; sleep 3; done
  if ! published "$name" "$ver"; then
    echo "FAILED $name@$ver — run this script again to resume"
    exit 1
  fi
  echo "done $name@$ver"
done
echo "ALL DONE"
