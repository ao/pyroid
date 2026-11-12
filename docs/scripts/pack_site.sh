#!/bin/sh
# Pack the built docs site as the release asset pyroid-docs-<version>.tar.gz, and write its .sha256
# (hex digest, two spaces, file name: sha256sum's format). ataiva.com fetches both from the GitHub
# Release and serves the files under https://ataiva.com/open-source/pyroid/docs/.
#
#   sh docs/scripts/pack_site.sh <version> <out-dir> [<site-dir>]    # from the repository root
#
# <site-dir> defaults to site/, the output of `zensical build --clean --strict`. The archive holds the
# site's files at its root (index.html at the top level, no wrapping directory). Entries are sorted,
# owned by 0:0, dated SOURCE_DATE_EPOCH (default 0) and gzipped without a name or timestamp, so the same
# site gives the same bytes. Needs GNU tar.
set -eu

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: sh docs/scripts/pack_site.sh <version> <out-dir> [<site-dir>]" >&2
  exit 2
fi
version=$1
out=$2
site=${3:-site}

if [ ! -f "$site/index.html" ] || [ ! -f "$site/search.json" ]; then
  echo "error: $site/ has no index.html or search.json; build the site first" >&2
  exit 1
fi
# The asset is built for ataiva.com only: a site built with another site_url has wrong canonical links.
if ! grep -q 'rel="canonical" href="https://ataiva.com/open-source/pyroid/docs/"' "$site/index.html"; then
  echo "error: $site/index.html was not built with site_url https://ataiva.com/open-source/pyroid/docs/" >&2
  exit 1
fi

name="pyroid-docs-$version.tar.gz"
mkdir -p "$out"
list="$out/.pack_site.list"
(cd "$site" && find . -mindepth 1 | sed 's,^\./,,' | LC_ALL=C sort) > "$list"
tar -C "$site" --no-recursion --owner=0 --group=0 --numeric-owner --mtime="@${SOURCE_DATE_EPOCH:-0}" \
  -T "$list" -cf - | gzip -9n > "$out/$name"
rm -f "$list"
(cd "$out" && sha256sum "$name" > "$name.sha256")

files=$(tar -tzf "$out/$name" | grep -vc '/$')
bytes=$(wc -c < "$out/$name" | tr -d ' ')
echo "$out/$name: $files files, $bytes bytes"
cat "$out/$name.sha256"
