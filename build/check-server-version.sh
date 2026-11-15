#!/usr/bin/env bash
# Usage: bash build/check-server-version.sh <tag>. The tag of a release of the service is server-v and the version of its project, a
# preview; the README names the image of that version.
set -euo pipefail
tag="$1"
version=$(dotnet msbuild src/ChromaDotNet.Embeddings.Server -getProperty:Version -nologo)
echo "tag $tag, version $version"
test "$tag" = "server-v$version" || { echo "the tag is not server-v$version" >&2; exit 1; }
case "$version" in *-preview*) ;; *) echo "only previews for now" >&2; exit 1 ;; esac
grep -qF "ghcr.io/chromadotnet/embeddings-server:$version" src/ChromaDotNet.Embeddings.Server/README.md \
  || { echo "the README does not name the image $version" >&2; exit 1; }
