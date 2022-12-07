#! /bin/bash

tmp=$(mktemp -d)

echo "$tmp"

cp -r src "$tmp"/.
cp -r src_core "$tmp"/.
cp -r src_kernels "$tmp"/.
cp -r resources "$tmp"/.
cp -r LICENSE README.md "$tmp"/.

### Publish the 2D version.
sed 's#\.\./\.\./src_core#src_core#g' crates/sparkl2d-core/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish
cd "$currdir" || exit

sed 's#\.\./\.\./src_kernels#src_kernels#g' crates/sparkl2d-kernels/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish --no-verify
cd "$currdir" || exit

sed 's#\.\./\.\./src#src#g' crates/sparkl2d/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish
cd "$currdir" || exit


### Publish the 3D version.
sed 's#\.\./\.\./src_core#src_core#g' crates/sparkl3d-core/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish
cd "$currdir" || exit

sed 's#\.\./\.\./src_kernels#src_kernels#g' crates/sparkl3d-kernels/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish --no-verify
cd "$currdir" || exit

sed 's#\.\./\.\./src#src#g' crates/sparkl3d/Cargo.toml > "$tmp"/Cargo.toml
currdir=$(pwd)
cd "$tmp" && cargo publish
cd "$currdir" || exit

rm -rf "$tmp"
