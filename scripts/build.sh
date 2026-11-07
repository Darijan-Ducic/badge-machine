#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

app="build/Baijiu Machine.app"
resources="$app/Contents/Resources"
iconset="build/AppIcon.iconset"
mkdir -p "$app/Contents/MacOS" "$resources" "$iconset"
cp Assets/Info.plist "$app/Contents/Info.plist"

swiftc -target "$(uname -m)-apple-macosx14.0" -O -parse-as-library Sources/*.swift \
    -o "$app/Contents/MacOS/Baijiu Machine"

sips -s format png Assets/Mark.svg --out "$resources/Mark.png" >/dev/null
sed -e '/<rect /d' -e 's/0 0 1024 1024/160 160 704 704/' Assets/Mark.svg > build/MenuMark.svg
sips -s format png -z 68 68 build/MenuMark.svg --out "$resources/MenuMark.png" >/dev/null
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$resources/Mark.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    retina=$((size * 2))
    sips -z "$retina" "$retina" "$resources/Mark.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$resources/AppIcon.icns"
codesign --force --sign - "$app"
