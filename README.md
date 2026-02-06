## From Source

better aigc lib

mkdir src && git -C src clone https://github.com/TorrentBrave/aigco.git

git submodule add --force https://github.com/TorrentBrave/aigco.git src/aigco

uv add --editable ./src/aigco/ <!-- will update aigco.egg.info -->

cd src/aigco

uv lock

uv sync

## From PypI
uv pip install aigco[flash_attn]
or add to deependencies
"flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl"
