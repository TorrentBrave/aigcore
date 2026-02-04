better aigc lib

mkdir src && git -C src clone https://github.com/TorrentBrave/aigco.git

git submodule add --force https://github.com/TorrentBrave/aigco.git src/aigco

uv add --editable ./src/aigco/ <!-- will update aigco.egg.info -->
