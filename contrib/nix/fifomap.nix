{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "nlohmann_fifomap";
    version = "1.0";

    buildInputs = [
        # NOP
    ];

    src = source;

    installPhase = ''
        mkdir -p $out/include
        cp src/fifo_map.hpp $out/include
    '';

    meta = {
        description = "A FIFO-ordered associative container for C++";
        homepage = https://github.com/nlohmann/fifo_map;
    };
}

