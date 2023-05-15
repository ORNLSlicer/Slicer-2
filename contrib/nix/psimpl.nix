{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "psimpl";
    version = "v7";

    buildInputs = [
        # NOP
    ];

    src = source;

    installPhase = ''
        mkdir -p $out/include
        cp psimpl.h $out/include
    '';

    meta = {
        description = "Generic n-dimensional polyline simplification library for C++";
        homepage = https://sourceforge.net/projects/psimpl/;
    };
}

