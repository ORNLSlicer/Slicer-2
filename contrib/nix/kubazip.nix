{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "kubazip";
    version = "0.2.3";

    buildInputs = [
        pkgs.cmake
        pkgs.pkg-config
    ];

    src = source;

    configurePhase = ''
        mkdir -p build
        cd build
        cmake -DBUILD_SHARED_LIBS=true ..
    '';

    buildPhase = ''
        make -j $NIX_BUILD_CORES
    '';

    installPhase = ''
        mkdir -p $out/lib
        cp libzip.so $out/lib

        mkdir -p $out/include
        cd ../src
        cp miniz.h zip.h $out/include
    '';

    meta = {
        description = "A portable, simple zip library written in C";
        homepage = https://github.com/kuba--/zip;
    };
}

