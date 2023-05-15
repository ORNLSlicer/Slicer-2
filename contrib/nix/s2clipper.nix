{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "s2clipper";
    version = "512";

    buildInputs = [
        pkgs.cmake
        pkgs.pkg-config
    ];

    src = source;

    configurePhase = ''
        mkdir -p build
        cd build
        cmake ..
    '';

    buildPhase = ''
        make -j $NIX_BUILD_CORES
    '';

    installPhase = ''
        mkdir -p $out/lib
        cp libpolyclipping.a $out/lib

        mkdir -p $out/include
        cd ../include
        cp clipper.hpp $out/include
    '';

    meta = {
        description = "Version of clipperlib2 used by Slicer 2 before it became a proper project.";
        homepage = https://github.com/AngusJohnson/Clipper2;
    };
}

