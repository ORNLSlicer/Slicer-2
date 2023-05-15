{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "ornl-tcp-sockets";
    version = "0.1";

    buildInputs = [
        pkgs.cmake
        pkgs.qt5.qtbase
        pkgs.pkg-config
    ];

    dontWrapQtApps = true;

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
        cp libORNL_TCP_Sockets.a $out/lib

        mkdir -p $out/include
        cd ../include
        cp * $out/include
    '';

    meta = {
        description = "A static library that provides a TCP server and client using Qt.";
        homepage = https://github.com/mdfbaam/ORNL-TCP-Sockets;
    };
}

