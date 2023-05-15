{ pkgs, source }:

pkgs.stdenv.mkDerivation rec {
    pname = "qtxlsxwriter";
    version = "0.3";

    buildInputs = [
        pkgs.cmake
        pkgs.pkg-config
        pkgs.qt5.qtbase
    ];

    dontWrapQtApps = true;

    src = source;


    configurePhase = ''
        mkdir -p build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX:PATH=$out ..
    '';

    buildPhase = ''
        make -j $NIX_BUILD_CORES
    '';

    installPhase = ''
        make install
    '';

    meta = {
        description = ".xlsx file reader and writer for Qt5";
        homepage = https://github.com/VSRonin/QtXlsxWriter;
    };
}

