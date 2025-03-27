{
  src, version,

  lib, stdenv,

  cmake, ninja, wrapQtAppsHook, git,

  qtbase, qtcharts, qt5compat, assimp, boost184, cgal_5, eigen, gmp, nlohmann_json, mpfr,
  hdf5, vtk-qt, tbb, kuba-zip, clipper, psimpl, sockets
}:

stdenv.mkDerivation rec {
  pname = "slicer2";
  inherit version;
  inherit src;

  buildInputs = [
    qtbase
    qtcharts
    qt5compat
    assimp
    boost184
    cgal_5
    eigen
    gmp
    nlohmann_json
    mpfr
    hdf5
    vtk-qt
    tbb
    kuba-zip
    clipper
    psimpl
    sockets
  ];

  cmakeFlags = [
    "-DSLICER2_AUTO_GENERATE_MASTER_CONFIG=OFF"
  ];

  nativeBuildInputs = [
    cmake
    ninja
    wrapQtAppsHook
    git
  ];

  qtWrapperArgs = lib.optionals stdenv.isLinux [
    "--suffix LD_FALLBACK_PATH : /usr/lib/x86_64-linux-gnu"
  ];

  meta = {
  };
}
