{
  src, version,

  lib, stdenv,

  cmake, pkg-config, ninja, wrapQtAppsHook, deployQtWinPluginsHook,

  qtbase, qtcharts, qt5compat, assimp, boost184, cgal_5, eigen, nlohmann_json, gmp, mpfr,
  hdf5, vtk-qt, kuba-zip, clipper, psimpl, sockets
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
    nlohmann_json
    hdf5
    vtk-qt
    kuba-zip
    clipper
    psimpl
    sockets
    gmp
    mpfr
  ];

  cmakeFlags = [
    "-DSLICER2_AUTO_GENERATE_MASTER_CONFIG=OFF"
  ];

  nativeBuildInputs = [
    cmake
    pkg-config
    ninja
  ] ++ lib.optionals stdenv.isLinux [
    wrapQtAppsHook
  ] ++ lib.optionals stdenv.hostPlatform.isMinGW [
    deployQtWinPluginsHook
  ];

  dontWrapQtApps = stdenv.hostPlatform.isMinGW;

  qtWrapperArgs = lib.optionals stdenv.isLinux [
    "--suffix LD_FALLBACK_PATH : /usr/lib/x86_64-linux-gnu"
  ];

  meta = {
    description = "An advanced object slicer for toolpathing by ORNL";
    homepage = "https://github.com/ORNLSlicer/Slicer-2/";
    license = lib.licenses.gpl3;
    maintainers = with lib.maintainers; [
      cadkin
    ];
  };
}
