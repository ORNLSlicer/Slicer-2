{
  lib, stdenv,

  cmake,

  qtbase, clipper, boost184
}:

stdenv.mkDerivation rec {
  pname = "singlepath";
  version = "0.0.1";

  outputs = [
    "out"
    "dev"
  ];

  src = builtins.fetchGit {
    url = "git@github.com:mdfbaam/ORNL-Slicer-2-Single-Path.git";
    ref = "v${version}";
    rev = "aeabce0af77c9992e49febd084099b18d11e60d9";
  };

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    qtbase
    clipper
    boost184
  ];

  cmakeFlags = [
    (lib.cmakeBool "BUILD_SHARED_LIBS" true)
  ];

  dontWrapQtApps = true;
}
