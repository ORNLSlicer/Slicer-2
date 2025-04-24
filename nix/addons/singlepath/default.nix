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
    rev = "5b9cf5650cdc37dbda49bb82e1fc8c4c01f9e51b";
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
