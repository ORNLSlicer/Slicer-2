{
  lib, stdenv, fetchFromGitHub,

  cmake,

  qtbase, wrapQtAppsHook
}:

stdenv.mkDerivation rec {
  pname = "sockets";
  version = "0.1.1";

  src = fetchFromGitHub {
    owner = "mdfbaam";
    repo  = "ORNL-TCP-Sockets";
    rev   = "v${version}";
    hash  = "sha256-1hL4DpqSUhJFSe2yDmt/LDse1cH/nfCev6j9n4zuJUw=";
  };

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
  ];

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    qtbase
  ] ++ lib.optionals stdenv.isLinux [
    wrapQtAppsHook
  ];

  dontWrapQtApps = stdenv.hostPlatform.isMinGW;

  meta = {
    description = "TCP Sockets built around Qt";
    homepage    = "https://github.com/mdfbaam/ORNL-TCP-Sockets";
  };
}
