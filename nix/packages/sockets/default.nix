{
  stdenv, fetchFromGitHub,

  cmake,

  qtbase, wrapQtAppsHook
}:

stdenv.mkDerivation rec {
  pname = "sockets";
  version = "0.1";

  src = fetchFromGitHub {
    owner = "mdfbaam";
    repo  = "ORNL-TCP-Sockets";
    rev   = "v${version}";
    hash  = "sha256-tCRAGfsTkJFVXTEFeYjDy0cGe5AwWdW2bL1/BdxsxcA=";
  };

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
  ];

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    qtbase
    wrapQtAppsHook
  ];

  meta = {
    description = "TCP Sockets built around Qt";
    homepage    = "https://github.com/mdfbaam/ORNL-TCP-Sockets";
  };
}
