{
  lib, stdenv, fetchFromGitHub,

  cmake
}:

stdenv.mkDerivation rec {
  pname = "kubazip";
  version = "0.3.3";

  src = fetchFromGitHub {
    owner = "kuba--";
    repo  = "zip";
    rev   = "v${version}";
    hash  = "sha256-SyV5QJFSs4YKXPcZ3FmTc2t4q3nCzlp/u7vobVMBV6A=";
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    # NOP
  ] ++ lib.optionals stdenv.hostPlatform.isMinGW [
    "-DCMAKE_DISABLE_TESTING=ON"
  ];

  meta = {
    description = "A portable, simple zip library written in C";
    homepage    = "https://github.com/kuba--/zip";
  };
}
