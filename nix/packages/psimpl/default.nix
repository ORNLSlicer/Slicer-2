{
  stdenv, fetchFromGitHub,

  cmake
}:

stdenv.mkDerivation rec {
  pname = "psimpl";
  version = "7.0.1";

  src = fetchFromGitHub {
    owner = "mdfbaam";
    repo  = "psimpl";
    rev   = "${version}";
    hash  = "sha256-nZViPaHnNoMqBRiT+72nU084Osjaw+W0KHc2y+MmxQ8=";
  };

  nativeBuildInputs = [
    cmake
  ];

  meta = {
    description = "Generic n-dimensional polyline simplification";
    homepage    = "https://github.com/mdfbaam/psimpl";
  };
}
