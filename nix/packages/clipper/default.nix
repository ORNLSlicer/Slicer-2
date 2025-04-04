{
  stdenv, fetchFromGitHub,

  cmake
}:

stdenv.mkDerivation rec {
  pname = "clipper";
  version = "512am";

  src = fetchFromGitHub {
    owner = "mdfbaam";
    repo  = "clipper";
    rev   = "v${version}";
    hash  = "sha256-OldI2Hqd5sriENMQjhLr/ZoNJrIV505NMomUWs4y5ok=";
  };

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=OFF"
  ];

  nativeBuildInputs = [
    cmake
  ];

  meta = {
    description = "Angus Johnson's clipper v1, with modifications made for ORNL.";
    homepage    = "https://github.com/mdfbaam/clipper";
  };
}
