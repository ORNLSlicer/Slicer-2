{
  runCommand,
  nsis,
  windowsApplication,
  installerScript,
  licenseFile,
  iconFile,
}:

let
  inherit (windowsApplication) version;
  installerFileName = "ornlslicer-${version}-installer.exe";
in
# NSIS preserves source mtimes, so retain Nix's normalized timestamps below.
runCommand "ornlslicer-windows-installer-${version}" {
  pname = "ornlslicer-windows-installer";
  inherit version;

  nativeBuildInputs = [ nsis ];

  passthru = {
    inherit installerFileName windowsApplication;
  };

  meta = windowsApplication.meta // {
    description = "Windows installer for ORNLSlicer";
    platforms = [ "x86_64-linux" ];
  };
} ''
  mkdir -p build/scripts build/resources/icons "$out"

  cp ${installerScript} build/scripts/installer.nsi
  cp ${licenseFile} build/LICENSE.md
  cp ${iconFile} build/resources/icons/ornlslicer_logo.ico
  touch --reference=${installerScript} build/scripts/installer.nsi
  touch --reference=${licenseFile} build/LICENSE.md
  touch --reference=${iconFile} build/resources/icons/ornlslicer_logo.ico
  ln -s ${windowsApplication} build/ornlslicer

  cd build/scripts
  makensis \
    -DOUTFILE="$out/${installerFileName}" \
    -DEXE_SOURCES=ornlslicer \
    -DVERSION="${version}" \
    installer.nsi
''
