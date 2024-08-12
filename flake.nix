{
    description = "ORNL Slicer 2 - An advanced object slicer";

    inputs = {
        nixpkgs.url = github:mdfbaam/nixpkgs/slicer2;
        utils.url = github:numtide/flake-utils;
    };

    outputs = attrs @ { self, ... }: attrs.utils.lib.eachDefaultSystem (system: rec {
        config = rec {
            pkgsNative = import attrs.nixpkgs {
                inherit system;
                config = {
                    glibc.withLdFallbackPatch = true;
                };
            };

            pkgsMinGW64 = (import attrs.nixpkgs { inherit system; }).pkgsCross.mingwW64;

            llvm = pkgsNative.llvmPackages_18;
        };

        derivations = { pkgs, stdenv ? pkgs.stdenv }: with config; rec {
            ornl = {
                slicer2 = stdenv.mkDerivation rec {
                    pname = "ornl-slicer2";
                    version = "1.01-" + (builtins.substring 0 8 (if (self ? rev) then self.rev else "dirty"));

                    src = self;

                    buildInputs = [
                        pkgs.qt5.qtbase
                        pkgs.qt5.qtcharts

                        pkgs.assimp
                        pkgs.boost184
                        pkgs.cgal_5
                        pkgs.eigen
                        pkgs.gmp
                        pkgs.nlohmann_json
                        pkgs.mpfr
                        pkgs.hdf5
                        pkgs.vtk-qt5
                        pkgs.tbb

                        llvm.openmp
                    ];

                    nativeBuildInputs = [
                        pkgsNative.cmake
                        pkgsNative.ninja
                        pkgsNative.pkg-config
                        pkgsNative.breakpointHook
                        pkgsNative.qt5.wrapQtAppsHook
                    ];

                    # Workaround for non-NixOS to find GPU drivers
                    # TODO: add more linux distro paths as they are found or query ldconfig
                    qtWrapperArgs = pkgs.lib.optionals stdenv.isLinux [
                        "--suffix LD_FALLBACK_PATH : /usr/lib/x86_64-linux-gnu"
                    ];

                    # The current build system is a bit weird, so we have to do some manual copying.
                    installPhase = ''
                        runHook preInstall

                        mkdir -p $out/bin
                        cp ornl_slicer_2 $out/bin/slicer2
                        cp -r templates $out/bin
                        cp -r layerbartemplates $out/bin


                        mkdir -p $out/share/doc
                        cp Slicer_2_User_Guide.pdf $out/share/doc

                        runHook postInstall
                    '';

                    meta.mainProgram = "slicer2";
                };
            };

            ide = {
                qtcreator = pkgs.qtcreator.overrideAttrs (final: prev: {
                    qtWrapperArgs = prev.qtWrapperArgs ++ pkgs.lib.optionals stdenv.isLinux [
                        "--suffix LD_FALLBACK_PATH : /usr/lib/x86_64-linux-gnu"
                    ];

                    cmakeFlags = prev.cmakeFlags ++ [
                        "-DBUILD_PLUGIN_CLANGFORMAT=OFF"
                    ];
                });
            };
        };

        packages = with config; rec {
            default = ornl.slicer2;

            inherit ( derivations { pkgs = pkgsNative; stdenv = llvm.stdenv; } ) ornl libs ide;
            windows = derivations { pkgs = pkgsMinGW64; };
        };

        bundlers.${system} = rec {
            default = drv: self.apps.${system}.default;
        };

        apps.${system} = rec {
            default = ornl-slicer2;

            ornl-slicer2 = {
                type = "app";
                program = "${self.packages.${system}.ornl.slicer2}/bin/slicer2";
            };
        };

        devShells = with config; rec {
            default = s2-dev;

            # Main developer shell.
            s2-dev = pkgsNative.mkShell.override { stdenv = llvm.stdenv; } rec {
                name = "s2-dev";

                packages = [
                    pkgsNative.cntr

                    pkgsNative.ccache
                    pkgsNative.git
                    pkgsNative.python3
                    pkgsNative.jq
                    pkgsNative.moreutils

                    pkgsNative.doxygen
                    pkgsNative.graphviz

                    pkgsNative.clazy
                    llvm.lldb
                    llvm.clang-tools
                ] ++ self.outputs.packages.${system}.ornl.slicer2.buildInputs
                  ++ self.outputs.packages.${system}.ornl.slicer2.nativeBuildInputs;

                nativeBuildInputs = [
                    pkgsNative.qt5.wrapQtAppsHook
                    pkgsNative.makeWrapper
                    pkgsNative.openssl
                ];

                # For dev, we want to disable hardening.
                hardeningDisable = [
                    "bindnow"
                    "format"
                    "fortify"
                    "fortify3"
                    "pic"
                    "relro"
                    "stackprotector"
                    "strictoverflow"
                ];

                shellHook = ''
                    # Some util variables
                    export FLAKE_ROOT=$(git rev-parse --show-toplevel)

                    # Utility function to update tools in the cmake preset, called manually.
                    function manualUpdatePreset() {
                        local NIX_CMAKE_PRESET_FILE="$FLAKE_ROOT/cmake/presets/nix.json"
                        local QT5_PLUGIN_PATH="${pkgsNative.qt5.qtbase}/${pkgsNative.qt5.qtbase.qtPluginPrefix}:${pkgsNative.qt5.qtdeclarative}/${pkgsNative.qt5.qtbase.qtPluginPrefix}"

                        jq --indent 4 '.configurePresets[0].cmakeExecutable                   = "${pkgsNative.cmake}/bin/cmake"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_C_COMPILER   = "${llvm.clang}/bin/clang"'       "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_CXX_COMPILER = "${llvm.clang}/bin/clang++"'     "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_MAKE_PROGRAM = "${pkgsNative.ninja}/bin/ninja"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].environment.QT_PLUGIN_PATH        = "'"$QT5_PLUGIN_PATH"'"'          "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"

                        jq --indent 4 '.configurePresets[0].vendor."qt.io/QtCreator/1.0".debugger.Binary  = "${llvm.lldb}/bin/lldb"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].vendor."qt.io/QtCreator/1.0".debugger.Version = "${llvm.lldb.version}"'  "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"

                        echo "Updated cmake preset."
                    }

                    # Allow software to find OpenGL drivers.
                    export LD_FALLBACK_PATH=/usr/lib/x86_64-linux-gnu

                    # Lastly, let the XDG_CONFIG_HOME be set to the flake root.
                    export XDG_CONFIG_HOME="$FLAKE_ROOT/.xdg_config"
                '';
            };
        };
    });
}
