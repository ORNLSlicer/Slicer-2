{
    description = "ORNL Slicer 2 - An advanced object slicer";

    inputs = {
        nixpkgs.url = github:mdfbaam/nixpkgs/pathcompiler-staging;
        utils.url = github:numtide/flake-utils;

        kubazip-src = {
            url = github:kuba--/zip;
            flake = false;
        };
        nlohmann-fifomap-src = {
            url = github:nlohmann/fifo_map;
            flake = false;
        };
        psimpl-src = {
            url = github:skyrpex/psimpl;
            flake = false;
        };
        qtxlsxwriter-src = {
            url = github:VSRonin/QtXlsxWriter;
            flake = false;
        };
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
                    version = "1.0-" + (builtins.substring 0 8 (if (self ? rev) then self.rev else "dirty"));

                    src = self;

                    buildInputs = [
                        pkgs.cmake
                        pkgs.pkg-config
                        pkgs.ninja
                        pkgs.libGL
                        pkgs.qt5.qtbase
                        pkgs.qt5.qtcharts

                        pkgs.assimp
                        pkgs.boost184
                        pkgs.cgal_5
                        pkgs.eigen
                        pkgs.gmp
                        pkgs.nlohmann_json
                        pkgs.mpfr

                        libs.kubazip
                        libs.nlohmann_fifomap
                        libs.s2clipper
                        libs.ornltcp-sockets
                        libs.psimpl
                        libs.qtxlsxwriter
                    ];

                    nativeBuildInputs = [
                        pkgsNative.breakpointHook
                        pkgsNative.qt5.wrapQtAppsHook
                    ];

                    # Workaround for non-NixOS to find GPU drivers
                    # TODO: add more linux distro paths as they are found or query ldconfig
                    qtWrapperArgs = [
                        "--suffix LD_LIBRARY_PATH : /usr/lib/x86_64-linux-gnu"
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

            libs = {
                kubazip          = import contrib/nix/kubazip.nix      { inherit pkgs; source = attrs.kubazip-src; };
                nlohmann_fifomap = import contrib/nix/fifomap.nix      { inherit pkgs; source = attrs.nlohmann-fifomap-src; };
                ornltcp-sockets  = import contrib/nix/tcpsockets.nix   { inherit pkgs; source = "${contrib/ORNL-TCP-Sockets}"; };
                psimpl           = import contrib/nix/psimpl.nix       { inherit pkgs; source = attrs.psimpl-src; };
                qtxlsxwriter     = import contrib/nix/qtxlsxwriter.nix { inherit pkgs; source = attrs.qtxlsxwriter-src; };
                # Clipper lib is a bit different since we use a version not available anywhere else.
                # So we have to use the one from the slicer 2 repo itself.
                s2clipper        = import contrib/nix/s2clipper.nix    { inherit pkgs; source = "${contrib/clipper}"; };
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
                    pkgsNative.qt6.wrapQtAppsHook
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
                        export NIX_CMAKE_PRESET_FILE="$FLAKE_ROOT/cmake/presets/nix.json"
                        jq --indent 4 '.configurePresets[0].cmakeExecutable                   = "${pkgsNative.cmake}/bin/cmake"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_C_COMPILER   = "${llvm.clang}/bin/clang"'       "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_CXX_COMPILER = "${llvm.clang}/bin/clang++"'     "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].cacheVariables.CMAKE_MAKE_PROGRAM = "${pkgsNative.ninja}/bin/ninja"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"

                        jq --indent 4 '.configurePresets[0].vendor."qt.io/QtCreator/1.0".debugger.Binary  = "${llvm.lldb}/bin/lldb"' "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"
                        jq --indent 4 '.configurePresets[0].vendor."qt.io/QtCreator/1.0".debugger.Version = "${llvm.lldb.version}"'  "$NIX_CMAKE_PRESET_FILE" | sponge "$NIX_CMAKE_PRESET_FILE"

                        echo "Updated cmake preset."
                    }

                    # Bit of a weird one - Qt programs can't find plugin path unwrapped, so we make a temporary wrapper and source its env.
                    # See: https://discourse.nixos.org/t/python-qt-woes/11808/10
                    setQtEnvironment=$(mktemp)
                    random=$(openssl rand -base64 20 | sed "s/[^a-zA-Z0-9]//g")
                    makeWrapper "$(type -p sh)" "$setQtEnvironment" "''${qtWrapperArgs[@]}" --argv0 "$random"
                    sed "/$random/d" -i "$setQtEnvironment"
                    source "$setQtEnvironment"

                    # Make sure that git hooks are enabled.
                    git config core.hooksPath "$FLAKE_ROOT/.gitsetup/hooks"

                    # Allow software to find OpenGL drivers.
                    export LD_FALLBACK_PATH=/usr/lib/x86_64-linux-gnu

                    # Lastly, let the XDG_CONFIG_HOME be set to the flake root.
                    export XDG_CONFIG_HOME="$FLAKE_ROOT/.xdg_config"
                '';
            };
        };
    });
}
