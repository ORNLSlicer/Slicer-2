{
    description = "ORNL Slicer 2 - An advanced object slicer";

    inputs = {
        nixpkgs.url = github:NixOS/nixpkgs/nixos-22.11;

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

    outputs = attrs @ { self, nixpkgs, ... }: rec {
        system = "x86_64-linux";
        pkgs = import nixpkgs {
            inherit system;
        };

        packages.${system} = rec {
            default = ornl-slicer2;

            # Main Slicer 2 derivation.
            ornl-slicer2 = pkgs.stdenv.mkDerivation rec {
                pname = "ornl-slicer2";
                version = "0.99.6-" + (builtins.substring 0 8 (if (self ? rev) then self.rev else "dirty"));

                src = self;

                buildInputs = [
                    pkgs.cmake
                    pkgs.pkg-config
                    pkgs.ninja
                    pkgs.libGL
                    pkgs.qt5.qtbase
                    pkgs.qt5.qtcharts

                    pkgs.assimp
                    pkgs.boost172
                    pkgs.cgal_5
                    pkgs.eigen
                    pkgs.gmp
                    pkgs.nlohmann_json
                    pkgs.mpfr

                    kubazip
                    nlohmann_fifomap
                    s2clipper
                    ornltcp-sockets
                    psimpl
                    qtxlsxwriter
                ];

                nativeBuildInputs = [
                    pkgs.breakpointHook
                    pkgs.qt5.wrapQtAppsHook
                ];

                S2_USE_NIX = true;

                # Workaround for non-NixOS to find GPU drivers
                # TODO: add more linux distro paths as they are found or query ldconfig
                qtWrapperArgs = [
                    "--suffix LD_LIBRARY_PATH : /usr/lib/x86_64-linux-gnu"
                ];

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

            # Other packages we make derivations for - i.e. not in nixpkgs already.
            kubazip          = import contrib/nix/kubazip.nix      { inherit pkgs; source = attrs.kubazip-src; };
            nlohmann_fifomap = import contrib/nix/fifomap.nix      { inherit pkgs; source = attrs.nlohmann-fifomap-src; };
            ornltcp-sockets  = import contrib/nix/tcpsockets.nix   { inherit pkgs; source = "${contrib/ORNL-TCP-Sockets}"; };
            psimpl           = import contrib/nix/psimpl.nix       { inherit pkgs; source = attrs.psimpl-src; };
            qtxlsxwriter     = import contrib/nix/qtxlsxwriter.nix { inherit pkgs; source = attrs.qtxlsxwriter-src; };
            # Clipper lib is a bit different since we use a version not available anywhere else.
            # So we have to use the one from the slicer 2 repo itself.
            s2clipper        = import contrib/nix/s2clipper.nix    { inherit pkgs; source = "${contrib/clipper}"; };
        };

        bundlers.${system} = rec {
            default = drv: self.apps.${system}.default;
        };

        apps.${system} = rec {
            default = ornl-slicer2;

            ornl-slicer2 = {
                type = "app";
                program = "${self.packages.${system}.ornl-slicer2}/bin/slicer2";
            };
        };

        devShells.${system} = rec {
            default = ornl-slicer2-dev;

            # Main developer shell.
            ornl-slicer2-dev = pkgs.mkShell rec {
                name = "s2-dev";

                packages = [
                    pkgs.llvmPackages_14.llvm
                    pkgs.llvmPackages_14.clang
                    pkgs.llvmPackages_14.libclang
                ] ++ self.outputs.packages.${system}.ornl-slicer2.buildInputs;

                inputsFrom = [
                    # NOP
                ];

                shellHook = ''
                    export PS1='\n\[\033[1;36m\][${name}:\W]\$\[\033[0m\] '
                    export S2_USE_NIX=true
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"
                    # Bit of a weird one - Qt programs can't find plugin path unwrapped, so we define it here instead for dev.
                    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
                '';
            };
        };
    };
}
