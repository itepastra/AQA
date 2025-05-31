{
  description = "Nix Development Flake the QSVM optimiser";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/master";

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312Full;
        pythonPackages = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "your_package";
          nativeBuildInputs = [ pkgs.bashInteractive ];
          buildInputs = with pythonPackages; [
            # tensorflow
            setuptools
            wheel
            venvShellHook
          ];
          venvDir = ".venv";
          src = null;
          postVenv = ''
            unset SOURCE_DATE_EPOCH

          '';
          postShellHook = ''
            unset SOURCE_DATE_EPOCH
            unset LD_PRELOAD

            PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
            # fixes libstdc++ issues and libgl.so issues
            LD_LIBRARY_PATH=${pkgs.libz}/lib/:${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
          '';
        };
      }
    );
}
