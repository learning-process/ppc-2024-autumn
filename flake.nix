{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    treefmt-nix,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      py3 = pkgs.python3.withPackages (ps: [ps.xlsxwriter]);
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          ninja
          cmake
          openmpi
          py3
          cppcheck
        ];
      };
      formatter =
        (treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.alejandra.enable = true;
        })
        .config
        .build
        .wrapper;
    });
}
