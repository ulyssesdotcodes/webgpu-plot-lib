{
  description = "typescript build";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        buildNodeJs = pkgs.callPackage "${<nixpkgs>}/pkgs/development/web/nodejs/nodejs.nix" {
          python = pkgs.python3;
        };

        nodejs = buildNodeJs {
          enableNpm = true;
          version = "24.15.0";
          sha256 = "";
        };
      in rec {
        flakedPkgs = pkgs;

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
          ];
        };
      }
    );
}