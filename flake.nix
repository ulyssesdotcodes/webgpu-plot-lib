{
  description = "typescript build";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let pkgs = import nixpkgs {inherit system;};
      in {
        devShells = with pkgs; {
          default = mkShell {
          buildInputs = with pkgs; [
            nodejs
          ];

          npmDeps = importNpmLock.buildNodeModules {
            npmRoot = ./.;
            inherit nodejs;
          };

          shellHook = ''
            npx tsc --watch &
            npx http-server  -c-1 public
          '';
        };
      };
    });
}