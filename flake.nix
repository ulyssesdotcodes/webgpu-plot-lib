{
  description = "typescript + rust/wgpu webgpu chart";

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
            rustup
            wasm-pack
          ];

          npmDeps = importNpmLock.buildNodeModules {
            npmRoot = ./.;
            inherit nodejs;
          };

          shellHook = ''
            rustup target add wasm32-unknown-unknown
            npx tsc --watch &
            npx http-server  -c-1 public
          '';
        };
      };
    });
}
