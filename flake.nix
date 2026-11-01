{
  description = "cosi dev shell";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "cosi-shell";
        packages = [
          (pkgs.python3.withPackages (p: with p; [
            flask
            numpy
            nltk
            pytest
          ]))
        ];

        shellHook = ''
          echo "cosi dev shell ready. Run:"
          echo "  python3 offline_ingest.py"
          echo "  python3 cosi_engine.py"
          echo "  python3 app.py"
          echo "  python3 -m pytest tests/"
          echo "  python3 eval/run_eval.py"
        '';
      };
    };
}
