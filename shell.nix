let
  sources = import ./tools/nix/npins;
  fenix = import sources.fenix { inherit pkgs; };
  pkgs = import sources.nixpkgs { };
  poetry2nix = import sources.poetry2nix { inherit pkgs; };

  versions = builtins.fromJSON (builtins.readFile ./tools/nix/versions.json);

  rust-combined =
    let
      stable = fenix.toolchainOf {
        channel = versions.stable.version;
        sha256 = versions.stable.sha256;
      };
      nightly = fenix.toolchainOf {
        channel = "nightly";
        date = versions.nightly.date;
        sha256 = versions.nightly.sha256;
      };
    in
    fenix.combine [
      nightly.rustfmt # should be the first
      stable.rust
      stable.rust-analyzer
      stable.rust-src
    ];

  # A workaround to allow running `cargo +nightly fmt`
  cargo-wrapper = pkgs.writeScriptBin "cargo" ''
    #!${pkgs.stdenv.shell}
    [ "$1" != "+nightly" ] || [ "$2" != "fmt" ] || shift
    exec ${rust-combined}/bin/cargo "$@"
  '';

  # Python dependencies used in tests
  python-env = poetry2nix.mkPoetryEnv {
    projectDir = ./tests; # reads pyproject.toml and poetry.lock
    preferWheels = true; # wheels speed up building of the environment
  };

in
pkgs.mkShell {
  buildInputs = [
    # Rust toolchain
    cargo-wrapper # should be before rust-combined
    rust-combined

    # Crates' build dependencies
    pkgs.libunwind # for unwind-sys
    pkgs.pkg-config # for unwind-sys and other deps
    pkgs.protobuf # for prost-wkt-types
    pkgs.rustPlatform.bindgenHook # for bindgen deps

    # For tests and tools
    pkgs.cargo-nextest # mentioned in .github/workflows/rust.yml
    pkgs.curl # used in ./tests
    pkgs.gnuplot # optional runtime dep for criterion
    pkgs.jq # used in ./tests and ./tools
    pkgs.nixfmt-rfc-style # to format this file
    pkgs.npins # used in tools/nix/update.py
    pkgs.iconv
    pkgs.ccache # compile cache for c/cxx code
    pkgs.sccache # compile cache for rust code
    pkgs.mold # mold linker
    pkgs.poetry # used to update poetry.lock
    pkgs.wget # used in tests/storage-compat
    pkgs.yq-go # used in tools/generate_openapi_models.sh
    pkgs.ytt # used in tools/generate_openapi_models.sh
    python-env # used in tests
  ];

  shellHook = ''
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # Override minimally supported macos version
      export CFLAGS="-mmacosx-version-min=10.13"
      export CXXFLAGS="-mmacosx-version-min=10.13"
      export MACOSX_DEPLOYMENT_TARGET="10.13"
    fi
    # Caching for C/C++ deps, particularly for librocksdb-sys
    export CC="ccache $CC"
    export CXX="ccache $CXX"

    # use mold linker
    export LD="${pkgs.mold}/bin/ld.mold"

    # Caching for Rust
    PATH="${pkgs.sccache}/bin:$PATH"
    export RUSTC_WRAPPER="sccache"

    # Caching for lindera-unidic
    [ "''${LINDERA_CACHE+x}" ] ||
      export LINDERA_CACHE="''${XDG_CACHE_HOME:-$HOME/.cache}/lindera"

    # https://qdrant.tech/documentation/guides/common-errors/#too-many-files-open-os-error-24
    [ "$(ulimit -n)" -ge 10000 ] || ulimit -n 10000
  '';
}
