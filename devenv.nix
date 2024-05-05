{ pkgs, lib, config, inputs, ... }:

{
  languages.python = {
    version = "3.12.3";
    enable = true;
    venv = {
      enable = true;
      requirements = ''
      python-lsp-server
      numpy
      torch@https://download.pytorch.org/whl/nightly/cu124/torch-2.4.0.dev20240502%2Bcu124-cp312-cp312-linux_x86_64.whl
      '';
    };
  };
  enterShell = ''
    export LD_PRELOAD="/run/opengl-driver/lib/libcuda.so"
  '';
}
