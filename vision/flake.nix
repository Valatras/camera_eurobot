{
  description = "Eurobot 2026 Vision — ArUco/QR detection pipeline";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs =
    { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      opencv-gtk3 = pkgs.python312Packages.opencv4.override { enableGtk3 = true; };
      python = pkgs.python312.withPackages (ps: [
        opencv-gtk3
        ps.numpy
        ps.pyserial
        ps.requests
        ps.python-socketio
        ps.websocket-client # python-socketio[client] extra
        ps.pytest
      ]);
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = [
          python
          pkgs.scrcpy
          pkgs.android-tools # adb
        ];

        shellHook = ''
          echo ""
          echo "✅ Vision dev shell ready"
          echo "   python : $(python --version)"
          echo "   scrcpy : $(scrcpy --version 2>&1 | head -1)"
          echo ""
          echo "   Lancer la caméra  : scrcpy --video-source=camera --camera-size=3840x2160 --camera-facing=back --v4l2-sink=/dev/video2 --no-playback"
          echo "   Lancer la vision  : python detect_markers.py"
          echo ""
        '';
      };

      # Lightweight camera relay for remote setups (capture → MJPEG, no detection)
      # Usage: nix run .#camera-relay -- --device 2 --port 8082
      packages.x86_64-linux.camera-relay =
        let
          opencv-headless = pkgs.python312Packages.opencv4;
          py = pkgs.python312.withPackages (_: [
            opencv-headless
            pkgs.python312Packages.numpy
          ]);
        in
        pkgs.writeShellApplication {
          name = "camera-relay";
          runtimeInputs = [ py ];
          text = ''python ${./camera_relay.py} "$@"'';
        };
    };
}
