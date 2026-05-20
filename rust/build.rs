//! Build script - compiles GLSL shaders to SPIR-V

use std::process::Command;
use std::path::Path;

fn main() {
    inject_git_info();

    let shader_dir = Path::new("../shaders");

    // Rerun if shaders change
    println!("cargo:rerun-if-changed=../shaders/boris.comp");
    println!("cargo:rerun-if-changed=../shaders/fullscreen.vert");
    println!("cargo:rerun-if-changed=../shaders/detector3d.vert");
    println!("cargo:rerun-if-changed=../shaders/detector.frag");
    println!("cargo:rerun-if-changed=../shaders/volume.frag");
    println!("cargo:rerun-if-changed=../shaders/marker.vert");
    println!("cargo:rerun-if-changed=../shaders/marker.frag");
    println!("cargo:rerun-if-changed=../shaders/egui.vert");
    println!("cargo:rerun-if-changed=../shaders/egui.frag");

    // Find glslangValidator — skip shader recompilation if not present (uses committed .spv files)
    let glslang = match find_glslang() {
        Some(g) => g,
        None => {
            println!("cargo:warning=glslangValidator not found — using pre-compiled .spv files");
            return;
        }
    };

    // Compile all shaders
    let shaders = [
        ("boris.comp", "boris.spv"),
        ("fullscreen.vert", "fullscreen.vert.spv"),
        ("detector3d.vert", "detector3d.vert.spv"),
        ("detector.frag", "detector.frag.spv"),
        ("volume.frag", "volume.frag.spv"),
        ("marker.vert", "marker.vert.spv"),
        ("marker.frag", "marker.frag.spv"),
        ("egui.vert", "egui.vert.spv"),
        ("egui.frag", "egui.frag.spv"),
    ];

    for (src, dst) in shaders {
        let input = shader_dir.join(src);
        let output = shader_dir.join(dst);

        let status = Command::new(&glslang)
            .args(["-V", input.to_str().unwrap(), "-o", output.to_str().unwrap()])
            .status()
            .expect("Failed to run glslangValidator");

        if !status.success() {
            panic!("Shader compilation failed: {}", src);
        }

        println!("cargo:warning=Compiled {} -> {}", src, dst);
    }
}

fn inject_git_info() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");

    if let Some(commit) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
    {
        println!("cargo:rustc-env=GIT_COMMIT={}", commit.trim());
    }

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    println!("cargo:rustc-env=GIT_DIRTY={}", dirty);
}

fn find_glslang() -> Option<String> {
    // Check common locations
    let paths = [
        "/opt/homebrew/bin/glslangValidator",
        "/usr/local/bin/glslangValidator",
        "/usr/bin/glslangValidator",
        "glslangValidator",
    ];

    for path in paths {
        if Command::new(path).arg("--version").output().is_ok() {
            return Some(path.to_string());
        }
    }
    None
}
