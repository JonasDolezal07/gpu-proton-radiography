//! Build script - compiles GLSL shaders to SPIR-V

use std::process::Command;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("../shaders");
    let out_dir = shader_dir;

    // Rerun if shaders change
    println!("cargo:rerun-if-changed=../shaders/boris.comp");

    // Find glslangValidator
    let glslang = find_glslang().expect("glslangValidator not found. Install Vulkan SDK.");

    // Compile boris.comp
    let input = shader_dir.join("boris.comp");
    let output = out_dir.join("boris.spv");

    let status = Command::new(&glslang)
        .args(["-V", input.to_str().unwrap(), "-o", output.to_str().unwrap()])
        .status()
        .expect("Failed to run glslangValidator");

    if !status.success() {
        panic!("Shader compilation failed");
    }

    println!("cargo:warning=Compiled boris.comp -> boris.spv");
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
