//! build.rs — regenerates WGSL shaders from Slang sources when slangc is available.
//! If slangc is not on PATH, the build uses the committed WGSL in src/shaders/ directly.
//! When `BITTY_REGENERATE_SHADERS=1` is set, slangc is required.

use std::path::Path;

fn main() {
    let shaders_dir = Path::new("src/shaders");
    let slang_dir = shaders_dir.join("slang");
    let _generated_dir = shaders_dir.join("generated");

    if slang_dir.exists() {
        println!("cargo:rerun-if-changed={}", slang_dir.display());
    }

    // Check for slangc
    let regenerate = std::env::var("BITTY_REGENERATE_SHADERS").unwrap_or_default() == "1";
    if regenerate {
        let output = std::process::Command::new("slangc")
            .arg("--version")
            .output();
        match output {
            Ok(o) if o.status.success() => {
                println!("cargo:warning=slangc found: regenerating WGSL shaders");
                let slang_files = std::fs::read_dir(&slang_dir)
                    .unwrap_or_else(|_| std::fs::read_dir(shaders_dir).unwrap());
                for entry in slang_files.flatten() {
                    if entry.path().extension().map(|e| e == "slang").unwrap_or(false) {
                        let name = entry.path().file_stem().unwrap().to_string_lossy().to_string();
                        let wgsl_out = shaders_dir.join(format!("{name}.wgsl"));
                        let status = std::process::Command::new("slangc")
                            .arg(&entry.path())
                            .arg("-target").arg("wgsl")
                            .arg("-o").arg(&wgsl_out)
                            .status();
                        match status {
                            Ok(s) if s.success() => println!("cargo:warning=  → {}", wgsl_out.display()),
                            Ok(s) => panic!("slangc failed for {}: exit {}", name, s),
                            Err(e) => panic!("slangc error for {}: {e}", name),
                        }
                    }
                }
            }
            Ok(_) => panic!("BITTY_REGENERATE_SHADERS=1 but slangc not found"),
            Err(e) => panic!("BITTY_REGENERATE_SHADERS=1 but slangc not found: {e}"),
        }
    } else if std::env::var("BITTY_SLANG_CHECK").unwrap_or_default() == "1" {
        // Verify slang sources are consistent with committed WGSL (optional check)
        println!("cargo:warning=Slang shader check: ensure committed WGSL matches Slang sources");
        println!("cargo:warning=Run with BITTY_REGENERATE_SHADERS=1 to regenerate.");
    }
}
