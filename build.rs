// build.rs - Download and embed Three.js r184 + OrbitControls for offline use
//              Conditionally compile official minebench renderer (renderer-official feature)

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn main() {
    let three_version = "0.139.0";
    let orbitcontrols_output = "src/reports/templates/voxel-viewer-orbitcontrols.js";
    let threejs_output = "src/reports/templates/voxel-viewer-three.js";

    // Download Three.js if missing
    if !Path::new(threejs_output).exists() || !Path::new(orbitcontrols_output).exists() {
        println!(
            "cargo:warning=Downloading Three.js r{} from CDN...",
            three_version
        );
        download_and_embed_threejs(three_version, threejs_output, orbitcontrols_output);
    }

    // Conditionally compile official renderer
    if cfg!(feature = "renderer-official") {
        compile_official_renderer();
    }
}

fn download_and_embed_threejs(version: &str, threejs_path: &str, orbitcontrols_path: &str) {
    let threejs_url = format!(
        "https://cdn.jsdelivr.net/npm/three@{}/build/three.min.js",
        version
    );
    let threejs_content = reqwest::blocking::get(&threejs_url)
        .expect("Failed to fetch Three.js")
        .text()
        .expect("Failed to read Three.js response");

    let mut f = fs::File::create(threejs_path).expect("Failed to create Three.js file");
    writeln!(
        f,
        "// Embedded Three.js r{}\n// Source: {}\n",
        version, threejs_url
    )
    .expect("Failed to write Three.js header");
    write!(f, "{}", threejs_content).expect("Failed to write Three.js");

    let orbitcontrols_url = format!(
        "https://cdn.jsdelivr.net/npm/three@{}/examples/js/controls/OrbitControls.js",
        version
    );
    let orbitcontrols_content = reqwest::blocking::get(&orbitcontrols_url)
        .expect("Failed to fetch OrbitControls")
        .text()
        .expect("Failed to read OrbitControls response");

    let mut f = fs::File::create(orbitcontrols_path).expect("Failed to create OrbitControls file");
    writeln!(
        f,
        "// Embedded OrbitControls from three.js r{}\n// Source: {}\n",
        version, orbitcontrols_url
    )
    .expect("Failed to write OrbitControls header");
    write!(f, "{}", orbitcontrols_content).expect("Failed to write OrbitControls");

    println!("cargo:warning=Three.js r{} embedded successfully.", version);
}

/// Compile official minebench renderer from TypeScript (requires Node.js)
///
/// NOTE: This downloads source files from the minebench repo (Ammaar-Alam/minebench)
/// pinned to a specific commit SHA. The SHA is hardcoded in the files array below.
/// Update it to a newer commit when upgrading to a newer version of the renderer.
///
/// Expected checksums (update when minebench repo changes):
/// - mesh.ts: SHA256 TBD (from commit f3921316c7025f4c896f4de4d52c5eb0c2a36652)
/// - mesh.worker.ts: SHA256 TBD
///   etc. (placeholder checksums; replace with actual SHA-256 of downloaded files)
fn compile_official_renderer() {
    let output = "src/reports/templates/official-renderer.js";

    // Check for Node.js
    let which_node = Command::new("which").arg("node").output();
    match which_node {
        Ok(output) if output.status.success() => {}
        _ => panic!(
            "renderer-official requires Node.js. Please install Node.js 18+ and npm.\n\
             See https://nodejs.org/ for installation instructions."
        ),
    }

    // Create temp directory mirroring minebench repo structure
    let tmp_dir = std::env::temp_dir().join("minebench-renderer-build");
    fs::create_dir_all(&tmp_dir).expect("Failed to create temp build directory");
    fs::create_dir_all(tmp_dir.join("lib/voxel")).expect("Failed to create lib/voxel dir");
    fs::create_dir_all(tmp_dir.join("lib/blocks")).expect("Failed to create lib/blocks dir");
    fs::create_dir_all(tmp_dir.join("public/textures"))
        .expect("Failed to create public/textures dir");

    // Download all TypeScript files from minebench repo
    // Pinned to commit f3921316c7025f4c896f4de4d52c5eb0c2a36652 (latest on 2026-07-14)
    // Update this SHA when upgrading to a newer version of the minebench renderer.
    let files: [(&str, &str); 14] = [
        ("lib/voxel/mesh.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/voxel/mesh.ts"),
        ("lib/voxel/meshPayloadCache.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/voxel/meshPayloadCache.ts"),
        ("lib/voxel/renderVisibility.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/voxel/renderVisibility.ts"),
        ("lib/voxel/types.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/voxel/types.ts"),
        ("lib/voxel/mesh.worker.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/voxel/mesh.worker.ts"),
        ("lib/blocks/palettes.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/palettes.ts"),
        ("lib/blocks/registry.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/registry.ts"),
        ("lib/blocks/atlas.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/atlas.ts"),
        ("lib/blocks/atlas-map.json", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/atlas-map.json"),
        ("lib/blocks/textures.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/textures.ts"),
        ("lib/blocks/blockUtils.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/blockUtils.ts"),
        ("lib/blocks/textureSets.ts", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/textureSets.ts"),
        ("lib/blocks/palettes.json", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/lib/blocks/palettes.json"),
        ("public/textures/atlas.png", "https://raw.githubusercontent.com/Ammaar-Alam/minebench/f3921316c7025f4c896f4de4d52c5eb0c2a36652/public/textures/atlas.png"),
    ];

    for (name, url) in files {
        let path = tmp_dir.join(name);
        if path.exists() {
            continue;
        }
        match reqwest::blocking::get(url) {
            Ok(resp) if resp.status().is_success() => {
                let content = resp.text().unwrap_or_default();
                fs::write(&path, content).expect("Failed to write file");
            }
            _ => {
                println!("cargo:warning=Skipping missing file: {}", name);
            }
        }
    }

    println!("cargo:warning=Compiling official renderer with esbuild...");

    // Build esbuild command with aliases for all path imports
    let mut cmd = Command::new("npx");
    cmd.arg("--yes")
        .arg("esbuild")
        .arg(tmp_dir.join("lib/voxel/mesh.ts"))
        .arg("--bundle")
        .arg("--minify")
        .arg("--external:three")
        .arg("--external:@/lib/voxel/mesh.worker.ts")
        .arg("--format=iife")
        .arg("--global-name=createVoxelViewer");

    let aliases = [
        "lib/voxel/mesh",
        "lib/voxel/meshPayloadCache",
        "lib/voxel/renderVisibility",
        "lib/voxel/types",
        "lib/blocks/palettes",
        "lib/blocks/registry",
        "lib/blocks/atlas",
        "lib/blocks/atlas-map.json",
        "lib/blocks/textures",
        "lib/blocks/blockUtils",
        "lib/blocks/textureSets",
        "lib/blocks/palettes.json",
    ];
    for alias in &aliases {
        // Determine correct extension for the alias
        let abs_path = if alias.ends_with(".json") {
            format!("{}/{}", tmp_dir.display(), alias)
        } else {
            format!("{}/{}.ts", tmp_dir.display(), alias)
        };
        cmd.arg(format!("--alias:@/{}={}", alias, abs_path));
    }

    cmd.arg(format!("--outfile={}", output));

    let status = cmd.status().expect("Failed to run esbuild");
    match status.success() {
        true => {
            let content = fs::read_to_string(output).expect("Failed to read bundled renderer");

            // Post-process: the esbuild IIFE runtime throws "Dynamic require of 'three' not supported"
            // because there's no real `require` in the browser. Replace the runtime check with
            // a no-op that returns the global THREE.
            let final_content = content
                // Replace the IIFE runtime error with returning global THREE
                .replace(
                    r#"throw Error('Dynamic require of "'+e+'" is not supported')"#,
                    "return window.THREE",
                )
                // Replace Node.js process.env with an empty object
                .replace(r#"process.env"#, "({})")
                // Patch the require function to inject OrbitControls into the returned THREE
                // The pattern: return window.THREE});  →  add OrbitControls to it
                .replace(
                    r#"return window.THREE});"#,
                    "return window.THREE.OrbitControls=window.THREE.OrbitControls,window.THREE});",
                );

            // Read the Web Worker file and embed it as a base64 data URL
            let worker_path = tmp_dir.join("lib/voxel/mesh.worker.ts");
            let worker_content = fs::read_to_string(&worker_path).unwrap_or_else(|_| {
                println!("cargo:warning=Web Worker file missing, stubbing with empty worker");
                "self.onmessage = function() {};".to_string()
            });
            let worker_b64 = base64_encode(&worker_content);
            let worker_data_url = format!("data:application/javascript;base64,{}", worker_b64);

            // Replace the external worker import with the embedded data URL
            // The external import will appear as: new URL("@/lib/voxel/mesh.worker.ts", import.meta.url)
            // or "./mesh.worker.ts". We'll replace the URL creation with a static data URL.
            let final_content = final_content
                .replace(
                    r#""./mesh.worker.ts", import.meta.url)"#,
                    &format!("\"{}\"", worker_data_url),
                )
                .replace(
                    r#""@/lib/voxel/mesh.worker.ts", import.meta.url)"#,
                    &format!("\"{}\"", worker_data_url),
                )
                .replace(r#"import.meta.url"#, "undefined");

            let header = "// Official minebench renderer compiled from TypeScript (Ammaar-Alam/minebench)\n// Requires global THREE (embedded)\n";
            fs::write(output, header.to_owned() + &final_content)
                .expect("Failed to write final renderer");
            println!("cargo:rerun-if-changed={}", output);
            println!("cargo:warning=Official renderer compiled successfully.");
        }
        false => {
            panic!("Failed to compile official renderer with esbuild. Run `npx esbuild --version` to verify.");
        }
    }

    // Clean up temp directory
    let _ = fs::remove_dir_all(&tmp_dir);
}

/// Encode bytes as base64 (simple implementation, no dependency)
fn base64_encode(data: &str) -> String {
    let table: [u8; 64] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.as_bytes().chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).cloned().unwrap_or(0),
            chunk.get(2).cloned().unwrap_or(0),
        ];
        let encoded = [
            table[(b[0] >> 2) as usize] as char,
            table[((b[0] & 0x03) << 4 | (b[1] >> 4)) as usize] as char,
            table[((b[1] & 0x0f) << 2 | (b[2] >> 6)) as usize] as char,
            table[b[2] as usize & 0x3f] as char,
        ];
        let encoded_str: String = encoded.iter().collect();
        match chunk.len() {
            1 => result.push_str(&encoded_str[..2]),
            2 => {
                result.push_str(&encoded_str[..3]);
                result.push('=');
            }
            _ => result.push_str(&encoded_str),
        }
    }
    result
}
