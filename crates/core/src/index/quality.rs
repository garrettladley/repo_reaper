use std::{
    collections::HashMap,
    path::{Component, Path},
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StaticQualitySignals {
    pub file_depth: usize,
    pub file_size_bytes: u64,
    pub generated: bool,
    pub vendor: bool,
    pub test: bool,
    pub entry_point: bool,
    pub readme: bool,
    pub config_or_manifest: bool,
    pub public_entry_point: bool,
    pub reference_count: usize,
}

impl StaticQualitySignals {
    pub fn analyze(path: &Path, content: &str, file_size_bytes: u64) -> Self {
        let normalized_path = path
            .to_string_lossy()
            .replace('\\', "/")
            .to_ascii_lowercase();
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let file_depth = path
            .components()
            .filter(|component| matches!(component, Component::Normal(_)))
            .count()
            .saturating_sub(1);
        let generated = is_generated(&normalized_path, content);
        let vendor = is_vendor_path(&normalized_path);
        let test = is_test_path(&normalized_path, &file_name);
        let entry_point = is_entry_point(&normalized_path, &file_name);
        let readme = file_name == "readme.md" || file_name == "readme";
        let config_or_manifest = is_config_or_manifest(&file_name);
        let public_entry_point = is_public_entry_point(content);
        let reference_count = approximate_reference_count(content);

        Self {
            file_depth,
            file_size_bytes,
            generated,
            vendor,
            test,
            entry_point,
            readme,
            config_or_manifest,
            public_entry_point,
            reference_count,
        }
    }

    pub fn prior_score(&self) -> f64 {
        let mut score = 0.0;

        if self.generated {
            score -= 0.55;
        }
        if self.vendor {
            score -= 0.45;
        }
        if self.test {
            score -= 0.08;
        }
        if self.entry_point {
            score += 0.28;
        }
        if self.readme {
            score += 0.22;
        }
        if self.config_or_manifest {
            score += 0.20;
        }
        if self.public_entry_point {
            score += 0.12;
        }

        score += match self.file_depth {
            0 | 1 => 0.10,
            2 => 0.04,
            3..=5 => 0.0,
            _ => -0.08,
        };

        if self.file_size_bytes > 750_000 {
            score -= 0.18;
        } else if self.file_size_bytes > 250_000 {
            score -= 0.08;
        }

        score + (self.reference_count.min(12) as f64 * 0.015)
    }

    pub fn features(&self) -> HashMap<&'static str, f64> {
        HashMap::from([
            ("file_depth", self.file_depth as f64),
            ("file_size_bytes", self.file_size_bytes as f64),
            ("generated", bool_feature(self.generated)),
            ("vendor", bool_feature(self.vendor)),
            ("test", bool_feature(self.test)),
            ("entry_point", bool_feature(self.entry_point)),
            ("readme", bool_feature(self.readme)),
            ("config_or_manifest", bool_feature(self.config_or_manifest)),
            ("public_entry_point", bool_feature(self.public_entry_point)),
            ("reference_count", self.reference_count as f64),
            ("static_prior_score", self.prior_score()),
        ])
    }
}

fn bool_feature(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn is_generated(normalized_path: &str, content: &str) -> bool {
    normalized_path.contains("/generated/")
        || normalized_path.contains("/gen/")
        || normalized_path.ends_with(".generated.rs")
        || normalized_path.ends_with(".pb.go")
        || normalized_path.ends_with(".pb.rs")
        || content
            .lines()
            .take(8)
            .any(|line| line.to_ascii_lowercase().contains("generated"))
}

fn is_vendor_path(normalized_path: &str) -> bool {
    normalized_path.contains("/vendor/")
        || normalized_path.starts_with("vendor/")
        || normalized_path.contains("/node_modules/")
        || normalized_path.starts_with("node_modules/")
        || normalized_path.contains("/third_party/")
        || normalized_path.starts_with("third_party/")
        || normalized_path.contains("/target/")
        || normalized_path.starts_with("target/")
        || normalized_path.contains("/dist/")
        || normalized_path.starts_with("dist/")
        || normalized_path.contains("/build/")
        || normalized_path.starts_with("build/")
}

fn is_test_path(normalized_path: &str, file_name: &str) -> bool {
    normalized_path.contains("/tests/")
        || normalized_path.contains("/test/")
        || file_name.starts_with("test_")
        || file_name.ends_with("_test.rs")
        || file_name.ends_with("_test.go")
        || file_name.ends_with(".test.ts")
        || file_name.ends_with(".spec.ts")
}

fn is_entry_point(normalized_path: &str, file_name: &str) -> bool {
    matches!(
        file_name,
        "main.rs" | "lib.rs" | "mod.rs" | "main.py" | "index.ts" | "index.js" | "main.go"
    ) || normalized_path == "src/main.rs"
        || normalized_path == "src/lib.rs"
}

fn is_config_or_manifest(file_name: &str) -> bool {
    matches!(
        file_name,
        "cargo.toml"
            | "pyproject.toml"
            | "package.json"
            | "tsconfig.json"
            | "dockerfile"
            | "makefile"
            | "justfile"
            | "readme.md"
    ) || file_name.ends_with(".yaml")
        || file_name.ends_with(".yml")
        || file_name.ends_with(".toml")
        || file_name.ends_with(".json")
}

fn is_public_entry_point(content: &str) -> bool {
    content.lines().take(80).any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("pub fn main")
            || trimmed.starts_with("fn main")
            || trimmed.starts_with("pub mod ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("export ")
            || trimmed.starts_with("def main")
    })
}

fn approximate_reference_count(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("use ")
                || trimmed.starts_with("pub use ")
                || trimmed.starts_with("pub mod ")
                || trimmed.starts_with("mod ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("#include ")
                || trimmed.starts_with("require(")
        })
        .count()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::StaticQualitySignals;

    #[test]
    fn detects_generated_vendor_and_test_markers() {
        let signals = StaticQualitySignals::analyze(
            Path::new("vendor/generated/client_test.rs"),
            "// generated by tool\nfn test_case() {}",
            120,
        );

        assert!(signals.generated);
        assert!(signals.vendor);
        assert!(signals.test);
        assert!(signals.prior_score() < 0.0);
    }

    #[test]
    fn boosts_entry_points_manifests_and_reference_like_lines() {
        let signals = StaticQualitySignals::analyze(
            Path::new("src/lib.rs"),
            "pub mod search;\nuse crate::search::Index;\npub struct Engine;",
            90,
        );

        assert!(signals.entry_point);
        assert!(signals.public_entry_point);
        assert_eq!(signals.reference_count, 2);
        assert!(signals.prior_score() > 0.0);
    }
}
