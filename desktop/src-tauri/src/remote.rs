//! The engine host on another machine (remote mode).
//!
//! The photographs stay on the Mac; the Linux box renders them. Three things
//! change, and only in remote mode:
//!
//! - **the launch**: `ssh -T … <target> <root>/host/spektralab-host --resources
//!   <root>/host/engine`, so the frames on stdin/stdout are still the host's
//!   own — HOST-PROTOCOL.md is unchanged and knows nothing of this module;
//! - **the paths**: the host reads and writes *its* filesystem, so every file
//!   the user picked is copied up to `<root>/inbox/…` before the request, and
//!   every file the host wrote is copied back down to where the user asked for
//!   it;
//! - **the transfers**: they are announced (`remote_transfer` events), because
//!   a 45 MP copy is not instant and silence would look like a hang.
//!
//! The photographs are copied, never moved: the inbox is a copy, and the only
//! things removed here are the copies this module made itself (the per-export
//! `out/<uuid>` directory). The sidecars stay on the Mac, in the app's own
//! store — nothing about them is remote.
//!
//! **What the far machine gets.** One `spektralab-host` under
//! `<root>/host/`, with its resources in `<root>/host/engine/`. The inbox and
//! the export directories are created on demand.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::process::Command;
use tokio::sync::{Mutex, OnceCell};
use uuid::Uuid;

use crate::files;
use crate::host::{HostError, Launch};

/// The root when nothing says otherwise: the layout the deployment uses.
pub const DEFAULT_ROOT: &str = "/home/jacky/spektralab";

/// Where the host lives when it is not on this machine.
#[derive(Debug, Clone)]
pub struct RemoteConfig {
    /// The ssh target: `user@host`, or an alias from `~/.ssh/config`.
    pub target: String,
    /// The remote root: `<root>/host/spektralab-host`, `<root>/host/engine`.
    pub root: PathBuf,
    /// `--device` for the remote host, exactly as the user wrote it.
    pub device: Option<String>,
}

impl RemoteConfig {
    /// The configuration, from the environment first and the app's own file
    /// second. Without a target there is no remote mode.
    ///
    /// - `SPEKTRALAB_REMOTE` — the ssh target; set it and the app is remote.
    /// - `SPEKTRALAB_REMOTE_ROOT` — where the host is deployed (default
    ///   `/home/jacky/spektralab`).
    /// - `SPEKTRALAB_HOST_DEVICE` — the device index, passed to the far host.
    ///
    /// The same three fields, lower-cased, in `<app config dir>/remote.json`:
    /// `remote`, `remote_root`, `host_device`. The environment wins.
    pub fn load<R: Runtime>(app: &AppHandle<R>) -> Option<Self> {
        let target = env("SPEKTRALAB_REMOTE").or_else(|| file_field(app, "remote"))?;
        let root = env("SPEKTRALAB_REMOTE_ROOT")
            .or_else(|| file_field(app, "remote_root"))
            .unwrap_or_else(|| DEFAULT_ROOT.to_string());
        let device = env("SPEKTRALAB_HOST_DEVICE").or_else(|| file_field(app, "host_device"));
        Some(Self { target, root: PathBuf::from(root), device })
    }

    /// The host's own path on the far machine.
    pub fn host_path(&self) -> PathBuf {
        self.root.join("host").join("spektralab-host")
    }

    /// The engine's resources on the far machine.
    pub fn engine_path(&self) -> PathBuf {
        self.root.join("host").join("engine")
    }

    /// Where the copies of the user's photographs live.
    pub fn inbox_path(&self) -> PathBuf {
        self.root.join("inbox")
    }

    /// Where the exports are written before they are copied back.
    pub fn out_path(&self) -> PathBuf {
        self.root.join("out")
    }

    /// The ssh command line that starts the host on the far machine. The
    /// frames on stdin/stdout are the host's own, unchanged.
    pub fn launch(&self) -> Launch {
        let mut remote = vec![
            quote(&self.host_path().to_string_lossy()),
            "--resources".to_string(),
            quote(&self.engine_path().to_string_lossy()),
        ];
        if let Some(device) = &self.device {
            remote.push("--device".to_string());
            remote.push(quote(device));
        }
        Launch {
            program: PathBuf::from("ssh"),
            args: vec![
                "-T".to_string(),
                "-o".to_string(),
                "BatchMode=yes".to_string(),
                "-o".to_string(),
                "ServerAliveInterval=15".to_string(),
                "-o".to_string(),
                "ServerAliveCountMax=4".to_string(),
                self.target.clone(),
                remote.join(" "),
            ],
            mock: false,
            origin: format!("ssh {} ({})", self.target, self.root.display()),
        }
    }

    /// The ssh options the launch uses, as rsync's `-e` argument.
    fn ssh_options(&self) -> String {
        "ssh -T -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4".to_string()
    }
}

/// One field of the environment, trimmed and not empty.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// One field of `<app config dir>/remote.json` — the environment variable's
/// name, lower-cased.
fn file_field<R: Runtime>(app: &AppHandle<R>, key: &str) -> Option<String> {
    let path = app.path().app_config_dir().ok()?.join("remote.json");
    let bytes = std::fs::read(path).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    value.get(key).and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// One shell word, for the command the far machine runs.
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// rsync hands the remote path to the far machine's shell, and macOS's rsync
/// is 2.6 (no `--protect-args`): spaces have to be escaped by hand.
fn escape(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\").replace(' ', "\\ ")
}

/// The file's name on the far machine: the local name with every character
/// outside `[A-Za-z0-9._-]` replaced by `_`. The remote half of an rsync spec
/// goes through the far machine's shell, so a name with `(`, `&` or a quote
/// would break it; the folder already makes the name unique and the host only
/// needs the extension.
fn file_name(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let safe: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect();
    if safe.trim_matches('.').is_empty() { "file".to_string() } else { safe }
}

/// The file's size and modification time — the cache key of a copy.
fn stat(path: &Path) -> Result<(u64, u64), String> {
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mtime_ms = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    Ok((md.len(), mtime_ms))
}

/// `sha1(absolute path)[:16]` — the inbox folder of a file. Sixteen hex
/// characters of the path's own hash: the same photograph always lands in the
/// same folder, and two photographs never share one.
pub fn inbox_key(local: &Path) -> String {
    let abs = files::standardized(local);
    let digest = Sha1::digest(abs.to_string_lossy().as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The request's input file: what the host reads.
fn input_field(method: &str) -> Option<&'static str> {
    match method {
        "probe" | "thumbnail" | "open" => Some("path"),
        "write_image" => Some("source_path"),
        _ => None,
    }
}

/// The request's output file: what the host writes.
fn output_field(method: &str) -> Option<&'static str> {
    match method {
        "export_image" | "write_image" | "export_cube" | "export_di" => Some("path"),
        _ => None,
    }
}

/// The far machine's path for an export: `<root>/out/<uuid>/<name>`. One
/// directory per export, so two exports never write over each other and the
/// whole lot can be removed in one go.
pub fn out_path(cfg: &RemoteConfig, local: &Path) -> PathBuf {
    cfg.out_path().join(Uuid::new_v4().to_string()).join(file_name(local))
}

/// Whether a destination may be written. The host refuses an existing file
/// unless `overwrite` says so; so do we, before anything is copied — the
/// answer is the same whether the host is here or over there.
pub fn may_write(local: &Path, overwrite: bool) -> bool {
    overwrite || !local.exists()
}

/// Put a local path back where a remote one was written.
fn rewrite(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(s) => {
            if s == from {
                *s = to.to_string();
            }
        }
        Value::Array(items) => {
            for item in items {
                rewrite(item, from, to);
            }
        }
        Value::Object(map) => {
            for (_, item) in map.iter_mut() {
                rewrite(item, from, to);
            }
        }
        _ => {}
    }
}

/// What one request needs on the far machine, and what its answer must say
/// back here.
#[derive(Debug, Default, Clone)]
pub struct Plan {
    /// Remote path → the local path it stands for.
    pub map: Vec<(String, String)>,
    /// The files the host wrote, to be copied back.
    pub downloads: Vec<(PathBuf, PathBuf)>,
}

/// One file already copied up, or being copied.
struct Upload {
    size: u64,
    mtime_ms: u64,
    /// Where it lives on the far machine.
    remote: PathBuf,
    /// The copy itself — one for everybody who asks at the same time.
    cell: OnceCell<Result<String, String>>,
}

/// The far machine: the paths, the copies and the transfers.
pub struct Remote<R: Runtime> {
    pub cfg: RemoteConfig,
    app: AppHandle<R>,
    /// The files already copied up, by their inbox key.
    uploads: Mutex<HashMap<String, Arc<Upload>>>,
}

impl<R: Runtime> Remote<R> {
    pub fn new(cfg: RemoteConfig, app: AppHandle<R>) -> Arc<Self> {
        Arc::new(Self { cfg, app, uploads: Mutex::new(HashMap::new()) })
    }

    /// Copy the request's files up and rewrite its paths.
    ///
    /// The output path is dealt with first: it is the one that may be refused
    /// before anything is copied.
    pub async fn prepare(&self, method: &str, params: Value) -> Result<(Value, Plan), HostError> {
        let mut params = params;
        let mut plan = Plan::default();

        let output = str_field(&params, output_field(method)).map(str::to_string);
        if let Some(local) = output {
            let overwrite = params.get("overwrite").and_then(Value::as_bool).unwrap_or(false);
            if !may_write(Path::new(&local), overwrite) {
                return Err(HostError::new("io_error", format!("{local} already exists")));
            }
            let remote = out_path(&self.cfg, Path::new(&local));
            plan.map.push((remote.to_string_lossy().into_owned(), local.clone()));
            plan.downloads.push((remote.clone(), PathBuf::from(&local)));
            set_field(&mut params, output_field(method), &remote.to_string_lossy());
        }

        let input = str_field(&params, input_field(method)).map(str::to_string);
        if let Some(local) = input {
            let remote = self.upload(Path::new(&local)).await.map_err(|e| HostError::new("io_error", e))?;
            plan.map.push((remote.clone(), local));
            set_field(&mut params, input_field(method), &remote);
        }

        Ok((params, plan))
    }

    /// Put the local paths back into the answer.
    pub fn restore(&self, result: &mut Value, plan: &Plan) {
        for (remote, local) in &plan.map {
            rewrite(result, remote, local);
        }
    }

    /// Copy the host's files back to where the user asked for them, and
    /// remove the far machine's copies.
    pub async fn fetch_back(&self, plan: &Plan) -> Result<(), String> {
        for (remote, local) in &plan.downloads {
            if let Err(e) = self.transfer_down(remote, local).await {
                // The file stays on the far machine: a copy that failed to come
                // back is better than no file at all.
                return Err(format!(
                    "{e} (the finished file is still at {} on {})",
                    remote.display(),
                    self.cfg.target
                ));
            }
            self.discard_dir(remote).await;
        }
        Ok(())
    }

    /// Remove what this module made on the far machine: the export's own
    /// directory. The inbox is left alone — it is the cache.
    pub async fn discard(&self, plan: &Plan) {
        for (remote, _) in &plan.downloads {
            self.discard_dir(remote).await;
        }
    }

    /// Remove the far machine's copy of one export.
    async fn discard_dir(&self, remote: &Path) {
        if let Some(dir) = remote.parent() {
            if let Err(e) = self.ssh(&format!("rm -rf {}", quote(&dir.to_string_lossy()))).await {
                log::warn!(target: "remote", "could not remove {}: {e}", dir.display());
            }
        }
    }

    /// Copy a local file up, once. A second request for the same file finds
    /// the copy already there; a file that changed since is copied again.
    async fn upload(&self, local: &Path) -> Result<String, String> {
        let abs = files::standardized(local);
        let key = inbox_key(&abs);
        let remote = self.cfg.inbox_path().join(&key).join(file_name(&abs));
        let (size, mtime_ms) = stat(&abs)?;

        let entry = {
            let mut map = self.uploads.lock().await;
            let existing = map.get(&key).filter(|e| e.size == size && e.mtime_ms == mtime_ms).cloned();
            match existing {
                Some(entry) => entry,
                None => {
                    let entry =
                        Arc::new(Upload { size, mtime_ms, remote: remote.clone(), cell: OnceCell::new() });
                    map.insert(key.clone(), Arc::clone(&entry));
                    entry
                }
            }
        };

        // Already copied (or already being copied by another request).
        if let Some(done) = entry.cell.get() {
            return done.clone();
        }
        let result = entry
            .cell
            .get_or_init(|| async {
                self.transfer_up(&abs, &entry.remote).await.map(|()| entry.remote.to_string_lossy().into_owned())
            })
            .await
            .clone();
        if result.is_err() {
            // A failed copy is not a cache entry: the next request tries again.
            let mut map = self.uploads.lock().await;
            if map.get(&key).is_some_and(|e| Arc::ptr_eq(e, &entry)) {
                map.remove(&key);
            }
        }
        result
    }

    /// Copy a local file up, and say so.
    async fn transfer_up(&self, local: &Path, remote: &Path) -> Result<(), String> {
        self.emit_transfer("up", local, None, false);
        let result = self.copy_up(local, remote).await;
        match &result {
            Ok(()) => self.emit_transfer("up", local, None, true),
            Err(e) => self.emit_transfer("up", local, Some(e), true),
        }
        result
    }

    async fn copy_up(&self, local: &Path, remote: &Path) -> Result<(), String> {
        if let Some(dir) = remote.parent() {
            self.ssh(&format!("mkdir -p {}", quote(&dir.to_string_lossy()))).await?;
        }
        let spec = format!("{}:{}", self.cfg.target, escape(remote));
        self.rsync(&[local.to_string_lossy().into_owned(), spec]).await
    }

    /// Copy a file the host wrote back to the user, and say so.
    async fn transfer_down(&self, remote: &Path, local: &Path) -> Result<(), String> {
        self.emit_transfer("down", local, None, false);
        let result = self.copy_down(remote, local).await;
        match &result {
            Ok(()) => self.emit_transfer("down", local, None, true),
            Err(e) => self.emit_transfer("down", local, Some(e), true),
        }
        result
    }

    async fn copy_down(&self, remote: &Path, local: &Path) -> Result<(), String> {
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let spec = format!("{}:{}", self.cfg.target, escape(remote));
        self.rsync(&[spec, local.to_string_lossy().into_owned()]).await
    }

    /// rsync over ssh, with the ssh options the launch uses — without
    /// `BatchMode` rsync's ssh would ask for a password and hang.
    async fn rsync(&self, args: &[String]) -> Result<(), String> {
        let mut cmd = Command::new("rsync");
        cmd.arg("-a").arg("--partial").arg("-e").arg(self.cfg.ssh_options());
        for arg in args {
            cmd.arg(arg);
        }
        let out = cmd
            .output()
            .await
            .map_err(|e| format!("rsync could not be started ({e}); rsync is needed to copy files to and from the render host"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(format!("rsync failed: {}", err.trim()));
        }
        Ok(())
    }

    /// Run a command on the far machine.
    async fn ssh(&self, remote_command: &str) -> Result<(), String> {
        let mut cmd = Command::new("ssh");
        cmd.args(["-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]);
        cmd.arg(&self.cfg.target).arg(remote_command);
        let out = cmd.output().await.map_err(|e| format!("ssh could not be started: {e}"))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let code = out.status.code().unwrap_or(-1);
            if code == 255 {
                return Err(format!("could not reach the render host {} over ssh: {}", self.cfg.target, err.trim()));
            }
            return Err(format!("ssh failed ({code}): {}", err.trim()));
        }
        Ok(())
    }

    /// Tell the webview that a copy is going on.
    fn emit_transfer(&self, direction: &str, path: &Path, error: Option<&str>, done: bool) {
        let mut payload = json!({
            "direction": direction,
            "path": path.to_string_lossy(),
            "done": done,
        });
        if let Some(error) = error {
            payload["error"] = json!(error);
        }
        let _ = self.app.emit("remote_transfer", payload);
    }
}

/// A string field of the parameters, when it has one.
fn str_field<'a>(params: &'a Value, field: Option<&'static str>) -> Option<&'a str> {
    let field = field?;
    params.get(field).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Write a string field of the parameters.
fn set_field(params: &mut Value, field: Option<&'static str>, value: &str) {
    let Some(field) = field else { return };
    if let Some(map) = params.as_object_mut() {
        map.insert(field.to_string(), Value::String(value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_names_are_shell_safe() {
        assert_eq!(file_name(Path::new("/p/DSC_0001.NEF")), "DSC_0001.NEF");
        assert_eq!(file_name(Path::new("/p/my photo (1)&'x'.cr3")), "my_photo__1___x_.cr3");
        assert_eq!(file_name(Path::new("/p/照片.dng")), "__.dng");
        assert_eq!(file_name(Path::new("/")), "file");
    }

    fn config() -> RemoteConfig {
        RemoteConfig {
            target: "ROG-laptop".into(),
            root: PathBuf::from("/home/jacky/spektralab"),
            device: None,
        }
    }

    #[test]
    fn the_inbox_key_is_sixteen_hex_of_the_path() {
        let key = inbox_key(Path::new("/photos/a/DSC_0001.NEF"));
        assert_eq!(key.len(), 16);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()), "{key} is not hex");
        // The same path, however it is written, gives the same folder.
        assert_eq!(key, inbox_key(Path::new("/photos/a/../a/DSC_0001.NEF")));
        // A different photograph gets a different one.
        assert_ne!(key, inbox_key(Path::new("/photos/a/DSC_0002.NEF")));
        // And it is the sha1 of the absolute path, computed independently.
        let digest = Sha1::digest(b"/photos/a/DSC_0001.NEF");
        let want: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
        assert_eq!(key, want);
    }

    #[test]
    fn an_existing_destination_is_refused_unless_overwrite() {
        let dir = std::env::temp_dir().join(format!("spk-remote-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("out.tif");
        std::fs::write(&file, b"x").unwrap();
        assert!(!may_write(&file, false));
        assert!(may_write(&file, true));
        let missing = dir.join("not-there.tif");
        assert!(may_write(&missing, false));
        assert!(may_write(&missing, true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_export_is_written_under_the_remote_out_directory() {
        let cfg = config();
        let remote = out_path(&cfg, Path::new("/Users/jacky/Desktop/print.tif"));
        let text = remote.to_string_lossy().into_owned();
        assert!(text.starts_with("/home/jacky/spektralab/out/"), "{text}");
        assert!(text.ends_with("/print.tif"), "{text}");
        // Two exports never share a directory.
        let other = out_path(&cfg, Path::new("/Users/jacky/Desktop/print.tif"));
        assert_ne!(remote.parent(), other.parent());
    }

    #[test]
    fn the_paths_are_the_protocol_s_ones() {
        assert_eq!(input_field("probe"), Some("path"));
        assert_eq!(input_field("thumbnail"), Some("path"));
        assert_eq!(input_field("open"), Some("path"));
        assert_eq!(input_field("write_image"), Some("source_path"));
        assert_eq!(input_field("render"), None);
        assert_eq!(output_field("export_image"), Some("path"));
        assert_eq!(output_field("write_image"), Some("path"));
        assert_eq!(output_field("export_cube"), Some("path"));
        assert_eq!(output_field("export_di"), Some("path"));
        assert_eq!(output_field("open"), None);
    }

    #[test]
    fn the_local_paths_come_back_in_the_answer() {
        let mut result = json!({
            "path": "/home/jacky/spektralab/out/abc/print.tif",
            "nested": {"paths": ["/home/jacky/spektralab/out/abc/print.tif", "other"]},
        });
        rewrite(&mut result, "/home/jacky/spektralab/out/abc/print.tif", "/Users/jacky/Desktop/print.tif");
        assert_eq!(result["path"], "/Users/jacky/Desktop/print.tif");
        assert_eq!(result["nested"]["paths"][0], "/Users/jacky/Desktop/print.tif");
        assert_eq!(result["nested"]["paths"][1], "other");
    }

    #[test]
    fn the_launch_is_an_ssh_that_runs_the_host_over_there() {
        let mut cfg = config();
        cfg.device = Some("1".into());
        let launch = cfg.launch();
        assert_eq!(launch.program, PathBuf::from("ssh"));
        assert_eq!(launch.args[0], "-T");
        assert!(launch.args.iter().any(|a| a == "BatchMode=yes"));
        assert!(launch.args.iter().any(|a| a == "ServerAliveInterval=15"));
        assert!(launch.args.iter().any(|a| a == "ServerAliveCountMax=4"));
        assert_eq!(launch.args[7], "ROG-laptop");
        let command = launch.args.last().unwrap();
        assert!(command.contains("/home/jacky/spektralab/host/spektralab-host"), "{command}");
        assert!(command.contains("--resources '/home/jacky/spektralab/host/engine'"), "{command}");
        assert!(command.contains("--device '1'"), "{command}");
    }

    #[test]
    fn a_path_with_a_space_still_reaches_the_far_machine_intact() {
        assert_eq!(escape(Path::new("/home/jacky/a b/print.tif")), "/home/jacky/a\\ b/print.tif");
        assert_eq!(quote("/home/jacky/a b"), "'/home/jacky/a b'");
    }
}
