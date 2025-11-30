"""
File management utilities for VAHAN web scraper.
Handles file operations, directory management, and data export.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from datetime import datetime

from ..core.config import Config
from ..core.exceptions import ExportError
from ..utils.logging_utils import get_logger

class FileManager:
    """Utility class for file and directory management."""
    
    def __init__(self):
        """Initialize the file manager with full validation, safety, and diagnostics."""
        
        # ------------------------------------------------------------------
        # 1. Robust logger creation
        # ------------------------------------------------------------------
        try:
            self.logger = get_logger(self.__class__.__name__)
        except Exception:
            # Fallback logger
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
    
        self.logger.debug("Initializing FileManager instance...")
    
        # ------------------------------------------------------------------
        # 2. Ensure required directories (with recovery)
        # ------------------------------------------------------------------
        try:
            Config.ensure_directories()
            self.logger.debug("Config directories verified.")
        except Exception as e:
            self.logger.error(f"Directory setup failed: {e}")
            try:
                # Auto-create fallback directory
                fallback_dir = Path("./runtime_fallback")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.logger.warning(f"Using fallback directory: {fallback_dir}")
            except Exception as inner_e:
                self.logger.critical(f"Failed to create fallback directory: {inner_e}")
    
        # ------------------------------------------------------------------
        # 3. Capture environment details (useful for debugging)
        # ------------------------------------------------------------------
        try:
            import platform, os
    
            self.environment_info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "cwd": os.getcwd(),
                "user": os.getenv("USERNAME") or os.getenv("USER") or "Unknown"
            }
    
            self.logger.debug(f"Environment: {self.environment_info}")
    
        except Exception as e:
            self.environment_info = {}
            self.logger.warning(f"Unable to fetch environment information: {e}")
    
        # ------------------------------------------------------------------
        # 4. Initialize internal state
        # ------------------------------------------------------------------
        self.initialized = True
        self.last_operation = None
        self.session_id = uuid.uuid4().hex  # unique context identifier
    
        self.logger.debug(f"FileManager initialized successfully. Session ID: {self.session_id}")
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        directory: Path = None,
        format: str = "csv",
        compress: bool = False,
        add_timestamp: bool = True
    ) -> str:
        """Save DataFrame to file with full validation, safety, and atomic write.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
            directory (Path, optional): Target directory
            format (str): 'csv', 'excel', 'json'
            compress (bool): Compress output (ZIP)
            add_timestamp (bool): Auto-append timestamp if missing
            
        Returns:
            str: Path to saved file
            
        Raises:
            ExportError
        """
    
        try:
            # ---------------------------------------------------------
            # 1. Validate DataFrame
            # ---------------------------------------------------------
            if not isinstance(df, pd.DataFrame):
                raise ExportError("Input is not a DataFrame")
    
            if df.empty:
                self.logger.warning("Attempting to save an EMPTY DataFrame")
    
            # ---------------------------------------------------------
            # 2. Directory handling with fallback
            # ---------------------------------------------------------
            if directory is None:
                directory = Config.OUTPUT_DIR
    
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Primary directory creation failed: {e}")
                directory = Path("./export_fallback")
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.warning(f"Using fallback directory: {directory}")
    
            # ---------------------------------------------------------
            # 3. Infer format from filename extension
            # ---------------------------------------------------------
            if "." in filename:
                ext = filename.rsplit(".", 1)[1].lower()
                if ext in ["csv", "xlsx", "json"]:
                    format = ext if ext != "xlsx" else "excel"
    
            # ---------------------------------------------------------
            # 4. Validate format
            # ---------------------------------------------------------
            allowed_formats = ["csv", "excel", "json"]
            if format.lower() not in allowed_formats:
                raise ExportError(
                    f"Unsupported format '{format}'. Allowed: {allowed_formats}"
                )
    
            # ---------------------------------------------------------
            # 5. Sanitize filename
            # ---------------------------------------------------------
            safe_filename = "".join(
                c for c in filename if c not in r'<>:"/\|?*'
            )
    
            if safe_filename != filename:
                self.logger.warning(f"Unsafe characters removed from filename: {filename}")
                filename = safe_filename
    
            # ---------------------------------------------------------
            # 6. Add timestamp (if not already present)
            # ---------------------------------------------------------
            if add_timestamp and not any(char.isdigit() for char in filename):
                ts = datetime.now().strftime(Config.TIMESTAMP_FORMAT)
                base, ext = (
                    filename.rsplit(".", 1)
                    if "." in filename
                    else (filename, format)
                )
                filename = f"{base}_{ts}.{ext}"
    
            # ---------------------------------------------------------
            # 7. Generate full path
            # ---------------------------------------------------------
            filepath = directory / filename
    
            # Add hash suffix if file exists
            if filepath.exists():
                hash_suffix = uuid.uuid4().hex[:8]
                filepath = filepath.with_name(
                    f"{filepath.stem}_{hash_suffix}{filepath.suffix}"
                )
    
            # Temporary atomic-write path
            temp_path = filepath.with_suffix(".tmp")
    
            # ---------------------------------------------------------
            # 8. Actual saving
            # ---------------------------------------------------------
            if format == "csv":
                df.to_csv(temp_path, index=False)
            elif format == "excel":
                df.to_excel(temp_path, index=False)
            elif format == "json":
                df.to_json(temp_path, orient="records", indent=2)
    
            # ---------------------------------------------------------
            # 9. Compression (ZIP)
            # ---------------------------------------------------------
            if compress:
                zip_path = str(filepath) + ".zip"
                import zipfile
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(temp_path, arcname=filename)
                temp_path.unlink(missing_ok=True)
                self.logger.info(f"💾 Saved compressed file: {zip_path}")
                return zip_path
    
            # ---------------------------------------------------------
            # 10. Atomic replace
            # ---------------------------------------------------------
            temp_path.replace(filepath)
    
            # ---------------------------------------------------------
            # 11. Logging & return
            # ---------------------------------------------------------
            self.logger.info(
                f"💾 Saved {len(df)} records → {filepath} (format={format}, compressed={compress})"
            )
            return str(filepath)
    
        except Exception as e:
            self.logger.exception("Save failed")
            raise ExportError(f"Failed to save DataFrame: {e}")
    
    def load_dataframe(
    self,
    filepath: Union[str, Path],
    format: str = None,
    encoding: str = "utf-8",
    safe_mode: bool = True,
) -> pd.DataFrame:
    
    """Fully validated and safe DataFrame loader with auto-detection.

    Args:
        filepath (str | Path): Input file path.
        format (str, optional): 'csv', 'excel', 'json'. Auto-detect if None.
        encoding (str): Primary encoding to try.
        safe_mode (bool): Enables extra integrity checks.

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError
        ExportError
    """
    try:
        filepath = Path(filepath)

        # ---------------------------------------------------------
        # 1. Existence check
        # ---------------------------------------------------------
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # ---------------------------------------------------------
        # 2. Auto-decompress ZIP
        # ---------------------------------------------------------
        if filepath.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(filepath, "r") as z:
                names = z.namelist()
                if not names:
                    raise ExportError("ZIP archive is empty")
                extracted_path = filepath.parent / names[0]
                z.extract(names[0], filepath.parent)
                filepath = extracted_path
                self.logger.info(f"🗜️ Extracted ZIP → {filepath}")

        # ---------------------------------------------------------
        # 3. Auto-detect format
        # ---------------------------------------------------------
        suffix = filepath.suffix.lower().lstrip(".")
        if format is None:
            format = "excel" if suffix in ["xlsx", "xls"] else suffix

        if format not in ["csv", "json", "excel"]:
            raise ExportError(
                f"Unsupported format '{format}'. "
                f"Allowed: csv, excel, json"
            )

        # ---------------------------------------------------------
        # 4. Safe file size check
        # ---------------------------------------------------------
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            self.logger.warning(
                f"⚠️ Large file detected ({size_mb:.2f} MB). "
                "Load may take time."
            )

        # ---------------------------------------------------------
        # 5. Load logic
        # ---------------------------------------------------------
        df = None

        if format == "csv":
            # Try multiple encodings if needed
            try:
                df = pd.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="ISO-8859-1")
            except Exception:
                df = pd.read_csv(filepath, engine="python")  # fallback

        elif format == "excel":
            try:
                df = pd.read_excel(filepath)
            except Exception:
                df = pd.read_excel(filepath, engine="openpyxl")

        elif format == "json":
            try:
                df = pd.read_json(filepath)
            except ValueError:
                # Handle JSON lines
                df = pd.read_json(filepath, lines=True)

        if df is None:
            raise ExportError("Unknown failure during load")

        # ---------------------------------------------------------
        # 6. Safe mode (integrity checks)
        # ---------------------------------------------------------
        if safe_mode:
            # Empty DataFrame warning
            if df.empty:
                self.logger.warning("⚠️ Loaded DataFrame is EMPTY")

            # Column sanity check
            if len(df.columns) == 1 and "," in df.columns[0]:
                raise ExportError("File appears corrupted (bad delimiter?)")

            # JSON corruption check
            if format == "json":
                if not isinstance(df, pd.DataFrame):
                    raise ExportError("JSON did not load as table structure")

        # ---------------------------------------------------------
        # 7. Logging + Schema summary
        # ---------------------------------------------------------
        self.logger.info(
            f"📖 Loaded {len(df)} rows | {len(df.columns)} columns | from {filepath}"
        )

        self.logger.debug(
            "📌 Column summary: " + ", ".join(df.columns.astype(str).tolist())
        )

        return df

    except Exception as e:
        self.logger.exception("Load failed")
        raise ExportError(f"Failed to load DataFrame: {e}")

    
    def create_backup(
    self,
    source_path: Union[str, Path],
    backup_dir: Path = None,
    compress: bool = True,
    ignore_patterns: List[str] = None,
    verify_integrity: bool = True,
    max_retries: int = 3,
) -> str:
    """
    Enterprise-grade backup creator with compression, versioning,
    integrity checks, large directory support, and collision handling.

    Args:
        source_path: File or directory to backup
        backup_dir: Destination directory (default: BASE_DIR/backups)
        compress: ZIP-compress the backup
        ignore_patterns: List of wildcard patterns to ignore
        verify_integrity: Perform hash verification after backup
        max_retries: Retries for transient IO errors

    Returns:
        str: Full path to created backup
    """
    import hashlib
    import fnmatch
    import zipfile

    source_path = Path(source_path)

    # ---------------------------------------------------------
    # 1. Validate existence
    # ---------------------------------------------------------
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    # ---------------------------------------------------------
    # 2. Backup directory setup
    # ---------------------------------------------------------
    if backup_dir is None:
        backup_dir = Config.BASE_DIR / "backups"

    backup_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 3. Timestamp + versioned backup name
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime(Config.TIMESTAMP_FORMAT)
    base_name = f"{source_path.name}_{timestamp}"

    # Avoid name collision
    backup_name = base_name
    counter = 1
    while (backup_dir / backup_name).exists():
        backup_name = f"{base_name}_v{counter}"
        counter += 1

    backup_path = backup_dir / backup_name

    # ---------------------------------------------------------
    # 4. Disk space check (threshold: 100MB free)
    # ---------------------------------------------------------
    try:
        free_space = shutil.disk_usage(str(backup_dir)).free / (1024**2)
        if free_space < 100:
            self.logger.warning("⚠️ Low disk space (<100MB). Backup may fail.")
    except:
        pass  # not fatal

    # ---------------------------------------------------------
    # 5. Ignore patterns
    # ---------------------------------------------------------
    default_ignores = ["__pycache__", "*.tmp", "*.log", ".DS_Store"]
    ignore_patterns = (ignore_patterns or []) + default_ignores

    def should_ignore(path: Path):
        return any(fnmatch.fnmatch(path.name, pat) for pat in ignore_patterns)

    # ---------------------------------------------------------
    # 6. File/Directory copy with retry
    # ---------------------------------------------------------
    def retry_copy_file(src, dst):
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

    try:
        if source_path.is_file():
            # Non-compressed direct backup
            retry_copy_file(source_path, backup_path)

        elif source_path.is_dir():
            # Directory backup (possibly compressed)
            if compress:
                # ZIP backup
                zip_path = Path(str(backup_path) + ".zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    for root, dirs, files in os.walk(source_path):
                        root_path = Path(root)
                        for file in files:
                            file_path = root_path / file
                            if not should_ignore(file_path):
                                arcname = file_path.relative_to(source_path)
                                try:
                                    z.write(file_path, arcname)
                                except PermissionError:
                                    self.logger.warning(f"⚠️ Skipping (permission denied): {file_path}")

                backup_path = zip_path

            else:
                # Normal directory copy
                shutil.copytree(
                    source_path,
                    backup_path,
                    ignore=lambda src, names: [
                        n for n in names if should_ignore(Path(n))
                    ],
                    dirs_exist_ok=False
                )

        else:
            raise FileNotFoundError(f"Unknown source type: {source_path}")

    except Exception as e:
        self.logger.error(f"❌ Backup failed: {e}")
        raise

    # ---------------------------------------------------------
    # 7. Integrity verification
    # ---------------------------------------------------------
    if verify_integrity and backup_path.is_file() and source_path.is_file():
        def sha256(file_path):
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        orig_hash = sha256(source_path)
        backup_hash = sha256(backup_path)

        if orig_hash != backup_hash:
            self.logger.error("❌ Backup integrity verification FAILED!")
            raise Exception(
                f"Backup corrupted: Hash mismatch for {backup_path}"
            )

    # ---------------------------------------------------------
    # 8. Success log
    # ---------------------------------------------------------
    self.logger.info(f"📦 Backup created: {backup_path}")
    return str(backup_path)

    
    def cleanup_old_files(
    self,
    directory: Path,
    days_old: int = 30,
    pattern: str = "*"
) -> int:
    """
    Clean up files older than a specified age inside a directory.

    Args:
        directory (Path): Directory to clean.
        days_old (int): Delete files older than this many days.
        pattern (str): Glob pattern to filter files.

    Returns:
        int: Number of deleted files.

    Raises:
        ValueError: If invalid arguments are provided.
    """
    # ----------- VALIDATION -----------
    if not isinstance(directory, Path):
        raise ValueError("directory must be a Path object")

    if days_old < 0:
        raise ValueError("days_old cannot be negative")

    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("Invalid pattern")

    # Directory not found
    if not directory.exists():
        self.logger.warning(f"⚠️ Directory does not exist: {directory}")
        return 0
    
    if not directory.is_dir():
        self.logger.error(f"❌ Path is not a directory: {directory}")
        return 0

    # ----------- DELETE LOGIC -----------
    cutoff_timestamp = datetime.now().timestamp() - (days_old * 86400)
    deleted_count = 0
    failed_count = 0

    self.logger.debug(
        f"🔍 Starting cleanup in: {directory}, "
        f"Older than: {days_old} days, Pattern: '{pattern}'"
    )

    try:
        for file_path in directory.glob(pattern):
            # Skip directories
            if not file_path.is_file():
                continue

            file_mtime = file_path.stat().st_mtime

            # Skip files newer than cutoff
            if file_mtime >= cutoff_timestamp:
                continue

            try:
                file_path.unlink()
                deleted_count += 1
                self.logger.debug(f"🗑️ Deleted: {file_path}")

            except Exception as del_err:
                failed_count += 1
                self.logger.error(
                    f"❌ Failed to delete {file_path}: {del_err}"
                )

        # ----------- SUMMARY LOGGING -----------
        if deleted_count > 0:
            self.logger.info(
                f"🧹 Cleanup complete — Deleted {deleted_count} old file(s) from {directory}"
            )

        if failed_count > 0:
            self.logger.warning(
                f"⚠️ {failed_count} file(s) could not be deleted due to permission or lock issues"
            )

        return deleted_count

    except Exception as e:
        self.logger.error(f"❌ Unexpected cleanup error: {e}")
        return deleted_count    # return partial progress instead of 0

    
    def get_file_info(self, filepath: Union[str, Path]) -> dict:
        """Get information about a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            dict: File information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {"exists": False}
        
        stat = filepath.stat()
        
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "is_file": filepath.is_file(),
            "is_directory": filepath.is_dir(),
            "extension": filepath.suffix.lower()
        }def get_file_info(
    self,
    filepath: Union[str, Path],
    compute_hash: bool = False
) -> dict:
    """
    Get detailed information about a file or directory.

    Args:
        filepath (str | Path): Path to the file or directory.
        compute_hash (bool): Whether to compute MD5 & SHA256 checksum (files only).

    Returns:
        dict: File metadata.
    """
    try:
        filepath = Path(filepath).expanduser().resolve()

        # ----------- BASE RESPONSE -----------
        info = {
            "path": str(filepath),
            "exists": filepath.exists(),
            "is_file": filepath.is_file(),
            "is_directory": filepath.is_dir(),
            "extension": filepath.suffix.lower() if filepath.is_file() else "",
        }

        if not filepath.exists():
            self.logger.warning(f"⚠️ File not found: {filepath}")
            return info

        # ----------- STAT METADATA -----------
        stat = filepath.stat()

        info.update({
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 4),
            "size_human": self._human_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "accessed": datetime.fromtimestamp(stat.st_atime),
            "permissions": oct(stat.st_mode)[-3:],   # like 755, 644, etc.
            "owner_uid": stat.st_uid if hasattr(stat, "st_uid") else None,
            "owner_gid": stat.st_gid if hasattr(stat, "st_gid") else None,
        })

        # ----------- HASH CALCULATION -----------
        if filepath.is_file() and compute_hash:
            try:
                info.update(self._compute_file_hash(filepath))
            except Exception as e:
                self.logger.error(f"❌ Error calculating hash for {filepath}: {e}")
                info.update({"md5": None, "sha256": None})

        self.logger.debug(f"📄 Retrieved info for: {filepath}")

        return info

    except Exception as e:
        self.logger.error(f"❌ Failed to get file info: {e}")
        return {"exists": False, "error": str(e)}


    # -------------------------------------------------------------------
    # Helper function for human-readable size
    # -------------------------------------------------------------------
    def _human_size(self, size_bytes: int) -> str:
        """Convert bytes into human-readble units."""
        try:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if size_bytes < 1024:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.2f} PB"
        except Exception:
            return f"{size_bytes} B"
    
    
    # -------------------------------------------------------------------
    # Hash calculation helper
    # -------------------------------------------------------------------
    def _compute_file_hash(self, filepath: Path) -> dict:
        """Compute MD5 and SHA256 for a file."""
        import hashlib
        
        md5_hash = hashlib.md5()
        sha_hash = hashlib.sha256()
    
        with filepath.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
                sha_hash.update(chunk)
    
        return {
            "md5": md5_hash.hexdigest(),
            "sha256": sha_hash.hexdigest(),
        }
    
        
    from typing import List, Union, Optional
    from pathlib import Path
    from datetime import datetime
    
    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
        include_hidden: bool = False,
        sort_by: str = "name",           # "name", "size", "modified"
        descending: bool = False,
        min_size: int = None,            # bytes
        max_size: int = None,            # bytes
        modified_after: datetime = None,
        modified_before: datetime = None,
        return_info: bool = False        # if True = returns list[dict] with metadata
    ) -> List[Union[Path, dict]]:
        """
        Ultra-extended file listing with filtering, sorting, hidden file control,
        and optional rich metadata.
    
        Args:
            directory (str|Path): Directory to search.
            pattern (str): File matching pattern.
            recursive (bool): Whether to scan subdirectories.
            include_hidden (bool): Include hidden files (.* files).
            sort_by (str): Sort key: "name", "size", "modified".
            descending (bool): Reverse sort order.
            min_size (int): Minimum file size in bytes.
            max_size (int): Maximum file size in bytes.
            modified_after (datetime): Filter: modified after this date.
            modified_before (datetime): Filter: modified before this date.
            return_info (bool): If True, return dict metadata instead of just paths.
    
        Returns:
            list[Path] or list[dict]: List of files or metadata dictionaries.
        """
    
        try:
            directory = Path(directory).expanduser().resolve()
    
            if not directory.exists() or not directory.is_dir():
                self.logger.warning(f"⚠️ Directory not found: {directory}")
                return []
    
            # --------------------- SEARCH ---------------------
            files = (
                directory.rglob(pattern) if recursive 
                else directory.glob(pattern)
            )
    
            files = [f for f in files if f.is_file()]
    
            # --------------------- HIDDEN FILTER ---------------------
            if not include_hidden:
                files = [f for f in files if not f.name.startswith(".")]
    
            # --------------------- SIZE FILTERS ---------------------
            if min_size is not None:
                files = [f for f in files if f.stat().st_size >= min_size]
    
            if max_size is not None:
                files = [f for f in files if f.stat().st_size <= max_size]
    
            # --------------------- DATE FILTERS ---------------------
            if modified_after is not None:
                files = [
                    f for f in files
                    if datetime.fromtimestamp(f.stat().st_mtime) >= modified_after
                ]
    
            if modified_before is not None:
                files = [
                    f for f in files
                    if datetime.fromtimestamp(f.stat().st_mtime) <= modified_before
                ]
    
            # --------------------- SORTING ---------------------
            if sort_by == "name":
                files.sort(key=lambda f: f.name.lower(), reverse=descending)
            elif sort_by == "size":
                files.sort(key=lambda f: f.stat().st_size, reverse=descending)
            elif sort_by == "modified":
                files.sort(key=lambda f: f.stat().st_mtime, reverse=descending)
            else:
                self.logger.warning(f"⚠️ Invalid sort_by '{sort_by}', using 'name'")
                files.sort(key=lambda f: f.name.lower())
    
            # --------------------- RETURN METADATA? ---------------------
            if return_info:
                enriched = []
                for f in files:
                    stat = f.stat()
                    enriched.append({
                        "path": str(f),
                        "name": f.name,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 4),
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "created": datetime.fromtimestamp(stat.st_ctime),
                        "extension": f.suffix.lower(),
                        "is_hidden": f.name.startswith("."),
                    })
                return enriched
    
            return files
    
        except Exception as e:
            self.logger.error(f"❌ Error listing files: {e}")
            return []

    
    from typing import Union
    from pathlib import Path
    import os
    import stat
    
    def ensure_directory(
        self,
        directory: Union[str, Path],
        create: bool = True,
        require_writable: bool = True,
        mode: int = 0o755,
        resolve_symlinks: bool = True
    ) -> Path:
        """
        Ultra-robust directory initializer.
    
        Args:
            directory (str|Path): Directory path to validate or create.
            create (bool): Create the directory if not present.
            require_writable (bool): Ensure directory is writable.
            mode (int): Permission mode to apply when creating.
            resolve_symlinks (bool): Resolve symlinks to their target paths.
    
        Returns:
            Path: The normalized, validated, resolved directory path.
    
        Raises:
            NotADirectoryError: If the path exists but is not a directory.
            PermissionError: If directory is not writable and require_writable=True.
            ValueError: If invalid directory path.
        """
    
        try:
            # Convert and resolve path safely
            directory = Path(directory).expanduser()
    
            if resolve_symlLinks:
                directory = directory.resolve()
    
            # Basic validation
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(f"Path exists but is not a directory: {directory}")
    
            # Create directory if needed
            if not directory.exists():
                if create:
                    directory.mkdir(parents=True, exist_ok=True)
                    os.chmod(directory, mode)
                    self.logger.info(f"📁 Created directory: {directory}")
                else:
                    raise FileNotFoundError(f"Directory does not exist: {directory}")
            else:
                self.logger.debug(f"📁 Directory already exists: {directory}")
    
            # Check write permissions
            if require_writable:
                test_file = directory / ".write_test.tmp"
                try:
                    with open(test_file, "w") as f:
                        f.write("ok")
                    test_file.unlink()
                except Exception:
                    raise PermissionError(f"Directory is not writable: {directory}")
    
            return directory
    
        except Exception as e:
            self.logger.error(f"❌ ensure_directory failed for '{directory}': {e}")
            raise

    
    def get_available_filename(
        self,
        directory: Path,
        base_name: str,
        extension: str = "csv",
        use_timestamp: bool = True,
        max_attempts: int = 9999
    ) -> str:
        """
        Generate a collision-proof filename inside a directory.
        
        Features:
            ✓ Auto-sanitizes filename
            ✓ Validates extension
            ✓ Optional timestamp prefix
            ✓ Infinite-safe counter fallback
            ✓ Logging
            ✓ prevents accidental overwrite
            ✓ Works on Windows/Linux/Mac safely
    
        Args:
            directory (Path): Directory to check in
            base_name (str): Base filename (no extension)
            extension (str): File extension without the dot
            use_timestamp (bool): Add a timestamp before falling back to counters
            max_attempts (int): Safety cap to avoid infinite loops
    
        Returns:
            str: A guaranteed-unique filename
        """
    
        # --- Normalize and validate ---
        directory = Path(directory)
        extension = extension.lstrip(".").lower()
    
        # Sanitize base_name: remove dangerous/special characters
        safe_name = "".join(c for c in base_name if c.isalnum() or c in ("_", "-", " ")).strip()
        if not safe_name:
            safe_name = "file"
    
        # Initial filename
        filename = f"{safe_name}.{extension}"
    
        # If free → return immediately
        if not (directory / filename).exists():
            return filename
    
        # Step 1: Add timestamp
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}.{extension}"
            if not (directory / filename).exists():
                self.logger.debug(f"Using timestamped filename: {filename}")
                return filename
    
        # Step 2: Fallback → numbered filenames
        for counter in range(1, max_attempts + 1):
            filename = f"{safe_name}_{counter}.{extension}"
            if not (directory / filename).exists():
                self.logger.debug(f"Using numbered filename: {filename}")
                return filename
    
        # Step 3: Absolute fallback (very unlikely)
        unique_hash = uuid.uuid4().hex[:8]
        filename = f"{safe_name}_{unique_hash}.{extension}"
        self.logger.warning(f"Fallback to hashed filename: {filename}")
        return filename
