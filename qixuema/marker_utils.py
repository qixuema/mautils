from pathlib import Path

def marker_path_for(file_path: Path, status: str = "done") -> Path:
    """根据状态返回不同的标记文件路径
    status: 任意字符串，将作为文件后缀
    推荐："done", "error", "timeout", "invalid"
    """
    return file_path.with_suffix(f".{status}")

def cleanup_orphaned_markers(output_dir: Path, marker_suffix: str = ".done", file_suffix: str = ".npz") -> int:
    """Remove marker files whose corresponding NPZ files no longer exist."""
    removed = 0
    for marker in output_dir.rglob(f"*{marker_suffix}"):
        npz_file = marker.with_suffix(file_suffix)
        if not npz_file.exists():
            marker.unlink()
            removed += 1
    print(f"Cleaned up {removed} orphaned marker files.")
    return removed

def generate_markers_for_existing_files(
    output_dir: Path, 
    verify_size: bool = True, 
    min_size_mb: float = 0.1,
    marker_suffix: str = ".done",
    file_suffix: str = ".npz",
) -> int:
    """Generate marker files for existing NPZ files."""
    created = 0
    min_size_bytes = min_size_mb * 1024 * 1024
    
    for file_path in output_dir.rglob(f"*{file_suffix}"):
        if verify_size and file_path.stat().st_size < min_size_bytes:
            continue
            
        marker = marker_path_for(file_path, status=marker_suffix.lstrip("."))
        marker.touch(exist_ok=True)
        created += 1
    
    print(f"Generated {created} marker files for existing {file_suffix} files.")
    return created


def count_converted_samples(output_dir: Path, marker_suffix: str = ".done") -> int:
    """Count successfully converted samples based on marker files."""
    return sum(1 for _ in output_dir.rglob(f"*{marker_suffix}"))


def exists_done_marker(out_path: Path) -> bool:
    return marker_path_for(out_path, status="done").exists()