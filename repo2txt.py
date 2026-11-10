import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

CHUNK_CHARS = 9_500_000


def _sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _utf8_stdout():
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def pack(src, out, include_git, chunk_mb):
    if not src.is_dir():
        sys.exit(f"[错误] 目录不存在: {src}")
    chunk_chars = int(chunk_mb * 1_000_000)
    out.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    count = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if not include_git and ".git" in rel.parts:
                continue
            tar.add(p, arcname=rel.as_posix(), recursive=False)
            count += 1

    blob = buf.getvalue()
    b64 = base64.b64encode(blob)
    parts = [b64[i:i + chunk_chars] for i in range(0, len(b64), chunk_chars)]
    for i, part in enumerate(parts, 1):
        (out / f"part_{i:04d}.txt").write_bytes(part)

    manifest = {
        "dir": src.name,
        "file_count": count,
        "tar_bytes": len(blob),
        "tar_sha256": _sha256(blob),
        "b64_chars": len(b64),
        "parts": len(parts),
        "chunk_chars": chunk_chars,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[打包完成] {count} 个文件,压缩后 {len(blob)/1e6:.1f}MB,"
          f"共 {len(parts)} 个 txt 分片(每个 ≤ {chunk_chars/1e6:.1f}MB)")
    print(f"[校验值]   tar_sha256 = {manifest['tar_sha256']}")
    print(f"[输出目录] {out.resolve()}")


def unpack(parts_dir, dest):
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"[错误] 找不到 {manifest_path},请确认 txt_parts 整个目录已拷全")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = sorted(parts_dir.glob("part_*.txt"))
    if len(files) != manifest["parts"]:
        sys.exit(f"[错误] 分片数量不符:期望 {manifest['parts']} 个,实际 {len(files)} 个。"
                 f"少拷或多拷了文件,请核对")
    for i, f in enumerate(files, 1):
        if f.name != f"part_{i:04d}.txt":
            sys.exit(f"[错误] 分片命名/顺序异常:{f.name},期望 part_{i:04d}.txt")

    b64 = b"".join(f.read_bytes() for f in files)
    if len(b64) != manifest["b64_chars"]:
        sys.exit(f"[错误] 拼接后长度不符:期望 {manifest['b64_chars']} 字符,实际 {len(b64)},"
                 f"某个分片内容不完整")

    try:
        blob = base64.b64decode(b64, validate=True)
    except Exception as e:
        sys.exit(f"[错误] base64 解码失败,分片可能被编辑或损坏:{e}")
    actual = _sha256(blob)
    if actual != manifest["tar_sha256"]:
        sys.exit(f"[错误] sha256 校验失败!\n  期望: {manifest['tar_sha256']}\n"
                 f"  实际: {actual}\n内容在传输中被改动,请重新拷贝")

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)

    print(f"[解包完成] 已恢复 {manifest['file_count']} 个文件到 {dest.resolve()}")
    print(f"[校验]     sha256 通过 ({actual[:16]}...)")


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="目录 ⇄ txt 分片互转")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pack", help="打包目录为 txt 分片")
    p1.add_argument("src", type=Path, help="要打包的源目录")
    p1.add_argument("-o", "--out", type=Path, default=Path("txt_parts"), help="分片输出目录")
    p1.add_argument("--include-git", action="store_true", help="连 .git 一起打包(默认排除)")
    p1.add_argument("--chunk-mb", type=float, default=9.5,
                    help="单个 txt 容量上限 MB(默认 9.5;设 12 通常可出单文件)")

    p2 = sub.add_parser("unpack", help="从 txt 分片恢复目录")
    p2.add_argument("parts_dir", type=Path, help="txt_parts 目录(含 manifest.json)")
    p2.add_argument("-o", "--out", type=Path, default=None, help="恢复到的目标目录(默认当前目录)")

    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.src, args.out, args.include_git, args.chunk_mb)
    else:
        dest = args.out or Path(".")
        unpack(args.parts_dir, dest)


if __name__ == "__main__":
    main()
