"""全局配置。API Key 从 .env 读取，不写进代码。"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK，统一用 UTF-8 输出中文
    _stream.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # 屏蔽 Windows 上无害的 symlink 警告

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROCESSED_DIR = ROOT / "data" / "processed"

# Chroma 1.x 在 Windows 上无法把 HNSW 索引文件写进含中文等非 ASCII 字符的路径（会报 "Error loading hnsw index"），
# 这种情况下改存到用户目录下的 .annual-report-rag/（不用 %LOCALAPPDATA%：打包应用访问 AppData 会被系统重定向，
# 不同程序看到的不是同一个目录）；也可用环境变量 CHROMA_DIR 指定
def _chroma_dir() -> Path:
    if os.getenv("CHROMA_DIR"):
        return Path(os.environ["CHROMA_DIR"])
    default = ROOT / "storage" / "chroma"
    if str(default).isascii() or os.name != "nt":
        return default
    return Path.home() / ".annual-report-rag" / "chroma"


CHROMA_DIR = _chroma_dir()

# 文件名 -> 公司简称，用于给 chunk 加公司前缀、按公司过滤检索
COMPANIES = ["宁德时代", "比亚迪", "工商银行"]

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# bge 官方建议：检索时给"问题"加这句指令，文档侧不加
EMBED_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

CHUNK_SIZES = [300, 800]  # 对比实验的两种切块大小（字）
CHUNK_OVERLAP_RATIO = 0.15
DEFAULT_CHUNK_SIZE = 800
DEFAULT_TOP_K = 5

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def company_of(doc_name: str) -> str:
    return next((c for c in COMPANIES if c in doc_name), doc_name)
