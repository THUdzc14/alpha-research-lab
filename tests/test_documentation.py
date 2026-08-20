import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = (
    REPOSITORY_ROOT / "README.md",
    *sorted((REPOSITORY_ROOT / "docs").glob("*.md")),
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_local_documentation_links_resolve_within_the_repository():
    broken_links = []

    for source_path in MARKDOWN_FILES:
        source = source_path.read_text(encoding="utf-8")

        for match in MARKDOWN_LINK_PATTERN.finditer(source):
            target = match.group(1).split(maxsplit=1)[0].strip("<>")
            parsed = urlsplit(target)

            if parsed.scheme or target.startswith("#"):
                continue

            relative_target = unquote(parsed.path)
            resolved_target = (source_path.parent / relative_target).resolve()

            try:
                resolved_target.relative_to(REPOSITORY_ROOT)
            except ValueError:
                broken_links.append((source_path, target, "outside repository"))
                continue

            if not resolved_target.exists():
                broken_links.append((source_path, target, "missing"))

    details = "\n".join(
        f"{source.relative_to(REPOSITORY_ROOT)} -> {target} ({reason})"
        for source, target, reason in broken_links
    )
    assert not broken_links, details
