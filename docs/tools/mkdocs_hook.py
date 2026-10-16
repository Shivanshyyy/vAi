"""빌드 전에 06-nodes.md 를 다시 생성하고, 빌드 후 테마 경계값을 바꾼다.

mkdocs.yml 의 hooks 에 등록되어 serve / build 양쪽에서 돈다. serve 는 watch 에
vai/nodes 가 들어 있어 노드 헤더를 저장하는 순간 여기까지 이어진다.

생성기는 내용이 같으면 파일을 쓰지 않으므로 재빌드 루프는 생기지 않는다.
"""

import importlib.util
from pathlib import Path

# 파일명에 하이픈이 있어 일반 import 로는 못 불러온다.
_PATH = Path(__file__).with_name("gen-node-docs.py")
_spec = importlib.util.spec_from_file_location("gen_node_docs", _PATH)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def on_pre_build(config, **kwargs):
    nodes, errors = _gen.collect()
    if errors:
        # --strict 여부와 무관하게 멈춘다. 스탬프를 못 읽었다는 건 문서에서
        # 노드가 통째로 빠진다는 뜻이다.
        raise RuntimeError(
            "노드 스탬프를 읽지 못했다:\n  " + "\n  ".join(errors)
        )
    for lang in _gen.STRINGS:
        _gen.write(nodes, lang)


# 왼쪽 내비게이션이 서랍에서 상시 표시로 바뀌는 창 너비. 테마는 이 값을
# CSS 와 JS 에 상수로 구워 내보내므로 mkdocs.yml 로는 손댈 수 없다.
# 기본 76.25em(1220px)은 늦어서, 우측 목차가 나타나는 60em(960px)에 맞춘다.
_BREAKPOINTS = {
    "76.25em": "60em",              # 데스크톱 레이아웃이 시작되는 하한
    "76.234375em": "59.984375em",   # 그 직전까지 쓰는 서랍 레이아웃의 상한
}


def on_post_build(config, **kwargs):
    site = Path(config["site_dir"])
    hits = dict.fromkeys(_BREAKPOINTS, 0)

    for path in [*site.glob("assets/stylesheets/main.*.min.css"),
                 *site.glob("assets/javascripts/bundle.*.min.js")]:
        text = path.read_text(encoding="utf-8")
        for old, new in _BREAKPOINTS.items():
            hits[old] += text.count(old)
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

    # 테마가 경계값 표기를 바꾸면 치환이 조용히 0건이 되어, 레이아웃만 슬그머니
    # 되돌아간다. 그 자리에서 멈춰야 원인을 찾아 헤매지 않는다.
    missed = [old for old, count in hits.items() if count == 0]
    if missed:
        raise RuntimeError(
            "테마에서 경계값을 찾지 못했다: " + ", ".join(missed) + "\n"
            "테마가 바뀐 것이니 새 표기를 찾아 _BREAKPOINTS 를 고칠 것."
        )
