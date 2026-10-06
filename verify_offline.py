"""禁止网络连接运行官方推理入口，核查文件、编号与必填输出。"""
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 3:
        raise SystemExit("用法：verify_offline.py 测试文件 结果目录")
    test_file, result_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    denied = []

    def reject_connection(sock, address):
        denied.append(str(address))
        raise RuntimeError("离线验证禁止建立网络连接")

    original_connect = socket.socket.connect
    socket.socket.connect = reject_connection
    sys.path.insert(0, str(ROOT / "participant"))
    spec = importlib.util.spec_from_file_location("entry", ROOT / "participant" / "run_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.main()
    finally:
        socket.socket.connect = original_connect
    inputs = module.read_jsonl(test_file)
    outputs = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in inputs] == [row["id"] for row in outputs], "预测编号或顺序不一致"
    for row in outputs:
        assert set(row) == {"id", "response_text", "emotion_label", "user_profile", "memory_refs"}
        assert row["response_text"].strip(), "回复为空"
        assert row["emotion_label"] in module.EMOTIONS, "情感字段无效"
        assert set(row["user_profile"]) == set(module.PROFILE_LABELS)
        for key, labels in row["user_profile"].items():
            assert len(labels) == len(set(labels)) and set(labels) <= module.PROFILE_LABELS[key]
        assert row["memory_refs"] == [], "没有公开记忆库时应输出空引用"
    assert not denied, "推理曾尝试访问网络"
    report = {"network_connections_attempted": len(denied), "samples": len(outputs),
              "id_and_order_preserved": True, "required_fields_valid": True,
              "说明": "本机Python推理离线验证；不替代官方Linux启动脚本和镜像验收。"}
    (result_dir / "offline_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("离线推理与输出检查通过", flush=True)


if __name__ == "__main__":
    main()
