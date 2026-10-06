"""复用自写训练数据，增加说话角色与事实边界的提示，不引入验收答案。"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
source = ROOT / '数据/companion_v2'
target = ROOT / '数据/companion_v3'
target.mkdir(parents=True, exist_ok=True)
for name in ('train.jsonl', 'eval.jsonl'):
    shutil.copy2(source / name, target / name)
style = (source / 'style.txt').read_text(encoding='utf-8').strip() + ' 你在回应用户，不替用户说话；“我”指心语，“你”指用户。只依据对话中已经说出的事实，不臆测第三人的想法。坦诚说明自己是AI，没有人类的爱情体验。'
(target / 'style.txt').write_text(style, encoding='utf-8')
print('v3沿用102条自写训练对话和18道验收题；只修改角色边界提示。')
