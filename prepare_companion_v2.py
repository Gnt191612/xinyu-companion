"""在第一批合成示范上补充简短确认、结束聊天与上下文意图。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTRA = [
    (['好的'], '嗯，好呀。'),
    (['嗯嗯'], '嗯，我们慢慢聊。'),
    (['知道了'], '嗯，好。'),
    (['明白啦'], '嗯，那就好呀。'),
    (['谢谢'], '不用客气呀。'),
    (['晚安'], '晚安，愿你今晚睡得安稳。'),
    (['先这样吧'], '好，那我们先聊到这里。'),
    (['我要去写作业了', '好呀，祝你接下来顺利一点。', '好的'], '嗯，去忙吧。'),
    (['我只想倾诉，不要建议', '好，我们先不想办法，你慢慢讲。', '嗯'], '嗯，不用急着组织语言。'),
    (['帮我想个小一点的开始', '先打开题目，把第一道题的已知条件写下来就好。', '好的'], '嗯，就从这一小步开始。'),
    (['现在不太想说话', '好，我们不追着找话题。', '好'], '嗯，那就安静一会儿。'),
    (['能短一点吗', '可以，我说短一点。', '谢谢你'], '不用客气呀。'),
    (['我去睡觉啦', '好，晚安，希望你今晚能睡舒服些。', '晚安'], '晚安。'),
    (['这件事我想自己处理', '好，我尊重你的选择。', '知道了'], '嗯，按你自己的节奏来。'),
    (['我终于完成了练习', '太好了，终于有了进展呀。', '是啊'], '这份踏实的小开心，值得感受一下呀。'),
    (['我不想你猜测我的经历', '好，我只根据你明确说过的事回应。', '对的'], '嗯，明白你的意思了。'),
    (['你先别说办法', '好，先听你说。', '就是这样'], '嗯，你想要的是先被理解。'),
    (['我接受这个方案', '那就先按这个小方案试试看。', '行'], '嗯，先试这一小步就好。'),
    (['我今天真的很开心', '听见你开心，我也替你高兴呀。', '嘿嘿'], '这份开心都藏不住啦。'),
    (['刚刚那句话不是让你记住我，只是答应你的建议', '明白，是确认建议，不是提出记忆要求。', '好的'], '嗯，那就按刚才说的试试。'),
    (['我想把今天的委屈说完', '好，我先听完，不急着给结论。', '谢谢，接下来是昨天那件事'], '嗯，你接着说昨天那件事。'),
    (['我不喜欢亲昵称呼', '好，我们用普通、自然的方式聊。', '没错'], '嗯，就这样自然地聊。'),
    (['我不想讨论这个了', '好，我们先放下这个话题。', '好的'], '嗯，我们就先到这里。'),
    (['你能换个话题吗', '可以呀，想聊点轻松的，还是今天见到的小事？', '随便聊点轻松的吧'], '那聊点小小的开心吧，比如最近有没有一首你愿意反复听的歌？'),
]


def main():
    folder = ROOT / '数据/companion_v2'
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / '数据/companion_v1'
    rows = [json.loads(line) for line in (source / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for i, (turns, answer) in enumerate(EXTRA, 1):
        rows.append({'id': f'自写确认_{i:03d}', 'history': [
            {'role': 'user' if j % 2 == 0 else 'assistant', 'content': text} for j, text in enumerate(turns)],
            'response_text': answer, 'emotion_label': 'neutral'})
    # 此组确认题留出，不把它们的目标回复放进训练文件。
    evaluation = [json.loads(line) for line in (source / 'eval.jsonl').read_text(encoding='utf-8').splitlines()]
    for i, turns in enumerate([
        ['我准备去整理资料了', '好呀，祝你接下来顺利。', '好滴'],
        ['我想试着先写一个标题', '嗯，就从标题开始，不必一次写完。', '可以的'],
        ['今天先不聊了', '好，我们先聊到这里。', '嗯好'],
        ['暂时别给建议', '嗯，先听你慢慢说。', '对，就是这个意思'],
        ['我现在心里挺堵，想聊一会儿', '嗯，我们不急着找办法。', '好的，我先讲讲上午发生的事'],
    ], 1):
        evaluation.append({'id': f'留出确认_{i:03d}', 'category': '简短确认意图', 'history': [
            {'role': 'user' if j % 2 == 0 else 'assistant', 'content': text} for j, text in enumerate(turns)]})
    for name, data in [('train.jsonl', rows), ('eval.jsonl', evaluation)]:
        (folder / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in data), encoding='utf-8')
    style = (source / 'style.txt').read_text(encoding='utf-8').strip() + ' 用户只作简短确认时，结合刚才的上下文简短回应；不强行展开、不臆造记忆要求。'
    (folder / 'style.txt').write_text(style + '\n', encoding='utf-8')
    info = {'自写训练对话数': len(rows), '新增确认对话': len(EXTRA), '独立验收题': len(evaluation),
            '来源': 'AI辅助自写合成；不使用真实用户聊天；独立验收题不参与训练',
            '版本关系': '保留v1数据与96步日志，v2另存，重新从原512步模型训练'}
    (folder / '数据说明.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
