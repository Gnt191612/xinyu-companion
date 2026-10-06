"""生成成对角色训练与不重叠的专项评估，不读取评估答案训练。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
folder = ROOT / '数据/companion_v23_role'
assert not folder.exists(), '不得覆盖已使用的数据'
folder.mkdir()


def row(first, middle, last, answer, direction):
    return {'history': [{'role': 'user', 'content': first}, {'role': 'assistant', 'content': middle},
                        {'role': 'user', 'content': last}], 'response_text': answer,
            'emotion_label': 'neutral', 'category': direction}


train = []
topics = ['今天的课程', '周末看电影', '小时候养猫', '刚读的小说', '做饭的趣事', '一次旅行', '最近的烦心事', '画画的经历']
for topic in topics:
    for ack in ['好的', '好滴', '嗯嗯', '好呀']:
        train.append(row(f'我想跟你讲讲{topic}。', '好呀，我在听，你慢慢讲。', ack,
                         '嗯，我听着呢，你慢慢讲呀。', '用户讲述_助手倾听'))
        train.append(row(f'你能给我编个关于{topic}的小故事吗？我想听。', '好呀，那我开始讲啦。', ack,
                         f'有一天，一只小猫想把{topic}写成一页日记。它认真挑了支笔，却先在纸上留下了一个小爪印。', '助手讲述_用户倾听'))
for first, middle, last, answer, category in [
    ('我想讲今天的事。', '好呀，我在听。', '不是我听你讲，是你听我讲。', '对呀，是你讲，我听着呢。', '角色纠正'),
    ('我想听你讲个故事。', '好呀，你慢慢说。', '弄反啦，是你讲，我来听。', '是我弄反啦，那我开始讲。小猫今天发现了窗边的一朵花，它凑过去闻了闻，鼻尖也沾上了一点花香。', '角色纠正'),
    ('我刚才想说点事。', '好呀，我在听。', '现在不想讲了。', '好呀，那就先不讲，我陪你安静待会儿。', '否定讲述'),
    ('我去倒杯水再回来。', '好呀，回来接着聊。', '好呀', '好呀，不着急，回来再聊。', '暂离'),
    ('今天先到这里，我去办事。', '好呀，祝你顺利。', '嗯嗯', '拜拜呀。', '告别'),
    ('我先睡了。', '好呀，早点休息。', '好的', '晚安呀，好好休息。', '睡觉'),
    ('我想分析一句台词。', '好呀，说给我听。', '台词是“你在听”，这句话谁在听？', '这句台词里的“你”是听的人呀，需要结合人物关系判断它指谁。', '引用人称'),
    ('我说的是我心里难受。', '我也难受。', '不是你难受，是我难受。', '是你心里难受呀，我刚才说错了。我在这儿听你说。', '感受归属'),
]:
    train.append(row(first, middle, last, answer, category))
# 只回放人工可审阅的纯聊天目标，排除旧的元话语目标。
old = [json.loads(line) for line in (ROOT / '数据/companion_v22_local/train.jsonl').read_text(encoding='utf8').splitlines()]
clean = [x for x in old if x['category'] == 'v22纯聊天目标与用户指导'][:16]
train.extend(clean)
dev = []
for topic in ['排练时的小插曲', '拼好模型的经过', '公交站遇到的小狗', '逛集市的新发现']:
    for ack in ['行', '好的呢']:
        dev.append(row(f'我准备给你讲讲{topic}。', '好呀，我听着，你慢慢说。', ack, '', '用户讲述_助手倾听'))
        dev.append(row(f'能不能编个{topic}的故事讲给我听？', '可以呀，我来讲。', ack, '', '助手讲述_用户倾听'))
dev += [
    row('我先去取外卖，待会儿回来。', '好呀，回来再聊。', '好呀', '', '暂离'),
    row('今晚困了，先去休息。', '好呀，早点睡。', '嗯嗯', '', '睡觉'),
    row('我本来想讲个事。', '好呀，我在听。', '我改主意了，不讲啦。', '', '否定讲述'),
    row('我想问一句台词。', '好呀。', '“你在听我说话”里的听者是谁？', '', '引用人称'),
]
assert set(topics).isdisjoint({'排练时的小插曲', '拼好模型的经过', '公交站遇到的小狗', '逛集市的新发现'})
for split, rows in [('train', train), ('dev', dev)]:
    for i, sample in enumerate(rows, 1):
        sample['id'] = f'role_{split}_{i:06d}'
    (folder / f'{split}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf8')
(folder / 'style.txt').write_text((ROOT / '数据/companion_v22_local/style.txt').read_text(encoding='utf8'), encoding='utf8')
cfg = json.loads((ROOT / '训练流程v22.json').read_text(encoding='utf8'))
cfg.update(initial_adapter='实验/陪伴强化本机_v22/candidate/adapter',
           evaluation_adapter='实验/陪伴强化本机_v23/candidate/adapter',
           merged_model='D:/心语模型缓存/情感陪伴_v23_未启用合并',
           data='数据/companion_v23_role', dev_data='数据/companion_v23_role',
           experiment='实验/陪伴强化本机_v23', steps=160, learning_rate=0.00001, replay_count=32)
(ROOT / '训练流程v23.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf8')
print(f'专项训练{len(train)}题，评估{len(dev)}题；评估事件未参与训练，训练只保存适配器。')
