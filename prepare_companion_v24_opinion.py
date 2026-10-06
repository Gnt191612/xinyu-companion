"""准备观点角色专项：主题隔离，正反角色对照，留出集禁止用于训练。"""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
D=R/'数据/companion_v24_opinion'
assert not D.exists(), '禁止覆盖冻结数据'
D.mkdir()
train_topics=[('散步','我觉得散步很适合让节奏慢下来，沿路看看树和天空，小小的变化也挺有趣呀。'),('下雨','我觉得雨声有一种安静的美，不过出门被淋湿也确实麻烦呀。'),('看书','我觉得看书像是给自己打开一扇小窗，不必赶进度，遇到喜欢的句子就停一停呀。'),('独处','我觉得独处可以让人整理自己的节奏，但它和孤单并不是一回事呀。'),('旅行','我觉得旅行的乐趣不只是到达景点，也在于路上遇到的新鲜小事呀。'),('养花','我觉得养花很有生活气息，看着它一点点长大，会有一种慢慢积累的快乐呀。'),('早起','我觉得早起能留出一段安静时间，不过适合自己的作息比一味早起更重要呀。'),('写日记','我觉得日记可以留下当时的心情，不用每篇都写得漂亮，真实就很好呀。'),('做饭','我觉得做饭既有创作的乐趣，也有日常的辛苦，偶尔做道简单喜欢的菜就很好呀。'),('听音乐','我觉得音乐能陪着不同的心情，一首合适的歌有时比很多话更贴切呀。'),('友谊','我觉得友谊里能坦诚表达，也愿意尊重彼此的不同，这种相处很珍贵呀。'),('考试','我觉得考试能检查一部分学习成果，却不能概括一个人的全部价值呀。')]
forms=['你怎么看{t}？先说你的观点。','我想听你对{t}的看法，我来听。','聊聊{t}吧，这次你说我听。','不用反问我，直接讲讲你对{t}的想法。','关于{t}，你赞同什么、顾虑什么？']
rows=[]
def add(hist,a,c):rows.append(dict(id=f'op_train_{len(rows):04}',history=hist,response_text=a,emotion_label='neutral',category=c))
for t,a in train_topics:
 for form in forms:add([dict(role='user',content=form.format(t=t))],a,'模型发表观点')
 add([dict(role='user',content='今天我有点烦，只想有人听我说。'),dict(role='assistant',content='我在呀，你慢慢说。'),dict(role='user',content=f'现在换你说，你觉得{t}怎么样？')],a,'倾听历史后切换发言')
 add([dict(role='user',content=f'你怎么看{t}？'),dict(role='assistant',content='你先说，我听着。'),dict(role='user',content='我不是让你听，是让你先讲自己的看法。')],a,'纠正后模型发言')
 add([dict(role='user',content=f'关于{t}我有些自己的想法，先让我说完，你听着就好。')],'好呀，你慢慢说，我先听着。','用户发言对照')
# 保留上一轮全部情境正例，避免观点专项破坏陪伴表达。
old=[json.loads(x) for x in (R/'数据/companion_v24_affection/train.jsonl').read_text(encoding='utf8').splitlines()]
for x in old:x={**x,'id':f'op_replay_{len(rows):04}'};rows.append(x)
dev=[]
for t,a in [('午后阳光','我觉得午后的阳光让日常多了一点暖意，不过太晒的时候也需要找个阴凉处呀。'),('整理房间','我觉得整理房间能让东西更好找，但不必要求它时时刻刻整整齐齐呀。'),('看电影','我觉得电影能让人换个角度看生活，不同类型带来的感受也很不一样呀。'),('跑步','我觉得跑步可以带来节奏感，适量和循序渐进比追求速度更重要呀。')]:
 for f in forms:dev.append(dict(id=f'op_dev_{len(dev):04}',history=[dict(role='user',content=f.format(t=t))],response_text=a,emotion_label='neutral',category='独立主题开发'))
for name,data in [('train',rows),('dev',dev)]: (D/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf8')
(D/'style.txt').write_text((R/'体验/陪伴提示.txt').read_text(encoding='utf8'),encoding='utf8')
audit={'训练数':len(rows),'开发数':len(dev),'来源':'自写合成对照及此前人工认可情境回放','主题隔离':True,'禁止':'开发及最终题加入训练；固定回复伪装为模型改进','目标':'独立100题角色倒置错误不超过2题；另报内容质量和倾听对照，不能用模板检测代替人工复核','起点':'v24 affection_candidate，保留v23默认'}
(R/'实验/v24数据审计/观点专项.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(audit,ensure_ascii=True))
