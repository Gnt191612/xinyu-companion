"""否定作用域与发言者对照；冻结数据不覆盖。"""
from pathlib import Path
import json,random
R=Path(__file__).resolve().parent;D=R/'数据/companion_v24_turn';assert not D.exists();D.mkdir()
old=[json.loads(s) for s in (R/'数据/companion_v24_opinion/train.jsonl').read_text(encoding='utf8').splitlines()]
topics=[('友谊','我觉得友谊里能坦诚表达，也愿意尊重彼此的不同，这样的相处很珍贵呀。'),('散步','我觉得散步能让节奏慢下来，沿路看看树和天空，不用赶着完成什么呀。'),('下雨','我觉得雨声有安静的美，不过出门淋湿也确实麻烦呀。'),('看书','我觉得读书的乐趣是遇到新的想法，不一定非要一口气看完呀。'),('旅行','我觉得旅行不只是去景点，也包括路上偶然遇到的小事呀。'),('音乐','我觉得合适的音乐能陪着不同心情，一首歌有时会让表达轻松一点呀。')]
asks=['不是让我说，我想先听你的看法。','我不是让你听我，是让你说给我听。','你误会啦，这次是你来讲，我负责听。','不要继续让我讲，先把你的意见告诉我。','别把问题又问回来，我问的是你的观点。','刚才说的是我听你讲，不是你听我讲。','换你发言，先聊你怎么看这件事。','我愿意听你说，没让你继续听我。']
rows=[]
def add(h,a,c):rows.append({'id':f'turn_train_{len(rows):04}','history':h,'response_text':a,'emotion_label':'neutral','category':c})
for t,a in topics:
 for i,q in enumerate(asks):
  h=[{'role':'user','content':f'聊聊{t}吧。'},{'role':'assistant','content':['你说吧，我听着。','你愿意把自己的想法告诉我吗？'][i%2]},{'role':'user','content':q}];add(h,a,'否定纠正模型发言')
 for q in ['不是要你发表意见，我想自己说，你先听着。','我没有问你的看法，先让我讲完关于这件事的想法。','别抢着说，这会儿我想讲，你听我说就好。','现在不用你来讲，换我说，你听着。']:
  add([{'role':'user','content':f'我想聊聊{t}。'},{'role':'assistant','content':a},{'role':'user','content':q}],'好呀，你慢慢说，我先听着。','否定纠正用户发言')
# 固定均衡回放：24观点、12用户倾听/角色、12亲密；不取旧最终题。
replay=[x for x in old if x.get('category')=='模型发表观点'][:24]+[x for x in old if x.get('category') in ['用户发言对照','倾听历史后切换发言']][:12]+[x for x in old if '亲密' in x.get('category','')][:12]
for x in replay:rows.append({**x,'id':f'turn_replay_{len(rows):04}'})
dev=[]
for t,a in [('做饭','我觉得做饭有创作的乐趣，也有日常辛苦，简单做一道喜欢的菜就很好呀。'),('养花','我觉得养花能看到慢慢积累的变化，但也需要耐心照顾呀。'),('写日记','我觉得写日记能留下当时的心情，不必要求每一页都漂亮呀。')]:
 for q in ['你刚才又让我说了，可我需要的是听你表达意见。','听的那个人是我，负责说的人是你，请讲讲你的判断。','先别问我啦，关于刚才的主题你持什么看法？','不是我来说哦，你先把想法讲出来。']:
  dev.append({'id':f'turn_dev_{len(dev):03}','history':[{'role':'user','content':f'聊聊{t}吧。'},{'role':'assistant','content':'你说吧，我在听。'},{'role':'user','content':q}],'response_text':a,'emotion_label':'neutral'})
for n,x in [('train',rows),('dev',dev)]: (D/(n+'.jsonl')).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in x),encoding='utf8')
(D/'style.txt').write_text((R/'体验/陪伴提示.txt').read_text(encoding='utf8'),encoding='utf8')
assert not ({json.dumps(x['history'],ensure_ascii=False) for x in rows}&{json.dumps(x['history'],ensure_ascii=False) for x in dev})
print('训练',len(rows),'开发',len(dev),'回放',len(replay))
