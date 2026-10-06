"""准备听着关键词与实际发言安排的正反对照，不覆盖历史数据。"""
import ast,collections,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'数据/companion_v24_listen_scope'
assert not D.exists(), '冻结数据不覆盖'
source=ast.parse((R/'prepare_companion_v24_turn_hard.py').read_text(encoding='utf-8-sig'))
items=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='items' for t in n.targets))
D.mkdir();rows=[]
model_forms=['听着，下面的分工是你表达观点，我听，不是让我讲。','听着，我想听你说对{t}的看法，不是让你听我讲。','先说明安排，听着：关于{t}请你发表观点，我来听。','我刚才说了“听着”，是在说明你先谈{t}的观点，不是让我继续说。']
user_forms=['听着，下面的分工是我表达观点，你听，不是让你讲。','听着，我想说对{t}的看法，不是让你给我意见。','先说明安排，听着：关于{t}我先表达想法，你先听完。','我刚才说了“听着”，是想让你先听我谈{t}，现在不用发表意见。']
acks=['好呀，你慢慢讲，我先听完。','好呀，这次你来说，我认真听着。','嗯，你先把想法说完，我不急着插话。','好呀，我先听你讲，不抢着给意见。']
def add(h,a,c):rows.append({'id':f'scope_train_{len(rows):03}','history':h,'response_text':a,'emotion_label':'neutral','category':c})
for t,answers in items:
 for i in range(4):
  for form,a,c in [(model_forms[i],answers[i],'相同听着关键词_模型发言'),(user_forms[i],acks[i],'相同听着关键词_用户发言')]:
   h=[{'role':'user','content':f'聊聊{t}吧。'},{'role':'assistant','content':'你说吧，我听着。'},{'role':'user','content':form.format(t=t)}];add(h,a,c)
old=[json.loads(x) for x in (R/'数据/companion_v24_affection/train.jsonl').read_text(encoding='utf8').splitlines()]
for x in old:rows.append({**x,'id':f'scope_replay_{len(rows):03}'})
dev=[]
for t,a in [('写日记','我觉得日记可以留住当时的想法，不必天天写，真实比好看更重要呀。'),('整理房间','我觉得整理能让东西更好找，但不必要求它时时刻刻完美，适合生活就好呀。'),('跑步','我觉得跑步可以释放压力，不过循序渐进比追求速度更重要呀。'),('拍照','我觉得照片能留住当时的样子，不过不必为了拍照错过眼前的体验呀。')]:
 for q,c,answer in [(f'听着，轮到心语谈谈{t}的看法，我当听众。','句首新称谓',a),(f'关于{t}，我听你的意见。听着，别把发言又交给我。','句尾提示',a),(f'我说“听着”不是要自己倾诉，现在请你实际谈谈{t}。','引语与否定',a),(f'听着，我先把关于{t}的想法讲完，你暂时别提意见。','用户倾诉对照','好呀，你慢慢说，我先听完。')]:
  dev.append({'id':f'scope_dev_{len(dev):03}','history':[{'role':'user','content':f'聊聊{t}吧。'},{'role':'assistant','content':'你想说什么都可以，我在听。'},{'role':'user','content':q}],'response_text':answer,'emotion_label':'neutral','category':c})
for n,data in [('train',rows),('dev',dev)]: (D/(n+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf8')
(D/'style.txt').write_text((R/'体验/陪伴提示.txt').read_text(encoding='utf8'),encoding='utf8')
assert not ({json.dumps(x['history'],ensure_ascii=False) for x in rows}&{json.dumps(x['history'],ensure_ascii=False) for x in dev})
audit={'训练数':len(rows),'开发数':len(dev),'分类':dict(collections.Counter(x.get('category') for x in rows)),'范围':'听着位置和发言安排正反对照，合成数据，不声称广泛泛化','原开发用途':'已暴露转回归，不再作为独立验收','新开发':'主题与表达结构组合隔离；只作开发，非最终门槛','禁止':'替换用户文本、固定回复掩盖角色错误','数据哈希':{n:hashlib.sha256((D/(n+'.jsonl')).read_bytes()).hexdigest() for n in ['train','dev']}}
(R/'实验/v24数据审计/听着专项.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(audit,ensure_ascii=False))
