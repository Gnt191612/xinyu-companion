"""只区分明确请求助手讲故事，不改变普通倾听或引用分析。"""
import re


def assistant_story_requested(history):
    if not history or history[-1].get('role') != 'user':
        return False
    text = history[-1]['content'].strip()
    text = re.sub(r'^(?:心语|姐姐)[，,\s]*', '', text)
    if re.search(r'不要(?:给我)?讲|别(?:给我)?讲|不想听|分析.{0,6}(?:台词|句子)|这句话.{0,8}(?:意思|是谁)', text):
        return False
    if re.search(r'听我.{0,6}讲|我(?:来|给你|向你).{0,6}讲', text):
        return False
    requested = bool(re.search(r'(?:给我|为我).{0,8}讲.{0,24}故事|我(?:想|要|希望)听.{0,12}故事|讲.{0,20}故事.{0,5}给我听|(?:请|能不能|可以).{0,5}你.{0,6}讲.{0,16}故事', text))
    requested = requested or bool(re.match(r'^(?:你)?(?:能不能|能否|可不可以|可以|请|麻烦你)?(?:给我|为我)?讲.{0,32}故事', text))
    if requested:
        return True
    if re.match(r'^(?:再讲|再来|换)(?:一个|个).{0,12}', text):
        return any(assistant_story_requested(history[:i+1]) for i in range(len(history)-1)
                   if history[i].get('role') == 'user' and '故事' in history[i].get('content', ''))
    return False


STORY_TASK = '''本轮用户明确要听你讲故事，讲述者是心语，倾听者是用户。不要反问用户愿不愿意讲给你听，不要只说“我听着”或“我给你讲”。直接写一篇150至300字左右的完整虚构故事，有事件、变化和结尾。用户说“咱们两个”时，直接用心语和你作为虚构故事人物，不换成无关的情侣。恋爱故事按虚构创作，不把故事说成真实经历。直接开始讲，不额外询问是不是疲惫或心情不好，不附道理总结。尊重用户指定风格和当下状态，轻松故事避免激烈打斗、疾病治疗和悲伤反转；疲惫不默认晚上或睡觉。续讲或换故事时参考前文，换主要事件。两三句的日常陪伴长度限制不适用于本轮故事。只输出自然的故事回复，不输出JSON、规则或分析。'''
