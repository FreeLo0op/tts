import re
from tal_frontend.frontend.g2p.sym_id_dic import ph2id_dict

def ph2id(oriphs: list[str], *args):
    phs = []
    for i in oriphs:
        phs.extend(i.strip().split())
    #phs = [i.strip().split(' ') for i in phs if i != '#1']
    #1, 8 分别对应sil和sp4，即开始和结束标志
    # ids = [1]
    ids = []
    if args:
        break_lists = args[0]
    break_list = {}
    for i in range(len(phs)):
        ph = phs[i]
        if ph == '#9':
            break_time = break_lists.pop(0)
            break_time = int(break_time.replace('ms', ''))//10
            break_list[i+1] = break_time
            ids.append(7)
        else:
            try:
                ids.append(ph2id_dict[ph])
            except KeyError:
                raise ValueError(f'音素错误，{ph} 不在合法音素集合中')
    # #3结尾直接修改 不增加
    if ids[-1] == 7 or ids[-1] == 8:
        pass
        #ids[-1] = 8
    else:
        ids.append(8)
    return ids
    #ids = [1, *[ph2id_dict[i] for i in phs if i != '#1'], 8]
    
def rhy2id(rhys: list[str]):
    rhy_id_map = {'_': 0, '0': 1, '#0': 2, '#1': 3, '#2': 4, '#3': 5, '#4': 6}
    ids = [1]
    for rhy in rhys:
        try:
            ids.append(rhy_id_map[rhy])
        except KeyError:
            raise ValueError(f'韵律错误，{rhy} 不在合法音素集合中')
    ids.append(1)
    return ids

    