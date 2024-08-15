

 
with open('/mnt/cfs/CV/lj/code/asr_project/tal_frontend/press/tts/text_1w.txt') as f:
    text = f.readlines()
    
lines = []

for t in text:
    txt = t.split('	')[1].strip().replace(' ','')
    lines.append(txt)
    
print(len(lines))

with open('/mnt/cfs/CV/lj/code/asr_project/tal_frontend/press/tts/text_1w_new.txt','w') as f:
    for l in lines:
        f.write(l+'\n')