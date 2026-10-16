import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
src=open('detect_lights7.py').read()
exec(src.split("pool=json.load")[0])          # sides(), ring_flat, I, LO, G, valid
D=json.load(open(W+'lights_v3.json')); P=D['panels']; H=D['hvac']
def ov(a,b):
    w=max(0,min(a['x']+a['sx']/2,b['x']+b['sx']/2)-max(a['x']-a['sx']/2,b['x']-b['sx']/2)); h=max(0,min(a['y']+a['sy']/2,b['y']+b['sy']/2)-max(a['y']-a['sy']/2,b['y']-b['sy']/2))
    return w*h/min(a['sx']*a['sy'],b['sx']*b['sy'])
new=[]
for it in range(1):
    seeds=P+new
    for a in seeds:
        for along in ('x','y'):
            for t in np.arange(-14,14.01 if along=='x' else 23.01,0.025) if along=='x' else np.arange(-23,23.01,0.025):
                pass
    # efficient: candidate positions along lines through each seed, step 2.5 cm, then local maxima
    cands=[]
    for a in seeds:
        for along in ('x','y'):
            rngv=np.arange(-14,14,0.025) if along=='x' else np.arange(-23,23,0.025)
            vals=[]
            for t in rngv:
                c=dict(a); c[along]=float(t)
                if abs(t-a[along])<0.3: vals.append(-1); continue
                hf,f=ring_flat(c['x']-c['sx']/2,c['x']+c['sx']/2,c['y']-c['sy']/2,c['y']+c['sy']/2)
                if hf>0.04 or f<0.85: vals.append(-1); continue
                s=sides(c)
                if np.isnan(s).sum()>1: vals.append(-1); continue
                sg=np.sign(np.nanmedian(s)); good=np.nansum((s*sg)>0.15)
                vals.append(float(np.nanmedian(np.abs(s))) if (good>=4 or (good>=3 and np.isnan(s).sum()==1)) else -1)
            vals=np.array(vals)
            for k in range(1,len(vals)-1):
                if vals[k]>0.25 and vals[k]>=vals[k-1] and vals[k]>=vals[k+1]:
                    c=dict(a); c[along]=round(float(rngv[k]),3); c['src']='rowcol'; c['edge']=vals[k]; cands.append(c)
    cands.sort(key=lambda c:-c['edge'])
    added=0
    for c in cands:
        if all(ov(c,b)<0.2 for b in P+new+H):
            new.append(c); added+=1
    print('iteration',it,'added',added)
    if added==0: break
allp=P+new
import collections; print(len(allp),collections.Counter(a['kind'] for a in allp))
json.dump({'panels':allp,'hvac':H},open(W+'lights_v3b.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for a in allp: ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec='lime' if a['src']!='rowcol' else 'magenta',lw=1.3))
for h in H: ax.add_patch(plt.Rectangle((h['x']-0.42,h['y']-0.42),0.84,0.84,fill=False,ec='orange',lw=1.3))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det8.png',dpi=40,bbox_inches='tight')
