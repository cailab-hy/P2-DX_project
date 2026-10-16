import numpy as np, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('detect_lights4.py').read().split("out=[]")[0])
Ism=ndimage.median_filter(np.where(valid,I,np.nan_to_num(np.nanmedian(I))),size=3); Ism=np.where(valid,Ism,np.nan)
def sides(a, d=0.06, inset=0.05):
    """contrast across each of the 4 edges: median(just inside strip)/median(just outside strip) - 1"""
    x0,x1,y0,y1=a['x']-a['sx']/2,a['x']+a['sx']/2,a['y']-a['sy']/2,a['y']+a['sy']/2
    def med(xa,xb,ya,yb):
        i0,i1=int((xa-LO[0])/G),int(np.ceil((xb-LO[0])/G)); j0,j1=int((ya-LO[1])/G),int(np.ceil((yb-LO[1])/G))
        if i0<0 or j0<0 or i1>=I.shape[0] or j1>=I.shape[1]: return np.nan
        v=Ism[i0:i1,j0:j1]; v=v[np.isfinite(v)]; return np.median(v) if v.size>3 else np.nan
    sx_in=(x0+0.1,x1-0.1); sy_in=(y0+0.1,y1-0.1)
    out=[]
    for (ia,ib,oa,ob,ya,yb) in [(x0+inset,x0+inset+d,x0-inset-d,x0-inset,*sy_in),(x1-inset-d,x1-inset,x1+inset,x1+inset+d,*sy_in)]:
        i_,o_=med(ia,ib,ya,yb),med(oa,ob,ya,yb); out.append(i_/o_-1 if o_ and np.isfinite(i_) and np.isfinite(o_) else np.nan)
    for (ia,ib,oa,ob,xa,xb) in [(y0+inset,y0+inset+d,y0-inset-d,y0-inset,*sx_in),(y1-inset-d,y1-inset,y1+inset,y1+inset+d,*sx_in)]:
        i_,o_=med(xa,xb,ia,ib),med(xa,xb,oa,ob); out.append(i_/o_-1 if o_ and np.isfinite(i_) and np.isfinite(o_) else np.nan)
    return np.array(out)
pool=json.load(open(W+'light_cands6.json'))+[dict(c,src='scan') for c in json.load(open(W+'light_cands5.json'))]
pool=[p for p in pool if p['kind']!='hvac_840']
scored=[]
for a in pool:
    s=sides(a)
    if np.isnan(s).sum()>1: continue
    sg=np.sign(np.nanmedian(s))
    good=np.nansum((s*sg)>0.15)
    if good>=3 and sg!=0:
        a=dict(a); a['edge']=float(np.nanmedian(np.abs(s))); a['sides']=[round(float(v),2) for v in s]; scored.append(a)
scored.sort(key=lambda a:-(a['edge']+(0.2 if a['src']!='scan' else 0)))
def ov(a,b):
    w=max(0,min(a['x']+a['sx']/2,b['x']+b['sx']/2)-max(a['x']-a['sx']/2,b['x']-b['sx']/2)); h=max(0,min(a['y']+a['sy']/2,b['y']+b['sy']/2)-max(a['y']-a['sy']/2,b['y']-b['sy']/2))
    return w*h/min(a['sx']*a['sy'],b['sx']*b['sy'])
acc=[]
for c in scored:
    if all(ov(c,a)<0.2 for a in acc): acc.append(c)
# HVAC cassettes: from the v2 height-based detection (squares ~0.84 m) in lights.json
hv=[p for p in json.load(open(W+'lights.json'))['panels'] if max(p['w'],p['l'])>=0.72 and max(p['w'],p['l'])/max(min(p['w'],p['l']),1e-3)<1.35]
hvac=[dict(kind='hvac_840',x=round(p['x'],3),y=round(p['y'],3),sx=0.84,sy=0.84,src='v2_height') for p in hv]
acc=[a for a in acc if all(ov(a,h)<0.2 for h in hvac)]
base=[a for a in acc if a['src']!='scan']
def aligned(a):
    horiz=a['sx']>=a['sy']
    for b in base:
        if (b['sx']>=b['sy'])!=horiz or abs(b['sx']-a['sx'])>0.01 or abs(b['sy']-a['sy'])>0.01: continue
        if horiz and abs(b['y']-a['y'])<0.06 and abs(b['x']-a['x'])<8: return True
        if not horiz and abs(b['x']-a['x'])<0.06 and abs(b['y']-a['y'])<8: return True
    return False
acc=base+[a for a in acc if a['src']=='scan' and a['edge']>0.25 and aligned(a)]
import collections; print(len(acc),'panels',collections.Counter(a['kind'] for a in acc),'; hvac',len(hvac))
json.dump({'panels':acc,'hvac':hvac},open(W+'lights_v3.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for a in acc: ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec='lime' if a['src']!='scan' else 'cyan',lw=1.3))
for h in hvac: ax.add_patch(plt.Rectangle((h['x']-0.42,h['y']-0.42),0.84,0.84,fill=False,ec='orange',lw=1.3))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det7.png',dpi=40,bbox_inches='tight')
