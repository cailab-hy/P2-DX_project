import numpy as np, cv2, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='work/'; I=np.load(W+'ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.])
valid=np.isfinite(I); If=np.where(valid,I,np.nanmedian(I))
If=ndimage.median_filter(If,size=3)
bg=ndimage.median_filter(If,size=49)                      # 1.2 m local background
rel=np.where(valid,If/np.maximum(bg,1),1)
cand=(rel<0.62)&(bg>70)&valid                              # dark on bright tile ceiling
cand=ndimage.binary_opening(ndimage.binary_closing(cand,iterations=1),iterations=1)
lab,n=ndimage.label(cand); out=[]
for k,sl in enumerate(ndimage.find_objects(lab),1):
    m=(lab[sl]==k).astype(np.uint8); a=m.sum()*G*G
    if a<0.05 or a>1.2: continue
    cnts,_=cv2.findContours(m.T.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    (cx,cy),(w,h),ang=cv2.minAreaRect(max(cnts,key=len))
    w,h=(w+1)*G,(h+1)*G; fill=a/(w*h)
    if fill<0.6: continue
    x=LO[0]+(sl[0].start+cx)*G; y=LO[1]+(sl[1].start+cy)*G
    s,l=sorted([w,h]); ang_long=ang if w>=h else ang+90
    out.append(dict(x=round(float(x),3),y=round(float(y),3),short=round(float(s),3),long=round(float(l),3),ang=round(float(ang_long)%180,1),fill=round(float(fill),2)))
print(len(out))
sh=np.array([o['short'] for o in out]); lg=np.array([o['long'] for o in out])
for o in sorted(out,key=lambda o:(o['short'],o['long'])): pass
import collections
print('short pct',np.percentile(sh,[10,50,90]).round(2),'long pct',np.percentile(lg,[10,50,90]).round(2))
h2,_,_=np.histogram2d(sh,lg,bins=[np.arange(0.1,1.0,0.1),np.arange(0.2,1.8,0.2)]); print(h2.astype(int))
json.dump(out,open(W+'light_cands.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(rel.T,origin='lower',extent=[-14,14,-23,23],cmap='gray',vmin=0.3,vmax=1.2)
for o in out:
    c='r' if o['long']/o['short']>1.8 else ('c' if o['short']>0.7 else 'y')
    r=cv2.boxPoints(((o['x'],o['y']),(o['long'],o['short']),o['ang'])); ax.add_patch(plt.Polygon(r,fill=False,ec=c,lw=1.2))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det.png',dpi=40,bbox_inches='tight')
