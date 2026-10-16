import numpy as np, json, cv2
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('detect_lights4.py').read().split("out=[]")[0])          # I, G, LO, cz, valid, ring_flat, STD_S/L
If=ndimage.median_filter(np.where(valid,I,np.nanmedian(I)),size=3)
res=[]
for win in (31,49):
    bg=ndimage.median_filter(If,size=win); rel=np.where(valid,If/np.maximum(bg,5),1)
    for pol,mask in (('dark',rel<0.78),('bright',rel>1.28)):
        m=ndimage.binary_opening(ndimage.binary_closing(mask&valid,iterations=2),iterations=1)
        lab,n=ndimage.label(m)
        for k,sl in enumerate(ndimage.find_objects(lab),1):
            mk=(lab[sl]==k).astype(np.uint8); a=mk.sum()*G*G
            if not (0.06<=a<=1.0): continue
            cnts,_=cv2.findContours(mk.T.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            (cx,cy),(w,h),ang=cv2.minAreaRect(max(cnts,key=len)); w,h=(w+1)*G,(h+1)*G
            off=min(ang%90,90-ang%90)
            if a/(w*h)<0.70 or off>6: continue
            ex,ey=(w,h) if ((ang%180)<45 or (ang%180)>135) else (h,w)
            s,l=sorted([ex,ey])
            sq = l/s<1.35
            if sq and not (0.45<=s<=0.95): continue
            if not sq and not (0.2<=s<=0.7 and 0.5<=l<=1.4): continue
            x=LO[0]+(sl[0].start+cx)*G; y=LO[1]+(sl[1].start+cy)*G
            if sq:
                kind='hvac_840' if s>0.72 else 'panel_600x600'; sx=sy=0.84 if s>0.72 else 0.6
            else:
                ss=min(STD_S,key=lambda v:abs(v-s)); ll=min(STD_L,key=lambda v:abs(v-l)); kind=f"panel_{int(ss*1000)}x{int(ll*1000)}"
                sx,sy=(ll,ss) if ex>=ey else (ss,ll)
            hf,f=ring_flat(x-sx/2,x+sx/2,y-sy/2,y+sy/2)
            if hf>0.04 or f<0.85: continue
            res.append(dict(kind=kind,x=round(float(x),3),y=round(float(y),3),sx=sx,sy=sy,src='seg2',pol=pol,meas=[round(float(ex),2),round(float(ey),2)]))
prev=json.load(open(W+'light_final_cands.json'))
def ov(a,b):
    w=max(0,min(a['x']+a['sx']/2,b['x']+b['sx']/2)-max(a['x']-a['sx']/2,b['x']-b['sx']/2)); h=max(0,min(a['y']+a['sy']/2,b['y']+b['sy']/2)-max(a['y']-a['sy']/2,b['y']-b['sy']/2))
    return w*h/min(a['sx']*a['sy'],b['sx']*b['sy'])
acc=[]
for c in res+prev:
    if all(ov(c,a)<0.3 for a in acc): acc.append(c)
import collections; print(len(res),'seg2;',len(acc),'total',collections.Counter(a['kind'] for a in acc))
json.dump(acc,open(W+'light_cands6.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for i_,a in enumerate(acc):
    c={'seg2':'lime','seg':'lime','tmpl':'cyan','row':'magenta'}[a['src']]
    if a['kind'] in ('hvac_840',): c='orange'
    ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec=c,lw=1.3)); ax.text(a['x'],a['y'],str(i_),fontsize=5,color='red')
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det6.png',dpi=40,bbox_inches='tight')
