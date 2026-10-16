import numpy as np, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='work/'; I=np.load(W+'ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.]); cz=np.load(W+'ceil_fine.npy')
valid=np.isfinite(I).astype(float); If=np.where(valid>0,I,0.0)
# ceiling flatness map (5 cm): local 10-90 range of the height in a 0.9 m window, upsampled
czf=np.where(np.isfinite(cz),cz,np.nan)
def rng(a):
    a=a[np.isfinite(a)]; return np.percentile(a,90)-np.percentile(a,10) if a.size>10 else 9
flat=ndimage.generic_filter(np.nan_to_num(czf,nan=-9),lambda v: rng(np.where(v<-5,np.nan,v)),size=19)
flat=np.kron(flat,np.ones((2,2)))[:I.shape[0],:I.shape[1]]
def bmean(a,wx,wy): return ndimage.uniform_filter(a,size=(wx,wy),mode='constant')
SIZES={'panel_300x1200':(0.3,1.2),'panel_600x600':(0.6,0.6),'hvac_840':(0.84,0.84),'panel_600x1200':(0.6,1.2),'panel_300x900':(0.3,0.9)}
M=int(0.125/G); C=[]
for name,(s,l) in SIZES.items():
    for (sx,sy) in ({(s,l),(l,s)}):
        wx,wy=int(round(sx/G)),int(round(sy/G))
        si=bmean(If,wx,wy)*wx*wy; ni=bmean(valid,wx,wy)*wx*wy
        so=bmean(If,wx+2*M,wy+2*M)*(wx+2*M)*(wy+2*M); no=bmean(valid,wx+2*M,wy+2*M)*(wx+2*M)*(wy+2*M)
        mi=si/np.maximum(ni,1); ring=(so-si)/np.maximum(no-ni,1)
        cov_i=ni/(wx*wy); cov_r=(no-ni)/((wx+2*M)*(wy+2*M)-wx*wy)
        # edge sharpness: contrast between inner and a ring 1 cell inside, both signs
        con=np.abs(1-mi/np.maximum(ring,1))
        sc=np.where((cov_i>0.9)&(cov_r>0.9)&(flat<0.05)&(ring>20),con,0)
        mx=ndimage.maximum_filter(sc,size=(wx+M,wy+M))
        for i,j in np.argwhere((sc==mx)&(sc>0.22)):
            C.append(dict(score=float(sc[i,j]),kind=name,x=LO[0]+(i+0.5)*G,y=LO[1]+(j+0.5)*G,sx=sx,sy=sy,dark=bool(mi[i,j]<ring[i,j])))
C.sort(key=lambda c:-c['score'])
def ov(a,b):
    w=max(0,min(a['x']+a['sx']/2,b['x']+b['sx']/2)-max(a['x']-a['sx']/2,b['x']-b['sx']/2)); h=max(0,min(a['y']+a['sy']/2,b['y']+b['sy']/2)-max(a['y']-a['sy']/2,b['y']-b['sy']/2))
    return w*h/min(a['sx']*a['sy'],b['sx']*b['sy'])
acc=[]
for c in C:
    if all(ov(c,a)<0.25 for a in acc): acc.append(c)
import collections; print(len(acc),collections.Counter(a['kind'] for a in acc))
json.dump(acc,open(W+'light_cands5.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid>0,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
col={'panel_300x1200':'lime','panel_300x900':'magenta','panel_600x600':'yellow','hvac_840':'orange','panel_600x1200':'cyan'}
for a in acc:
    ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec=col[a['kind']],lw=1.3)); ax.text(a['x']-a['sx']/2,a['y']+a['sy']/2,f"{a['score']:.2f}",fontsize=5,color='red')
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det5.png',dpi=40,bbox_inches='tight')
