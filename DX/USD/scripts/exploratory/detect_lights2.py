import numpy as np, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='work/'; I=np.load(W+'ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.])
valid=np.isfinite(I).astype(float)
If=np.where(valid>0,I,0.0)
def boxmean(a,wx,wy):
    s=ndimage.uniform_filter(a,size=(wx,wy),mode='constant'); n=ndimage.uniform_filter(valid,size=(wx,wy),mode='constant')
    return s/np.maximum(n,1e-6), n
SIZES={'panel_300x1200':(0.3,1.2),'panel_300x900':(0.3,0.9),'panel_600x600':(0.6,0.6),'hvac_840':(0.84,0.84),'panel_600x1200':(0.6,1.2)}
M=0.125
cands=[]
for name,(s,l) in SIZES.items():
    for orient in ((s,l),(l,s)) if s!=l else ((s,l),):
        wx,wy=int(round(orient[0]/G)),int(round(orient[1]/G))
        mi,ni=boxmean(If,wx,wy)
        mo_,no_=boxmean(If,wx+2*int(M/G),wy+2*int(M/G))
        tot_o=mo_*no_*(wx+2*int(M/G))*(wy+2*int(M/G)); tot_i=mi*ni*wx*wy
        ring=(tot_o-tot_i)/np.maximum(no_*(wx+2*int(M/G))*(wy+2*int(M/G))-ni*wx*wy,1)
        # inner uniformity: low std inside
        sc=np.where((ni>0.85)&(no_>0.8)&(ring>60),1-mi/np.maximum(ring,1),0)
        mx=ndimage.maximum_filter(sc,size=(wx,wy))
        pk=np.argwhere((sc==mx)&(sc>0.30))
        for i,j in pk:
            cands.append((float(sc[i,j]),name,LO[0]+(i+0.5)*G,LO[1]+(j+0.5)*G,orient[0],orient[1]))
cands.sort(key=lambda c:-c[0])
# NMS: accept best; reject overlapping (>20% of smaller) later ones
acc=[]
def ov(a,b):
    ax0,ax1,ay0,ay1=a[2]-a[4]/2,a[2]+a[4]/2,a[3]-a[5]/2,a[3]+a[5]/2
    bx0,bx1,by0,by1=b[2]-b[4]/2,b[2]+b[4]/2,b[3]-b[5]/2,b[3]+b[5]/2
    w=max(0,min(ax1,bx1)-max(ax0,bx0)); h=max(0,min(ay1,by1)-max(ay0,by0)); return w*h/min(a[4]*a[5],b[4]*b[5])
for c in cands:
    if all(ov(c,a)<0.2 for a in acc): acc.append(c)
# prefer the larger template when a smaller one sits inside a well-scoring larger one (already NMS by score)
print(len(acc)); import collections; print(collections.Counter(a[1] for a in acc))
json.dump([dict(score=round(a[0],3),kind=a[1],x=round(a[2],3),y=round(a[3],3),sx=a[4],sy=a[5]) for a in acc],open(W+'light_cands2.json','w'),indent=1)
rel=If/np.maximum(ndimage.median_filter(np.where(valid>0,If,np.median(If[valid>0])),size=49),1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid>0,rel,1).T,origin='lower',extent=[-14,14,-23,23],cmap='gray',vmin=0.3,vmax=1.2)
col={'panel_300x1200':'r','panel_300x900':'m','panel_600x600':'y','hvac_840':'c','panel_600x1200':'orange'}
for a in acc:
    ax.add_patch(plt.Rectangle((a[2]-a[4]/2,a[3]-a[5]/2),a[4],a[5],fill=False,ec=col[a[1]],lw=1.3)); ax.text(a[2],a[3]+0.2,f"{a[0]:.2f}",fontsize=5,color=col[a[1]])
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det2.png',dpi=40,bbox_inches='tight')
