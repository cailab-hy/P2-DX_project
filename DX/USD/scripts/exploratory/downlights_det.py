import numpy as np, json, cv2
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
I=np.load('work/ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.]); valid=np.isfinite(I)
If=ndimage.median_filter(np.where(valid,I,np.nanmedian(I)),size=2)
bg=ndimage.median_filter(If,size=17)                         # 0.4 m
rel=np.where(valid,If/np.maximum(bg,5),1)
out=[]
for pol,mask in (('dark',rel<0.6),('bright',rel>1.6)):
    lab,n=ndimage.label(mask&valid)
    for k,sl in enumerate(ndimage.find_objects(lab),1):
        mk=lab[sl]==k; a=mk.sum()*G*G
        if not (0.006<=a<=0.05): continue
        h,w=mk.shape; d=max(h,w)*G
        if not (0.08<=d<=0.26) or min(h,w)/max(h,w)<0.6 or mk.sum()/(h*w)<0.55: continue
        ij=np.argwhere(mk).mean(0)
        x,y=LO[0]+(sl[0].start+ij[0]+0.5)*G, LO[1]+(sl[1].start+ij[1]+0.5)*G
        # local surround must be fully observed (not a hole at a wall edge)
        i0,j0=int((x-0.3-LO[0])/G),int((y-0.3-LO[1])/G)
        if valid[i0:i0+24,j0:j0+24].mean()<0.97: continue
        out.append(dict(x=round(float(x),3),y=round(float(y),3),d=round(float(d),3),pol=pol,bg=round(float(bg[int((x-LO[0])/G),int((y-LO[1])/G)]),1)))
print(len(out)); json.dump(out,open('work/downlight_cands.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for o in out: ax.add_patch(plt.Circle((o['x'],o['y']),0.15,fill=False,ec='red' if o['pol']=='dark' else 'cyan',lw=1))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('downlights_det.png',dpi=40,bbox_inches='tight')
