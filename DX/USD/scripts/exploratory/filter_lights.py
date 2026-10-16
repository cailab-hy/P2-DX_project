import numpy as np, json
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='work/'; I=np.load(W+'ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.])
cz=np.load(W+'ceil_fine.npy')          # 5 cm
A=json.load(open(W+'light_cands2.json'))
valid=np.isfinite(I)
keep=[]
for a in A:
    x0,x1,y0,y1=a['x']-a['sx']/2,a['x']+a['sx']/2,a['y']-a['sy']/2,a['y']+a['sy']/2
    m=0.2
    i0,i1=int((x0-m-LO[0])/G),int((x1+m-LO[0])/G); j0,j1=int((y0-m-LO[1])/G),int((y1+m-LO[1])/G)
    if i0<0 or j0<0 or i1>=I.shape[0] or j1>=I.shape[1]: continue
    vf=valid[i0:i1,j0:j1].mean()
    k0,k1=int((x0-m-LO[0])/0.05),int((x1+m-LO[0])/0.05); l0,l1=int((y0-m-LO[1])/0.05),int((y1+m-LO[1])/0.05)
    hz=cz[k0:k1,l0:l1].copy()
    ik0,ik1=int((x0-LO[0])/0.05),int((x1-LO[0])/0.05)+1; il0,il1=int((y0-LO[1])/0.05),int((y1-LO[1])/0.05)+1
    hz[ik0-k0:ik1-k0, il0-l0:il1-l0]=np.nan                     # ring only
    hz=hz[np.isfinite(hz)]
    hstd=float(np.percentile(hz,90)-np.percentile(hz,10)) if hz.size>20 else 9
    ii0,ii1=int((x0+0.04-LO[0])/G),int((x1-0.04-LO[0])/G); jj0,jj1=int((y0+0.04-LO[1])/G),int((y1-0.04-LO[1])/G)
    inn=I[ii0:ii1,jj0:jj1]; inn=inn[np.isfinite(inn)]
    cv=float(np.std(inn)/max(np.mean(inn),1)) if inn.size>5 else 9
    a.update(vf=round(vf,3),hspan=round(hstd,3),cv=round(cv,3))
    keep.append(a)
json.dump(keep,open(W+'light_cands3.json','w'),indent=1)
sc=np.array([a['score'] for a in keep]); vf=np.array([a['vf'] for a in keep]); hs=np.array([a['hspan'] for a in keep]); cv=np.array([a['cv'] for a in keep])
print('hspan pct',np.percentile(hs,[10,25,50,75,90]).round(3)); print('cv pct',np.percentile(cv,[10,25,50,75,90]).round(2)); print('vf pct',np.percentile(vf,[10,25,50,75]).round(2))
good=(vf>0.95)&(hs<0.035)&(cv<0.40)&(sc>0.33)
print('kept',good.sum(),'of',len(keep))
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for a,g in zip(keep,good):
    ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec=('lime' if g else 'r'),lw=1.3 if g else 0.6))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det3.png',dpi=40,bbox_inches='tight')
