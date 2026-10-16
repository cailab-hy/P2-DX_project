import sys, json, numpy as np
sys.path.insert(0, "/home/cai/P2&DX_project/P2&DX/USD/scripts")
from common import *
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m=json.load(open(WORK/'occ3d.json')); C=m['C']; lo=np.array(m['lo'])
ceil=np.load(WORK/'ceil_raw.npy'); nx,ny=ceil.shape
fr=json.load(open(WORK/'frame.json'))
xyz,_=read_las(LAS); P=apply(np.array(fr['T_las_to_world']),xyz).astype(np.float32); del xyz
P=P[P[:,2]>2.2]
i=np.floor((P[:,0]-lo[0])/C).astype(int); j=np.floor((P[:,1]-lo[1])/C).astype(int)
ok=(i>=0)&(i<nx)&(j>=0)&(j<ny); P,i,j=P[ok],i[ok],j[ok]
c=ceil[i,j]; sel=np.isfinite(c)&(np.abs(P[:,2]-c)<0.06)
idx=i[sel]*ny+j[sel]
s=np.bincount(idx,P[sel,2],nx*ny); n=np.bincount(idx,minlength=nx*ny)
cz=np.where(n>=3, s/np.maximum(n,1), np.nan).reshape(nx,ny)
np.save(WORK/'ceil_fine.npy',cz)
v=cz[np.isfinite(cz)]
h,e=np.histogram(v,bins=np.arange(2.3,2.85,0.005))
hs=ndimage.gaussian_filter1d(h.astype(float),1.5)
pk=[k for k in range(1,len(hs)-1) if hs[k]>=hs[k-1] and hs[k]>=hs[k+1] and hs[k]>hs.max()*0.03]
print('peaks', [(round(e[k]+0.0025,3), int(hs[k])) for k in pk])
fig,ax=plt.subplots(figsize=(14,20)); a=ax.imshow(cz.T,origin='lower',vmin=2.4,vmax=2.78,cmap='turbo',extent=[lo[0],lo[0]+nx*C,lo[1],lo[1]+ny*C]); plt.colorbar(a); ax.grid(alpha=.3); plt.savefig('ceil_fine.png',dpi=45)
