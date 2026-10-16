import sys, json, numpy as np
sys.path.insert(0, "/home/cai/P2&DX_project/P2&DX/USD/scripts")
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J=json.load(open('/home/cai/P2&DX_project/P2&DX/USD/config/dx_hub_no_obj_v1_plan.json'))
xyz,rec=read_las(LAS); P=apply(np.array(J['frame']['T_las_to_world']),xyz).astype(np.float32); I=np.asarray(rec['I']).astype(np.float32); del xyz
cz=np.load(WORK/'ceil_fine.npy'); C=0.05; LO=np.array([-14.,-23.])
i=((P[:,0]-LO[0])/C).astype(int); j=((P[:,1]-LO[1])/C).astype(int)
ok=(i>=0)&(i<cz.shape[0])&(j>=0)&(j<cz.shape[1])&(P[:,2]>2.2)
i,j,P,I=i[ok],j[ok],P[ok],I[ok]
c=cz[i,j]; near=np.isfinite(c)&(np.abs(P[:,2]-c)<0.05)
G=0.025; gi=((P[near,0]-LO[0])/G).astype(int); gj=((P[near,1]-LO[1])/G).astype(int)
S=np.zeros((1120,1840)); N=np.zeros((1120,1840)); np.add.at(S,(gi,gj),I[near]); np.add.at(N,(gi,gj),1)
Imap=np.where(N>0,S/np.maximum(N,1),np.nan); np.save(WORK/'ceil_intensity.npy',Imap)
print('intensity pct', np.nanpercentile(Imap,[5,25,50,75,95,99]))
# splat brightness near the ceiling
_,props,gs=read_gs_ply(GSPLAT); cc={p:k for k,p in enumerate(props)}
Gw=apply(np.array(J['frame']['T_gsplat_to_world']),gs[:,[cc['x'],cc['y'],cc['z']]].astype(np.float64))
op=1/(1+np.exp(-gs[:,cc['opacity']])); rgb=np.clip(0.5+0.2821*gs[:,[cc['f_dc_0'],cc['f_dc_1'],cc['f_dc_2']]],0,3)
k1=((Gw[:,0]-LO[0])/C).astype(int); k2=((Gw[:,1]-LO[1])/C).astype(int)
okg=(k1>=0)&(k1<cz.shape[0])&(k2>=0)&(k2<cz.shape[1])&(op>0.3)
c2=np.full(len(Gw),np.nan); c2[okg]=cz[k1[okg],k2[okg]]
sel=okg&np.isfinite(c2)&(np.abs(Gw[:,2]-c2)<0.12)
b=rgb[sel].mean(1)
g1=((Gw[sel,0]-LO[0])/G).astype(int); g2=((Gw[sel,1]-LO[1])/G).astype(int)
S2=np.zeros((1120,1840)); N2=np.zeros((1120,1840)); np.add.at(S2,(g1,g2),b*op[sel]); np.add.at(N2,(g1,g2),op[sel])
Bmap=np.where(N2>0,S2/np.maximum(N2,1e-6),np.nan); np.save(WORK/'ceil_splat_bright.npy',Bmap)
fig,ax=plt.subplots(1,2,figsize=(30,24))
ax[0].imshow(Imap.T,origin='lower',extent=[-14,14,-23,23],cmap='gray',vmin=np.nanpercentile(Imap,2),vmax=np.nanpercentile(Imap,99.5)); ax[0].set_title('LiDAR intensity at ceiling')
ax[1].imshow(Bmap.T,origin='lower',extent=[-14,14,-23,23],cmap='gray',vmin=0.3,vmax=1.6); ax[1].set_title('splat brightness at ceiling')
for a in ax: a.grid(alpha=.25); a.set_xticks(range(-14,15,2)); a.set_yticks(range(-23,24,2))
plt.tight_layout(); plt.savefig('ceilmap.png',dpi=40)
