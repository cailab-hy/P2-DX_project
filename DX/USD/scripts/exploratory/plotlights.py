import numpy as np, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
I=np.load('work/ceil_intensity.npy'); L=json.load(open('/home/cai/P2&DX_project/P2&DX/USD/config/dx_hub_no_obj_v1_ceiling_lights.json'))
prev=json.load(open('work/lights_v3b.json'))['panels'] if len(sys.argv)>1 else []
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(np.isfinite(I),I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
col={'300x1200':'lime','300x900':'magenta','300x600':'yellow','600x600':'cyan','600x1200':'blue'}
for p in prev: ax.add_patch(plt.Rectangle((p['x']-p['sx']/2,p['y']-p['sy']/2),p['sx'],p['sy'],fill=False,ec='red',lw=0.6,ls='--'))
for p in L['panels']: ax.add_patch(plt.Rectangle((p['x']-p['sx']/2,p['y']-p['sy']/2),p['sx'],p['sy'],fill=False,ec=col.get(p['size_mm'],'white'),lw=1.4))
for h in L['hvac']: ax.add_patch(plt.Rectangle((h['x']-0.42,h['y']-0.42),0.84,0.84,fill=False,ec='orange',lw=1.4))
for d in L['downlights']: ax.add_patch(plt.Circle((d['x'],d['y']),0.12,fill=False,ec='red',lw=1.2))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24))
plt.savefig('/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1/validation/ceiling_lights_plan.png',dpi=40,bbox_inches='tight')
