import numpy as np, json, cv2
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W='work/'; I=np.load(W+'ceil_intensity.npy'); G=0.025; LO=np.array([-14.,-23.]); cz=np.load(W+'ceil_fine.npy')
valid=np.isfinite(I); If=ndimage.median_filter(np.where(valid,I,np.nanmedian(I)),size=3)
bg=ndimage.median_filter(If,size=49); rel=np.where(valid,If/np.maximum(bg,5),1)
STD_S=[0.3,0.6]; STD_L=[0.6,0.9,1.2]
def ring_flat(x0,x1,y0,y1,m=0.2):
    k0,k1=int((x0-m-LO[0])/0.05),int((x1+m-LO[0])/0.05); l0,l1=int((y0-m-LO[1])/0.05),int((y1+m-LO[1])/0.05)
    if k0<0 or l0<0 or k1>=cz.shape[0] or l1>=cz.shape[1]: return 9,0
    hz=cz[k0:k1,l0:l1].copy(); ik0,ik1=int((x0-LO[0])/0.05),int((x1-LO[0])/0.05)+1; il0,il1=int((y0-LO[1])/0.05),int((y1-LO[1])/0.05)+1
    ring=np.ones(hz.shape,bool); ring[ik0-k0:ik1-k0,il0-l0:il1-l0]=False
    f=np.isfinite(hz[ring]).mean(); hz=hz[ring]; hz=hz[np.isfinite(hz)]
    return (float(np.percentile(hz,90)-np.percentile(hz,10)) if hz.size>20 else 9), f
out=[]
for pol,mask in (('dark',rel<0.80),('bright',rel>1.25)):
    m=ndimage.binary_opening(ndimage.binary_closing(mask&valid,iterations=1),iterations=1)
    lab,n=ndimage.label(m)
    for k,sl in enumerate(ndimage.find_objects(lab),1):
        mk=(lab[sl]==k).astype(np.uint8); a=mk.sum()*G*G
        if not (0.08<=a<=1.0): continue
        cnts,_=cv2.findContours(mk.T.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        (cx,cy),(w,h),ang=cv2.minAreaRect(max(cnts,key=len))
        w,h=(w+1)*G,(h+1)*G
        if a/(w*h)<0.72: continue
        s,l=sorted([w,h]); along_x = (w>=h) == (abs(((ang+45)%90)-45)<10 and (ang%180)<45 or ang%180>135)
        off=min(ang%90,90-ang%90)
        if off>6 or not (0.2<=s<=0.7 and 0.5<=l<=1.4): continue
        x=LO[0]+(sl[0].start+cx)*G; y=LO[1]+(sl[1].start+cy)*G
        # orientation in world: the side with length w lies along image x (= world x since mk was transposed back)
        ex=w if (ang%180)<45 or (ang%180)>135 else h
        ey=h if ex==w else w
        ss=min(STD_S,key=lambda v:abs(v-s)); ll=min(STD_L,key=lambda v:abs(v-l))
        sx,sy=(ll,ss) if ex>=ey else (ss,ll)
        hf,f=ring_flat(x-sx/2,x+sx/2,y-sy/2,y+sy/2)
        if hf>0.035 or f<0.8: continue
        out.append(dict(kind=f"panel_{int(ss*1000)}x{int(ll*1000)}",x=round(float(x),3),y=round(float(y),3),sx=sx,sy=sy,meas=[round(float(ex),2),round(float(ey),2)],pol=pol,src='seg'))
T=[a for a in json.load(open(W+'light_cands3.json')) if a['vf']>0.95 and a['hspan']<0.035 and a['cv']<0.40 and a['score']>0.33]
for a in T: a['src']='tmpl'
allc=out+T
def ov(a,b):
    w=max(0,min(a['x']+a['sx']/2,b['x']+b['sx']/2)-max(a['x']-a['sx']/2,b['x']-b['sx']/2)); h=max(0,min(a['y']+a['sy']/2,b['y']+b['sy']/2)-max(a['y']-a['sy']/2,b['y']-b['sy']/2))
    return w*h/min(a['sx']*a['sy'],b['sx']*b['sy'])
acc=[]
for c in allc:                       # seg first (measured extent), then templates fill gaps
    if all(ov(c,a)<0.3 for a in acc): acc.append(c)

def contrast(x,y,sx,sy,m=0.12):
    """|1 - inner/ring| on the raw intensity, and ring flatness."""
    i0,i1=int((x-sx/2-LO[0])/G),int((x+sx/2-LO[0])/G); j0,j1=int((y-sy/2-LO[1])/G),int((y+sy/2-LO[1])/G)
    mi=int(m/G)
    if i0-mi<0 or j0-mi<0 or i1+mi>=I.shape[0] or j1+mi>=I.shape[1]: return 0,9,0
    outer=I[i0-mi:i1+mi,j0-mi:j1+mi].copy(); inner=I[i0+1:i1-1,j0+1:j1-1]
    outer[mi:mi+(i1-i0),mi:mi+(j1-j0)]=np.nan
    ri=np.nanmedian(inner); ro=np.nanmedian(outer)
    hf,f=ring_flat(x-sx/2,x+sx/2,y-sy/2,y+sy/2)
    return (abs(1-ri/ro) if ro>0 else 0), hf, f
# rows
rows=[]
for a in acc:
    horiz=a['sx']>=a['sy']
    key=(horiz,a['sx'],a['sy'])
    r=next((r for r in rows if r['key']==key and abs((r['c'] if True else 0)-(a['y'] if horiz else a['x']))<0.08),None)
    if r is None: rows.append({'key':key,'c':a['y'] if horiz else a['x'],'m':[a]})
    else: r['m'].append(a)
added=[]
for r in rows:
    if len(r['m'])<2: continue
    horiz,sx,sy=r['key']; ms=sorted(r['m'],key=lambda a:a['x'] if horiz else a['y'])
    pos=[a['x'] if horiz else a['y'] for a in ms]
    gaps=np.diff(pos); pitch=float(np.min(gaps)) if len(gaps) else None
    if not pitch or pitch<(sx if horiz else sy)+0.2: continue
    c=float(np.median([a['y'] if horiz else a['x'] for a in ms]))
    cand=[]
    for p0,p1 in zip(pos[:-1],pos[1:]):                 # fill gaps inside the row
        n=int(round((p1-p0)/pitch))
        cand+= [p0+k*(p1-p0)/n for k in range(1,n)]
    for side in (-1,1):                                   # extend at both ends while contrast holds
        p=(pos[0] if side<0 else pos[-1])+side*pitch
        for _ in range(4):
            cand.append(p); p+=side*pitch
    for p in cand:
        x,y=(p,c) if horiz else (c,p)
        con,hf,f=contrast(x,y,sx,sy)
        if con>=0.18 and hf<0.04 and f>0.8 and all(ov({'x':x,'y':y,'sx':sx,'sy':sy},b)<0.3 for b in acc+added):
            added.append(dict(kind=ms[0]['kind'],x=round(x,3),y=round(y,3),sx=sx,sy=sy,src='row',contrast=round(float(con),2)))
acc2=acc+added
# drop singletons that are weak
rowmember=set()
for r in rows:
    if len(r['m'])>=2:
        for a in r['m']: rowmember.add(id(a))
final=[]
for a in acc2:
    con,hf,f=contrast(a['x'],a['y'],a['sx'],a['sy'])
    a['contrast']=round(float(con),2)
    if a['src']=='row' or id(a) in rowmember or con>=0.35:
        final.append(a)
import collections
print(len(out),'seg',len(T),'tmpl ->',len(acc),'+',len(added),'row fills -> final',len(final), collections.Counter(a['kind'] for a in final))
acc=final
json.dump(acc,open(W+'light_final_cands.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(18,28)); ax.imshow(np.where(valid,I,np.nan).T,origin='lower',extent=[-14,14,-23,23],cmap='gray')
for a in acc:
    c={'seg':'lime','tmpl':'cyan','row':'magenta'}[a['src']] if not a['kind'].startswith('hvac') else 'orange'
    ax.add_patch(plt.Rectangle((a['x']-a['sx']/2,a['y']-a['sy']/2),a['sx'],a['sy'],fill=False,ec=c,lw=1.4))
ax.grid(alpha=.3); ax.set_xticks(range(-14,15)); ax.set_yticks(range(-23,24)); plt.savefig('lights_det4.png',dpi=40,bbox_inches='tight')
