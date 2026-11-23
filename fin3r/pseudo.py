"""Depth and camera targets from a fixed 16-view RGB context."""
import torch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def labels(model,images,n):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(images[None])
    z=pred['depth'][0,...,0].float();conf=pred['depth_conf'][0].float()
    E,K=pose_encoding_to_extri_intri(pred['pose_enc'].float(),images.shape[-2:])
    K=K[0]; h,w=z.shape[-2:]
    world_to_cam=torch.eye(4,device='cuda').repeat(len(z),1,1);world_to_cam[:,:3]=E[0]
    cam_to_world=torch.linalg.inv(world_to_cam)
    if not all(torch.isfinite(a).all() for a in (z,conf,K,cam_to_world)) or (K[:,0,0]<=0).any() or (K[:,1,1]<=0).any():
        raise ValueError('invalid teacher prediction')
    # Color is used to mask black padding in the processed CO3D images.
    valid=(z>0)&(images.amax(1)>3/255)
    limits=torch.stack([torch.quantile(c[m],.5) if m.any() else c.max()+1 for c,m in zip(conf,valid)])
    reliable=valid&(conf>=limits[:,None,None])
    med_depth=torch.median(z[reliable]) if reliable.any() else z.new_tensor(0)
    baseline=torch.pdist(cam_to_world[:,:3,3]); span=baseline.median()/med_depth.clamp_min(1e-8)
    if span<.002 or not torch.isfinite(span): raise ValueError('degenerate teacher geometry')
    y,x=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
    pix=torch.stack((x,y,torch.ones_like(x))).flatten(1).float()
    depth=[];fractions=[]
    for i in range(n):
        local=torch.linalg.solve(K[i],pix)*z[i].reshape(1,-1)
        rel=world_to_cam@cam_to_world[i]
        xyz=rel[:,:3,:3]@local+rel[:,:3,3,None]
        uv=K@xyz; denom=uv[:,2].clamp_min(1e-8)
        u=(uv[:,0]/denom).round().long();v=(uv[:,1]/denom).round().long()
        inside=(xyz[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        flat=v.clamp(0,h-1)*w+u.clamp(0,w-1)
        other_z=z.flatten(1).gather(1,flat)
        agreement=inside&reliable.flatten(1).gather(1,flat)&((other_z-xyz[:,2]).abs()<=.05*other_z)
        agreement &= (rel[:,:3,3].norm(dim=1)>.0001*med_depth)[:,None]
        agreement[i]=False
        keep=reliable[i]&(agreement.sum(0).reshape(h,w)>=2)
        f=float(keep.float().mean())
        if f<.03:raise ValueError('less than 3% verified pixels in a target')
        fractions.append(f);depth.append(torch.where(keep,z[i],0).cpu().numpy())
    return depth,K[:n].cpu().numpy(),cam_to_world[:n].cpu().numpy(),fractions,float(span)
