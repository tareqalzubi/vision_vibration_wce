# Data directory

Do not commit raw medical/simulation datasets into the repository.

Convert each dataset into a manifest CSV with columns:

```text
sequence_id,index,frame_prev,frame_ref,frame_next,vibration,depth_ref,depth_next,pose_ref,pose_next,fx,fy,cx,cy,split
```

- Paths may be absolute or relative to `data.root` in the config.
- Vibration files must contain a six-channel signal of length 240.
- Depth and pose are optional for self-supervised training but required for full evaluation.
- Use sequence-level splits to avoid temporal leakage.
