import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import resolve_path, save_json, sha256
from .download import DATASETS

PROTOCOL = "chronological-linear-3sigma-zscore-v3"

def interpolate_partition(values,fallback):
    result=values.copy();time=np.arange(len(values))
    for node in range(values.shape[1]):
        valid=np.isfinite(values[:,node,0])
        result[:,node,0]=np.interp(time,time[valid],values[valid,node,0]) if valid.any() else fallback[node,0]
    return result


def clean_traffic_flow(flow):
    values=np.asarray(flow,dtype=np.float32).copy()
    if values.ndim!=3 or values.shape[-1]!=1:raise ValueError('Expected [time,nodes,1] flow')
    total=len(values);cuts=[0,int(.6*total),int(.8*total),total]
    if min(np.diff(cuts))<24:raise ValueError('Each chronological partition requires at least 24 rows')
    missing=~np.isfinite(values)|(values<=0)
    train=np.where(missing[:cuts[1]],np.nan,values[:cuts[1]]).astype(np.float64)
    counts=np.isfinite(train).sum(0)
    if np.any(counts==0):raise ValueError('A sensor has no valid positive training observations')
    node_mean=np.nanmean(train,axis=0);node_std=np.nanstd(train,axis=0)
    lower=node_mean-3*node_std;upper=node_mean+3*node_std
    outlier=(~missing)&((values<lower)|(values>upper))
    observed=values.copy();observed[missing|outlier]=np.nan
    fallback=np.nanmean(observed[:cuts[1]],axis=0,dtype=np.float64)
    if not np.isfinite(fallback).all():raise ValueError('3-sigma rejection removed all training observations at a sensor')
    filled=np.concatenate([interpolate_partition(observed[a:b],fallback) for a,b in zip(cuts,cuts[1:])])
    mean=float(filled[:cuts[1]].mean(dtype=np.float64))
    std=max(float(filled[:cuts[1]].std(dtype=np.float64)),1e-6)
    inputs=((filled-mean)/std).astype(np.float32)
    if not np.isfinite(inputs).all():raise FloatingPointError('Nonfinite cleaned inputs')
    return dict(observed=observed,filled=filled,inputs=inputs,cuts=cuts,mean=mean,std=std,
                node_mean=node_mean,node_std=node_std,fallback=fallback,lower=lower,upper=upper,
                missing=missing,outlier=outlier)


def repair_history_edges(inputs,observed,fallback):
    valid=np.isfinite(observed)&(observed>0)
    positions=np.arange(len(inputs))[:,None,None]
    first=np.where(valid,positions,len(inputs)).min(axis=0)
    last=np.where(valid,positions,-1).max(axis=0)
    first_value=np.take_along_axis(inputs,np.minimum(first,len(inputs)-1)[None],axis=0)[0]
    last_value=np.take_along_axis(inputs,np.maximum(last,0)[None],axis=0)[0]
    result=np.where(positions<first,first_value,inputs)
    result=np.where(positions>last,last_value,result)
    return np.where(first[None]==len(inputs),fallback[None],result).astype(np.float32)


def adjacency(raw, name, n):
    mapping = None
    if name == "PEMS03":
        ids = [int(s) for s in (raw / f"{name}.txt").read_text().split()]
        if len(ids) != n or len(set(ids)) != n: raise ValueError("Invalid PEMS03 sensor map")
        mapping = {sensor:i for i,sensor in enumerate(ids)}
    a = np.zeros((n,n),dtype=np.float32)
    with open(raw / f"{name}.csv", newline="") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if not row: continue
            u,v=int(row[0]),int(row[1])
            if mapping is not None: u,v=mapping[u],mapping[v]
            if not (0 <= u < n and 0 <= v < n): raise ValueError(f"Bad sensor index {u},{v}")
            a[u,v] = a[v,u] = 1.0
    np.fill_diagonal(a,0)
    if not a.any(): raise ValueError("Empty road graph")
    return a


def preprocess(name, data_root='data'):
    base=resolve_path(data_root)
    dest=base/'processed/main'/name;dest.mkdir(parents=True,exist_ok=True)
    source=base/'raw'/name/(name+'.npz')
    with np.load(source,allow_pickle=False) as archive:raw=archive['data'].astype(np.float32)
    if list(raw.shape)!=DATASETS[name]['shape']:raise ValueError('Nonstandard dataset shape')
    clean=clean_traffic_flow(raw[...,0:1])
    arrays={'flow':clean['observed'],'filled_flow':clean['filled'],'inputs':clean['inputs'],
            'adjacency':adjacency(base/'raw'/name, name, raw.shape[1]),'history_fallback':((clean['fallback']-clean['mean'])/clean['std']).astype(np.float32)}
    for key,value in arrays.items():
        pending=dest/(key+'.pending.npy');np.save(pending,value);pending.replace(dest/(key+'.npy'))
    np.savez_compressed(dest/'cleaning_statistics.npz',training_node_mean=clean['node_mean'],
                        training_node_std=clean['node_std'],lower_3sigma=clean['lower'],upper_3sigma=clean['upper'],
                        history_fallback_raw=clean['fallback'])
    metadata={'dataset':name,'sensors':raw.shape[1],'time_steps':len(raw),'features':raw.shape[-1],
              'target':'flow','target_channel':0,'sampling_interval_minutes':5,'input_length':12,'output_length':12,
              'preprocessing_protocol':PROTOCOL,'split_ratios':[.6,.2,.2],'split_boundaries':clean['cuts'],
              'split_policy':'chronological 60/20/20',
              'timestamps_in_archive':False,'calendar_features_used':False,'reported_time_range':DATASETS[name]['reported_range'],
              'raw_sha256':sha256(source),'inputs_sha256':sha256(dest/'inputs.npy'),
              'adjacency_sha256':sha256(dest/'adjacency.npy'),'observed_flow_sha256':sha256(dest/'flow.npy'),'mean':clean['mean'],'std':clean['std'],
              'normalization':'z-score on the interpolated train partition',
              'outlier_policy':'per-sensor 3-sigma on train observations; rejected values set missing',
              'interpolation':'linear within each partition, endpoint held outside',
              'history_policy':'input endpoints clamped to observations inside the history window',
              'target_policy':'observed positive flow only; imputed labels masked',
              'missing_definition':'nonfinite or flow<=0',
              'counts_by_split':{label:{'rows':b-a,'missing':int(clean['missing'][a:b].sum()),'outliers':int(clean['outlier'][a:b].sum()),
                                      'valid_targets':int(np.isfinite(clean['observed'][a:b]).sum())}
                                 for label,a,b in zip(['train','val','test'],clean['cuts'],clean['cuts'][1:])}}
    pending=dest/'metadata.pending.json';save_json(pending,metadata);pending.replace(dest/'metadata.json')
    save_json(base/'metadata/main'/(name+'_statistics.json'),metadata)
    return metadata


def processed_dir(config, data_root=None):
    if data_root is not None:
        return resolve_path(data_root) / "processed/main" / config["dataset"]
    return resolve_path(config["data"]["processed_root"]) / config["dataset"]


def dataset_signature(folder):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("preprocessing_protocol") != PROTOCOL:
        raise ValueError("Unsupported preprocessing; run prepare_data.py")
    if not np.isfinite(metadata["mean"]) or not np.isfinite(metadata["std"]) or metadata["std"] <= 0:
        raise ValueError("Invalid normalization statistics")
    signature = {key: metadata[key] for key in ("preprocessing_protocol", "mean", "std", "sensors", "split_boundaries")}
    for filename, key in (
        ("inputs.npy", "inputs_sha256"), ("flow.npy", "observed_flow_sha256"),
        ("adjacency.npy", "adjacency_sha256"), ("history_fallback.npy", "history_fallback_sha256"),
    ):
        digest = sha256(folder / filename)
        if key in metadata and metadata[key] != digest:
            raise ValueError(f"Processed data changed: {filename}; run prepare_data.py again")
        signature[key] = digest
    return signature


class TrafficWindows(Dataset):
    def __init__(self, folder, split, input_length=12, output_length=12):
        self.meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if self.meta.get("preprocessing_protocol") != PROTOCOL:
            raise ValueError("Unsupported preprocessing; run prepare_data.py")
        self.x = np.load(folder / "inputs.npy", mmap_mode="r", allow_pickle=False)
        self.y = np.load(folder / "flow.npy", mmap_mode="r", allow_pickle=False)
        self.fallback = np.load(folder / "history_fallback.npy", allow_pickle=False)
        self.input_length, self.output_length = input_length, output_length
        if self.x.shape != self.y.shape or self.x.ndim != 3 or self.x.shape[1:] != (self.meta["sensors"], 1):
            raise ValueError("Processed input and target shapes do not match metadata")
        boundaries = self.meta["split_boundaries"]
        if len(boundaries) != 4 or boundaries[0] != 0 or boundaries[-1] != len(self.x) or any(
            end <= start for start, end in zip(boundaries, boundaries[1:])
        ):
            raise ValueError("Invalid chronological split boundaries")
        index = {"train": 0, "val": 1, "test": 2}[split]
        self.start, self.end = boundaries[index:index + 2]
        self.length = self.end - self.start - input_length - output_length + 1
        if self.length < 1:
            raise ValueError("No complete forecasting windows in partition")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        start = self.start + index
        middle = start + self.input_length
        history = repair_history_edges(self.x[start:middle].copy(), self.y[start:middle], self.fallback)
        target = self.y[middle:middle + self.output_length].copy()
        return torch.from_numpy(history), torch.from_numpy(target)


def loaders(config, device, data_root=None, splits=("train", "val", "test")):
    folder = processed_dir(config, data_root)
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    return {
        split: DataLoader(
            TrafficWindows(folder, split, config["data"]["input_length"], config["data"]["output_length"]),
            batch_size=config["training"]["batch_size"], shuffle=split == "train", drop_last=False,
            num_workers=0, generator=generator, pin_memory=device.type == "cuda",
        )
        for split in splits
    }
