from .replica_dataset import ReplicaDataset
from .mp3d_dataset import Mp3dDataset

__datasets__ = {
    'replica': ReplicaDataset,
    'mp3d': Mp3dDataset
}
