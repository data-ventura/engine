//! Zarr arrays read over HTTPS from public object stores, one chunk at a time.
//!
//! Reads are async: the caller owns the runtime, so a view can keep many chunk requests in flight
//! and cancel the ones it no longer needs.

use std::ops::Range;
use std::sync::Arc;

use zarrs::array::{Array, ArrayMetadata, ArrayMetadataV2, ArraySubset};
use zarrs::node::{NodePath, meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs::storage::AsyncReadableStorageTraits;
use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::http::HttpBuilder;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::object_store::{ClientOptions, ObjectStore};

use crate::{Error, Result};

type Store = AsyncObjectStore<Box<dyn ObjectStore>>;

/// One array in a Zarr store (format 2 or 3), on disk or reached over HTTPS.
#[derive(Clone)]
pub struct ZarrArray {
    array: Arc<Array<Store>>,
}

impl std::fmt::Debug for ZarrArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrArray")
            .field("shape", &self.shape())
            .field("chunks", &self.chunk_grid())
            .finish()
    }
}

impl ZarrArray {
    /// Opens the array at `path` (such as `/2m_temperature`) in the store at `url`, reading only
    /// its metadata.
    pub async fn open_http(url: &str, path: &str) -> Result<Self> {
        // HTTP/2 measured about twice as fast as HTTP/1.1 per chunk against Google Cloud Storage.
        let options = ClientOptions::new().with_allow_http2();
        let http = HttpBuilder::new()
            .with_url(url)
            .with_client_options(options)
            .build()
            .map_err(zarr_err)?;
        Self::open(Box::new(http), path).await
    }

    /// Opens the array at `path` in a store directory on disk. Read-only: nothing is written.
    pub async fn open_dir(dir: impl AsRef<std::path::Path>, path: &str) -> Result<Self> {
        let fs = LocalFileSystem::new_with_prefix(dir).map_err(zarr_err)?;
        Self::open(Box::new(fs), path).await
    }

    async fn open(store: Box<dyn ObjectStore>, path: &str) -> Result<Self> {
        let store = Arc::new(AsyncObjectStore::new(store));
        // The store's Zarr format isn't known until a metadata key answers, so ask for all three
        // at once: in turn, they cost two more round trips on every open (about 200 ms on a cold
        // connection to Google Cloud Storage).
        let node = NodePath::new(path).map_err(zarr_err)?;
        let keys = [
            meta_key_v3(&node),
            meta_key_v2_array(&node),
            meta_key_v2_attributes(&node),
        ];
        let (v3, v2, attrs) = tokio::join!(
            store.get(&keys[0]),
            store.get(&keys[1]),
            store.get(&keys[2]),
        );
        let metadata = if let Some(json) = v3.map_err(zarr_err)? {
            ArrayMetadata::V3(serde_json::from_slice(&json).map_err(zarr_err)?)
        } else if let Some(json) = v2.map_err(zarr_err)? {
            let mut m: ArrayMetadataV2 = serde_json::from_slice(&json).map_err(zarr_err)?;
            if let Some(json) = attrs.map_err(zarr_err)? {
                m.attributes = serde_json::from_slice(&json).map_err(zarr_err)?;
            }
            ArrayMetadata::V2(m)
        } else {
            return Err(Error::Zarr(format!("no Zarr array at {path}")));
        };
        let array = Array::new_with_metadata(store, path, metadata).map_err(zarr_err)?;
        Ok(Self {
            array: Arc::new(array),
        })
    }

    pub fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    /// Chunks along each dimension.
    pub fn chunk_grid(&self) -> &[u64] {
        self.array.chunk_grid_shape()
    }

    /// Reads and decodes one chunk as `f32`. Chunks missing from the store come back as the fill
    /// value.
    pub async fn chunk_f32(&self, indices: &[u64]) -> Result<Vec<f32>> {
        self.array
            .async_retrieve_chunk::<Vec<f32>>(indices)
            .await
            .map_err(zarr_err)
    }

    /// Reads one chunk and returns only `ranges` of it (relative to the chunk), as `f32` in C
    /// order. The whole chunk is still fetched and, for most codecs, decoded; only the subset is
    /// kept.
    pub async fn chunk_subset_f32(
        &self,
        indices: &[u64],
        ranges: &[Range<u64>],
    ) -> Result<Vec<f32>> {
        self.array
            .async_retrieve_chunk_subset::<Vec<f32>>(indices, &ArraySubset::new_with_ranges(ranges))
            .await
            .map_err(zarr_err)
    }
}

fn zarr_err(e: impl std::fmt::Display) -> Error {
    Error::Zarr(e.to_string())
}
